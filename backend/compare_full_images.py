"""
compare_full_images.py
------------------------
Runs Standard SMA and ESMA on FULL panoramic OPG images (not per-tooth
crops -- this is the actual Chapter 3/4 core comparison your thesis is
built around) across a whole folder of images, and reports paired
statistical significance (Wilcoxon signed-rank test) on Kapur's
entropy, PSNR, and SSIM.

Why this matters after the DENTEX crop-level result: small, low-
dimensional crops showed no advantage for ESMA's multi-leader guidance
(if anything, more leaders made things worse). Full panoramic images
have far richer, more complex, multi-modal histograms (many
overlapping anatomical structures) -- exactly the kind of landscape
ESMA's anti-stagnation mechanisms were designed for. This script tests
that directly, at scale, with a proper statistical test instead of a
single manual comparison.

You can point this at the DENTEX 'xrays' folder directly -- those are
full panoramic OPG images, no separate dataset needed.

Usage:
    python compare_full_images.py \
        --images_dir ..\\datasets\\training_data\\training_data\\quadrant-enumeration-disease\\xrays \
        --max_images 100 --sma_population 30 --sma_iterations 50 --d 4
"""

import argparse
import glob
import os
import random
import time

import cv2
import numpy as np
from scipy.stats import wilcoxon

from sma_algorithms import (
    apply_thresholds,
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)
from metrics import compute_psnr, compute_ssim


def run_comparison(images_dir, max_images, N, T, d, seed, out_path, esma_params=None, shuffle=True):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    if shuffle:
        # shuffled by --seed so a different seed reliably gives a
        # DIFFERENT sample -- important when verifying a candidate found
        # during hyperparameter tuning, so this run doesn't reuse any of
        # the images that tuning already looked at
        random.Random(seed).shuffle(paths)
    if max_images:
        paths = paths[:max_images]
    print(f"Found {len(paths)} images to process.")
    esma_params = esma_params or {}
    if esma_params:
        print(f"Using custom ESMA hyperparameters: {esma_params}")

    entropy_std, entropy_enh = [], []
    psnr_std, psnr_enh = [], []
    ssim_std, ssim_enh = [], []
    runtime_std, runtime_enh = [], []
    used_files = []

    t_start = time.perf_counter()
    for i, path in enumerate(paths):
        if i > 0 and i % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  ...{i}/{len(paths)} images done ({elapsed:.1f}s elapsed)", flush=True)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [skip] could not read '{os.path.basename(path)}'")
            continue
        image = autocrop_black_borders(image)
        prob = compute_histogram_prob(image)

        std_result = standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        enh_result = enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **esma_params)

        std_seg = apply_thresholds(image, std_result["thresholds"])
        enh_seg = apply_thresholds(image, enh_result["thresholds"])

        entropy_std.append(std_result["fitness"])
        entropy_enh.append(enh_result["fitness"])
        runtime_std.append(std_result["runtime_sec"])
        runtime_enh.append(enh_result["runtime_sec"])

        sp = compute_psnr(image, std_seg)
        ep = compute_psnr(image, enh_seg)
        if sp != float("inf") and ep != float("inf"):
            psnr_std.append(sp)
            psnr_enh.append(ep)

        ssim_std.append(compute_ssim(image, std_seg))
        ssim_enh.append(compute_ssim(image, enh_seg))
        used_files.append(os.path.basename(path))

    print(f"\nProcessed {len(entropy_std)} images successfully "
          f"({time.perf_counter() - t_start:.1f}s total).\n")

    def paired_report(name, std_vals, enh_vals, lower_is_better=False):
        std_vals = np.array(std_vals)
        enh_vals = np.array(enh_vals)
        diff = enh_vals - std_vals
        try:
            stat, p_value = wilcoxon(std_vals, enh_vals)
        except ValueError as e:
            stat, p_value = None, None
            print(f"  ({name}: Wilcoxon test could not run -- {e})")

        # "better" direction depends on the metric: for entropy/PSNR/SSIM,
        # higher is better (ESMA better = diff > 0). For runtime, LOWER
        # is better (ESMA better = diff < 0) -- using the same diff>0
        # count for both was reporting an inverted percentage for runtime.
        esma_better_mask = (diff < 0) if lower_is_better else (diff > 0)
        pct_better = 100 * np.mean(esma_better_mask)

        print(f"=== {name} ({'lower is better' if lower_is_better else 'higher is better'}) ===")
        print(f"  Standard SMA -- mean: {std_vals.mean():.4f}, std: {std_vals.std():.4f}")
        print(f"  Enhanced SMA -- mean: {enh_vals.mean():.4f}, std: {enh_vals.std():.4f}")
        print(f"  Mean gain (Enhanced - Standard): {diff.mean():+.4f} (std {diff.std():.4f})")
        print(f"  ESMA better in {int(esma_better_mask.sum())}/{len(diff)} images "
              f"({pct_better:.1f}%)")
        if p_value is not None:
            sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
            print(f"  Wilcoxon signed-rank p-value: {p_value:.4f} -- {sig}")
        print()
        return {
            "standard_mean": float(std_vals.mean()), "standard_std": float(std_vals.std()),
            "enhanced_mean": float(enh_vals.mean()), "enhanced_std": float(enh_vals.std()),
            "mean_gain": float(diff.mean()), "gain_std": float(diff.std()),
            "pct_images_esma_better": float(pct_better),
            "wilcoxon_p_value": float(p_value) if p_value is not None else None,
        }

    summary = {
        "n_images": len(entropy_std),
        "sma_params": {"N": N, "T": T, "d": d},
        "kapur_entropy": paired_report("Kapur's Entropy", entropy_std, entropy_enh),
        "psnr": paired_report("PSNR", psnr_std, psnr_enh) if psnr_std else None,
        "ssim": paired_report("SSIM", ssim_std, ssim_enh),
        "runtime_sec": paired_report("Runtime (s)", runtime_std, runtime_enh, lower_is_better=True),
    }

    import json
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--sma_population", type=int, default=30, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=50, dest="T")
    parser.add_argument("--d", type=int, default=4, help="Threshold levels -- match your Chapter 3")
    parser.add_argument("--seed", type=int, default=42,
                         help="Also controls which images are sampled (shuffled by this seed) -- "
                              "use a fresh seed to guarantee a non-overlapping verification set")
    parser.add_argument("--no_shuffle", action="store_true",
                         help="Use sorted file order instead of shuffling (old behavior)")
    parser.add_argument("--esma_alpha", type=float, default=None)
    parser.add_argument("--esma_beta", type=float, default=None)
    parser.add_argument("--esma_gamma", type=float, default=None)
    parser.add_argument("--esma_delta", type=float, default=None)
    parser.add_argument("--esma_h", type=int, default=None)
    parser.add_argument("--esma_k", type=int, default=None)
    parser.add_argument("--esma_adaptive_k", action="store_true",
                         help="Use diversity-driven adaptive leader count (extension beyond literal Algorithm 3.1 -- document in Chapter 3 if used)")
    parser.add_argument("--out", default="./full_image_comparison.json")
    args = parser.parse_args()

    esma_params = {}
    if args.esma_alpha is not None:
        esma_params["alpha"] = args.esma_alpha
    if args.esma_beta is not None:
        esma_params["beta"] = args.esma_beta
    if args.esma_gamma is not None:
        esma_params["gamma"] = args.esma_gamma
    if args.esma_delta is not None:
        esma_params["delta"] = args.esma_delta
    if args.esma_h is not None:
        esma_params["h"] = args.esma_h
    if args.esma_k is not None:
        esma_params["k"] = args.esma_k
    if args.esma_adaptive_k:
        esma_params["adaptive_k"] = True

    run_comparison(args.images_dir, args.max_images, args.N, args.T, args.d, args.seed,
                    args.out, esma_params=esma_params, shuffle=not args.no_shuffle)
    args = parser.parse_args()
    run_comparison(args.images_dir, args.max_images, args.N, args.T, args.d, args.seed, args.out)