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


def run_comparison(images_dir, max_images, N, T, d, seed, out_path):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    if max_images:
        paths = paths[:max_images]
    print(f"Found {len(paths)} images to process.")

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
        enh_result = enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)

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

    def paired_report(name, std_vals, enh_vals):
        std_vals = np.array(std_vals)
        enh_vals = np.array(enh_vals)
        diff = enh_vals - std_vals
        try:
            stat, p_value = wilcoxon(std_vals, enh_vals)
        except ValueError as e:
            stat, p_value = None, None
            print(f"  ({name}: Wilcoxon test could not run -- {e})")

        print(f"=== {name} ===")
        print(f"  Standard SMA -- mean: {std_vals.mean():.4f}, std: {std_vals.std():.4f}")
        print(f"  Enhanced SMA -- mean: {enh_vals.mean():.4f}, std: {enh_vals.std():.4f}")
        print(f"  Mean gain (Enhanced - Standard): {diff.mean():+.4f} (std {diff.std():.4f})")
        print(f"  ESMA better in {np.sum(diff > 0)}/{len(diff)} images "
              f"({100*np.mean(diff > 0):.1f}%)")
        if p_value is not None:
            sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
            print(f"  Wilcoxon signed-rank p-value: {p_value:.4f} -- {sig}")
        print()
        return {
            "standard_mean": float(std_vals.mean()), "standard_std": float(std_vals.std()),
            "enhanced_mean": float(enh_vals.mean()), "enhanced_std": float(enh_vals.std()),
            "mean_gain": float(diff.mean()), "gain_std": float(diff.std()),
            "pct_images_esma_better": float(100 * np.mean(diff > 0)),
            "wilcoxon_p_value": float(p_value) if p_value is not None else None,
        }

    summary = {
        "n_images": len(entropy_std),
        "sma_params": {"N": N, "T": T, "d": d},
        "kapur_entropy": paired_report("Kapur's Entropy", entropy_std, entropy_enh),
        "psnr": paired_report("PSNR", psnr_std, psnr_enh) if psnr_std else None,
        "ssim": paired_report("SSIM", ssim_std, ssim_enh),
        "runtime_sec": paired_report("Runtime (s)", runtime_std, runtime_enh),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./full_image_comparison.json")
    args = parser.parse_args()
    run_comparison(args.images_dir, args.max_images, args.N, args.T, args.d, args.seed, args.out)