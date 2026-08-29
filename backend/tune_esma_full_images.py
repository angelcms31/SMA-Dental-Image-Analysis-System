"""
tune_esma_full_images.py
--------------------------
Same idea as tune_esma_hyperparams.py, but targets the ACTUAL Chapter 4
task: full panoramic OPG images, not per-tooth crops. The crop-level
tuning found k=2 (fewer leaders) worked best on tiny, simple crops --
that result does not necessarily transfer to full images, which have
much richer, more complex, multi-modal histograms. This script finds
good ESMA hyperparameters specifically for that regime.

EFFICIENCY NOTE: Standard SMA's result does not depend on ESMA's
hyperparameters at all, so it is computed ONCE per tuning image
(cached), not re-run for every candidate -- this roughly halves the
runtime compared to naively re-running both algorithms per candidate.

Usage:
    python tune_esma_full_images.py \
        --images_dir ..\\datasets\\training_data\\training_data\\quadrant-enumeration-disease\\xrays \
        --tune_images 10 --verify_images 10 --n_candidates 15 \
        --sma_population 15 --sma_iterations 100 --d 4 \
        --optimize_for entropy
"""

import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np

from sma_algorithms import (
    apply_thresholds,
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)
from metrics import compute_psnr, compute_ssim


def load_full_images(images_dir, n_total, seed, exclude_paths=None):
    exclude_paths = exclude_paths or set()
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    paths = [p for p in paths if p not in exclude_paths]
    rng = random.Random(seed)
    rng.shuffle(paths)

    images, used_paths = [], []
    for p in paths:
        if len(images) >= n_total:
            break
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] could not read '{os.path.basename(p)}'")
            continue
        img = autocrop_black_borders(img)
        images.append(img)
        used_paths.append(p)
    return images, set(used_paths)


def precompute_baseline(images, N, T, d, seed):
    """Standard SMA result per image, computed once and reused for every candidate."""
    baseline = []
    for img in images:
        prob = compute_histogram_prob(img)
        std_result = standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        std_seg = apply_thresholds(img, std_result["thresholds"])
        baseline.append({
            "prob": prob,
            "entropy": std_result["fitness"],
            "psnr": compute_psnr(img, std_seg),
            "ssim": compute_ssim(img, std_seg),
        })
    return baseline


def evaluate_candidate(images, baseline, params, N, T, d, seed):
    entropy_gains, psnr_gains, ssim_gains = [], [], []
    for img, base in zip(images, baseline):
        enh_result = enhanced_sma(base["prob"], d=d, N=N, T=T, lb=0, ub=255, seed=seed, **params)
        enh_seg = apply_thresholds(img, enh_result["thresholds"])

        entropy_gains.append(enh_result["fitness"] - base["entropy"])

        enh_psnr = compute_psnr(img, enh_seg)
        if enh_psnr != float("inf") and base["psnr"] != float("inf"):
            psnr_gains.append(enh_psnr - base["psnr"])

        ssim_gains.append(compute_ssim(img, enh_seg) - base["ssim"])

    return {
        "entropy_gain_mean": float(np.mean(entropy_gains)),
        "entropy_gain_std": float(np.std(entropy_gains)),
        "psnr_gain_mean": float(np.mean(psnr_gains)) if psnr_gains else None,
        "ssim_gain_mean": float(np.mean(ssim_gains)),
        "ssim_gain_std": float(np.std(ssim_gains)),
    }


def random_search_space(rng):
    return {
        "alpha": rng.choice([0.05, 0.10, 0.15, 0.20]),
        "beta": rng.choice([0.05, 0.10, 0.15, 0.20]),
        "gamma": rng.choice([0.05, 0.10, 0.15, 0.20]),
        "delta": rng.choice([0.05, 0.10, 0.15, 0.20]),
        "h": rng.choice([3, 5, 8]),
        "k": rng.choice([2, 3, 5]),
    }


def score_for_ranking(result, optimize_for):
    if optimize_for == "entropy":
        return result["entropy_gain_mean"]
    if optimize_for == "psnr":
        return result["psnr_gain_mean"] if result["psnr_gain_mean"] is not None else -999
    if optimize_for == "ssim":
        return result["ssim_gain_mean"]
    # combined: normalize-ish by just summing signed gains (rough, but usable for ranking)
    psnr_part = result["psnr_gain_mean"] if result["psnr_gain_mean"] is not None else 0
    return result["entropy_gain_mean"] + 0.1 * psnr_part + result["ssim_gain_mean"]


def main(images_dir, tune_n, verify_n, n_candidates, N, T, d, seed, optimize_for, out_path):
    rng = random.Random(seed)

    print(f"Loading {tune_n} images for tuning...")
    tune_images, tune_paths = load_full_images(images_dir, tune_n, seed)
    print(f"Loading {verify_n} images for verification (disjoint from tuning)...")
    verify_images, _ = load_full_images(images_dir, verify_n, seed + 999, exclude_paths=tune_paths)

    print(f"\nComputing Standard SMA baseline on {len(tune_images)} tuning images "
          f"(N={N}, T={T}, d={d})...")
    t0 = time.perf_counter()
    tune_baseline = precompute_baseline(tune_images, N, T, d, seed)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print(f"\nSearching {n_candidates} ESMA hyperparameter combinations "
          f"(optimizing for: {optimize_for})...\n")

    results = []
    for i in range(n_candidates):
        params = random_search_space(rng)
        t0 = time.perf_counter()
        scores = evaluate_candidate(tune_images, tune_baseline, params, N, T, d, seed)
        elapsed = time.perf_counter() - t0
        results.append({"params": params, **scores})
        print(f"[{i+1}/{n_candidates}] {params} "
              f"-> entropy={scores['entropy_gain_mean']:+.4f} "
              f"psnr={scores['psnr_gain_mean']:+.4f} "
              f"ssim={scores['ssim_gain_mean']:+.4f} "
              f"({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda r: score_for_ranking(r, optimize_for), reverse=True)
    print(f"\n=== Top 5 candidates on TUNING set (optimizing for {optimize_for}) ===")
    for r in results[:5]:
        print(f"  entropy={r['entropy_gain_mean']:+.4f} psnr={r['psnr_gain_mean']:+.4f} "
              f"ssim={r['ssim_gain_mean']:+.4f} params={r['params']}")

    best = results[0]
    print(f"\nBest candidate on tuning set: {best['params']}")
    print("Verifying on a SEPARATE held-out image set (not used for tuning)...")
    verify_baseline = precompute_baseline(verify_images, N, T, d, seed + 999)
    verify_scores = evaluate_candidate(verify_images, verify_baseline, best["params"], N, T, d, seed + 999)
    print(f"Verification result: entropy={verify_scores['entropy_gain_mean']:+.4f} "
          f"psnr={verify_scores['psnr_gain_mean']:+.4f} "
          f"ssim={verify_scores['ssim_gain_mean']:+.4f}")

    tuning_score = score_for_ranking(best, optimize_for)
    verify_score = score_for_ranking(verify_scores, optimize_for)
    if verify_score <= 0 <= tuning_score:
        print("\nNOTE: the gain did not hold up on the verification set -- likely noise "
              "from the tuning search, not a genuine improvement. Report both numbers.")

    with open(out_path, "w") as f:
        json.dump({
            "optimize_for": optimize_for,
            "best_params": best["params"],
            "tuning_set_result": {k: v for k, v in best.items() if k != "params"},
            "verification_set_result": verify_scores,
            "all_candidates": results,
        }, f, indent=2)
    print(f"\nSaved full search results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--tune_images", type=int, default=10)
    parser.add_argument("--verify_images", type=int, default=10)
    parser.add_argument("--n_candidates", type=int, default=15)
    parser.add_argument("--sma_population", type=int, default=15, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=100, dest="T")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize_for", choices=["entropy", "psnr", "ssim", "combined"], default="entropy")
    parser.add_argument("--out", default="./esma_full_image_tuning.json")
    args = parser.parse_args()
    main(args.images_dir, args.tune_images, args.verify_images, args.n_candidates,
         args.N, args.T, args.d, args.seed, args.optimize_for, args.out)