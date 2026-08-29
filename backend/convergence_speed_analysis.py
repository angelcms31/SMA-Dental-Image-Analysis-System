"""
convergence_speed_analysis.py
-------------------------------
Measures "convergence speed" the way metaheuristics literature usually
means it: number of ITERATIONS needed to reach a given fraction of the
final best fitness -- not wall-clock seconds. This matters because
ESMA's per-iteration cost is higher (every agent runs the full
multi-leader computation every time, per the literal Algorithm 3.1,
with no cheap "skip" branch), so wall-clock runtime alone can make
ESMA look slower even if it reaches a good solution in FEWER
iterations, which is what Objective 3 ("accelerating convergence
speed") is actually about.

For each image, both algorithms' convergence curves (best fitness per
iteration, already returned by standard_sma()/enhanced_sma()) are
compared: for each of several thresholds (90%, 95%, 99% of that run's
own final fitness), find the first iteration at which the curve
reaches that fraction. Paired across images, with a Wilcoxon
signed-rank test per threshold.

Usage:
    python convergence_speed_analysis.py \
        --images_dir ..\\datasets\\training_data\\training_data\\quadrant-enumeration-disease\\xrays \
        --max_images 50 --sma_population 15 --sma_iterations 150 --d 4
"""

import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np
from scipy.stats import wilcoxon

from sma_algorithms import (
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)

THRESHOLDS = [0.90, 0.95, 0.99]


def iterations_to_threshold(convergence, frac):
    """First iteration index (0-based, matching the convergence list) at
    which the curve reaches frac * final_value. Since bF is greedily
    non-decreasing, this is well-defined."""
    final_value = convergence[-1]
    target = frac * final_value
    for i, v in enumerate(convergence):
        if v >= target:
            return i
    return len(convergence) - 1  # never reached exactly -- shouldn't happen


def run_analysis(images_dir, max_images, N, T, d, seed, out_path):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    random.Random(seed).shuffle(paths)
    if max_images:
        paths = paths[:max_images]
    print(f"Found {len(paths)} images to process.")

    results = {thr: {"standard": [], "enhanced": []} for thr in THRESHOLDS}
    t_start = time.perf_counter()
    n_done = 0

    for i, path in enumerate(paths):
        if i > 0 and i % 10 == 0:
            print(f"  ...{i}/{len(paths)} images done "
                  f"({time.perf_counter()-t_start:.1f}s elapsed)", flush=True)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [skip] could not read '{os.path.basename(path)}'")
            continue
        image = autocrop_black_borders(image)
        prob = compute_histogram_prob(image)

        std_result = standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        enh_result = enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)

        for thr in THRESHOLDS:
            results[thr]["standard"].append(
                iterations_to_threshold(std_result["convergence"], thr))
            results[thr]["enhanced"].append(
                iterations_to_threshold(enh_result["convergence"], thr))
        n_done += 1

    print(f"\nProcessed {n_done} images successfully "
          f"({time.perf_counter()-t_start:.1f}s total).\n")

    summary = {"n_images": n_done, "sma_params": {"N": N, "T": T, "d": d}, "thresholds": {}}

    for thr in THRESHOLDS:
        std_iters = np.array(results[thr]["standard"])
        enh_iters = np.array(results[thr]["enhanced"])
        diff = enh_iters - std_iters  # negative = ESMA needed FEWER iterations (faster convergence)

        try:
            stat, p_value = wilcoxon(std_iters, enh_iters)
        except ValueError as e:
            p_value = None
            print(f"  (threshold {thr}: Wilcoxon test could not run -- {e})")

        pct_faster = 100 * np.mean(diff < 0)

        print(f"=== Iterations to reach {int(thr*100)}% of final fitness ===")
        print(f"  Standard SMA -- mean: {std_iters.mean():.2f} iters, std: {std_iters.std():.2f}")
        print(f"  Enhanced SMA -- mean: {enh_iters.mean():.2f} iters, std: {enh_iters.std():.2f}")
        print(f"  Mean difference (Enhanced - Standard): {diff.mean():+.2f} iterations")
        print(f"  ESMA reached threshold in FEWER iterations in {int(np.sum(diff < 0))}/{len(diff)} "
              f"images ({pct_faster:.1f}%)")
        if p_value is not None:
            sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
            print(f"  Wilcoxon signed-rank p-value: {p_value:.4f} -- {sig}")
        print()

        summary["thresholds"][f"{int(thr*100)}pct"] = {
            "standard_mean_iters": float(std_iters.mean()),
            "standard_std_iters": float(std_iters.std()),
            "enhanced_mean_iters": float(enh_iters.mean()),
            "enhanced_std_iters": float(enh_iters.std()),
            "mean_diff_iters": float(diff.mean()),
            "pct_images_esma_faster": float(pct_faster),
            "wilcoxon_p_value": float(p_value) if p_value is not None else None,
        }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--sma_population", type=int, default=15, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=150, dest="T")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./convergence_speed_analysis.json")
    args = parser.parse_args()
    run_analysis(args.images_dir, args.max_images, args.N, args.T, args.d, args.seed, args.out)