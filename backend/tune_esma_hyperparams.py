"""
tune_esma_hyperparams.py
--------------------------
Searches for good ESMA control-parameter values (alpha, beta, gamma,
delta, h, k) -- the ones your thesis draft never pinned down numerically
(the "Algorithm 3.1" box was never inserted into the PDF, see the
sma_algorithms.py module docstring). This is legitimate hyperparameter
tuning, not p-hacking: the search target is Kapur's entropy itself
(what SMA/ESMA are actually designed to maximize), evaluated on a
SEPARATE tuning subset from whatever subset you use for final
reporting, and every candidate is scored the same way.

WHAT THIS DOES: for each candidate hyperparameter combination, runs
ESMA on a fixed tuning subset of crops and compares its Kapur's
entropy against Standard SMA's entropy on the SAME crops (paired
comparison, same images, same seed). Reports the combination with the
best mean entropy gain -- and its consistency (std across crops) --
so you can lock in real values instead of my earlier placeholder
defaults.

WHAT THIS DOES NOT DO: it does not guarantee ESMA "wins." If every
candidate comes back with ~0 gain, that is itself the honest answer --
report it. Re-verify whatever combination looks best on a *different*
random subset (--verify_subset) before finalizing it for Chapter 4,
since picking the single best result out of many candidates on the
SAME data is exactly the kind of overfitting-the-search you want to
avoid.

Usage:
    python tune_esma_hyperparams.py \
        --json train_quadrant_enumeration_disease.json \
        --images_dir ./xrays \
        --tune_samples 200 --verify_samples 200 --n_candidates 25
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from sma_algorithms import (
    apply_thresholds,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)
from metrics import compute_psnr, compute_ssim


def sample_crops(json_path, images_dir, n_total, seed, exclude_ann_ids=None):
    """
    Loads only as many crops as needed by shuffling the ANNOTATION LIST
    first (before touching any image file), then decoding images one at
    a time until n_total usable crops are collected -- unlike naively
    loading every crop in the whole dataset and discarding most of it,
    which is what made this painfully slow (and prone to looking like
    it "hung") on the full ~3,500-annotation dataset.

    Returns (crops, ann_ids_used) so a second call can pass ann_ids_used
    as exclude_ann_ids to guarantee the tuning and verification subsets
    never overlap.
    """
    exclude_ann_ids = exclude_ann_ids or set()
    with open(json_path) as f:
        data = json.load(f)

    id_to_label = {c["id"]: c["name"] for c in data["categories_3"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    anns = list(data["annotations"])
    rng = random.Random(seed)
    rng.shuffle(anns)

    crops = []
    ann_ids_used = set()
    n_examined = 0
    for ann in anns:
        if len(crops) >= n_total:
            break
        if ann["id"] in exclude_ann_ids:
            continue
        n_examined += 1
        if n_examined % 100 == 0:
            print(f"  ...examined {n_examined} annotations, "
                  f"{len(crops)}/{n_total} usable crops found so far", flush=True)

        filename = id_to_file.get(ann["image_id"])
        if filename is None:
            continue
        img_path = os.path.join(images_dir, filename)

        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            print(f"  [skip] could not read '{filename}' ({e.__class__.__name__}: {e})")
            continue
        if image is None:
            continue

        h, w = image.shape
        x, y, bw, bh = ann["bbox"]
        pad = 8
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(w, int(x + bw + pad)), min(h, int(y + bh + pad))
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue

        crop = image[y0:y1, x0:x1]
        label = id_to_label[ann["category_id_3"]]
        crops.append((crop, label))
        ann_ids_used.add(ann["id"])

    if len(crops) < n_total:
        print(f"  NOTE: only found {len(crops)} usable crops out of the "
              f"{n_total} requested (ran out of annotations).")
    return crops, ann_ids_used


def evaluate_params(crops, params, N, T, d, seed):
    """
    Runs Standard SMA once (fixed baseline) and ESMA with the given
    params, both on the same crops/seed, and returns paired gains.
    """
    entropy_gains, psnr_gains, ssim_gains = [], [], []
    for i, (crop, _label) in enumerate(crops):
        prob = compute_histogram_prob(crop)
        std_result = standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        enh_result = enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **params)

        std_seg = apply_thresholds(crop, std_result["thresholds"])
        enh_seg = apply_thresholds(crop, enh_result["thresholds"])

        entropy_gains.append(enh_result["fitness"] - std_result["fitness"])

        std_psnr = compute_psnr(crop, std_seg)
        enh_psnr = compute_psnr(crop, enh_seg)
        if std_psnr != float("inf") and enh_psnr != float("inf"):
            psnr_gains.append(enh_psnr - std_psnr)

        ssim_gains.append(compute_ssim(crop, enh_seg) - compute_ssim(crop, std_seg))

    return {
        "entropy_gain_mean": float(np.mean(entropy_gains)),
        "entropy_gain_std": float(np.std(entropy_gains)),
        "psnr_gain_mean": float(np.mean(psnr_gains)) if psnr_gains else None,
        "ssim_gain_mean": float(np.mean(ssim_gains)),
        "n_crops": len(crops),
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


def main(json_path, images_dir, tune_samples, verify_samples, n_candidates, N, T, d, seed, out_path):
    rng = random.Random(seed)

    print(f"Sampling {tune_samples} crops for tuning...")
    tune_crops, tune_ann_ids = sample_crops(json_path, images_dir, tune_samples, seed)
    print(f"Sampling {verify_samples} crops for verification "
          f"(guaranteed disjoint from the tuning set)...")
    verify_crops, _ = sample_crops(json_path, images_dir, verify_samples, seed + 999,
                                    exclude_ann_ids=tune_ann_ids)

    print(f"\nSearching {n_candidates} random ESMA hyperparameter combinations "
          f"on {len(tune_crops)} tuning crops (N={N}, T={T}, d={d})...\n")

    results = []
    for i in range(n_candidates):
        params = random_search_space(rng)
        t0 = time.perf_counter()
        scores = evaluate_params(tune_crops, params, N, T, d, seed)
        elapsed = time.perf_counter() - t0
        results.append({"params": params, **scores})
        print(f"[{i+1}/{n_candidates}] {params} "
              f"-> entropy_gain={scores['entropy_gain_mean']:+.4f} "
              f"(std {scores['entropy_gain_std']:.4f}), "
              f"ssim_gain={scores['ssim_gain_mean']:+.4f} "
              f"({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda r: r["entropy_gain_mean"], reverse=True)
    print("\n=== Top 5 candidates on TUNING set (by mean entropy gain) ===")
    for r in results[:5]:
        print(f"  entropy_gain={r['entropy_gain_mean']:+.4f} "
              f"ssim_gain={r['ssim_gain_mean']:+.4f} params={r['params']}")

    best = results[0]
    print(f"\nBest candidate on tuning set: {best['params']}")
    print("Verifying on a SEPARATE held-out subset (not used for tuning)...")
    verify_scores = evaluate_params(verify_crops, best["params"], N, T, d, seed + 999)
    print(f"Verification result: entropy_gain={verify_scores['entropy_gain_mean']:+.4f} "
          f"(std {verify_scores['entropy_gain_std']:.4f}), "
          f"ssim_gain={verify_scores['ssim_gain_mean']:+.4f}, "
          f"psnr_gain={verify_scores['psnr_gain_mean']}")

    if verify_scores["entropy_gain_mean"] <= 0:
        print("\nNOTE: the gain did not hold up on the verification subset. "
              "This is a real, reportable outcome -- it means the tuning-set "
              "result was likely noise, not a genuine, generalizable "
              "improvement. Report both numbers honestly rather than only "
              "the tuning-set result.")

    with open(out_path, "w") as f:
        json.dump({
            "best_params": best["params"],
            "tuning_set_result": {k: v for k, v in best.items() if k != "params"},
            "verification_set_result": verify_scores,
            "all_candidates": results,
        }, f, indent=2)
    print(f"\nSaved full search results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--tune_samples", type=int, default=200)
    parser.add_argument("--verify_samples", type=int, default=200)
    parser.add_argument("--n_candidates", type=int, default=25)
    parser.add_argument("--sma_population", type=int, default=20, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=40, dest="T")
    parser.add_argument("--d", type=int, default=4, help="Threshold levels -- match what your Chapter 3 reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./esma_tuning_results.json")
    args = parser.parse_args()
    main(args.json, args.images_dir, args.tune_samples, args.verify_samples,
         args.n_candidates, args.N, args.T, args.d, args.seed, args.out)