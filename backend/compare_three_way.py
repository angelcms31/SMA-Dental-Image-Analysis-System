"""
compare_three_way.py
---------------------
Three-way benchmark on FULL OPG images: Standard SMA vs the original ESMA
(v1, enhanced_sma) vs ESMA v2 (enhanced_sma_v2), run on identical images,
seeds and settings. Reports, per algorithm and per algorithm PAIR:

  * Kapur's entropy, PSNR, SSIM, runtime, iterations used (v2 may stop early)
  * the EXACT Kapur optimum of every image (kapur_optimal_thresholds, dynamic
    programming) -> optimality gap, share of images where the optimizer
    reaches the true optimum, and PSNR / SSIM *at* that optimum (what
    maximizing Kapur's entropy perfectly would give)
  * paired Wilcoxon signed-rank tests (same methodology as
    compare_full_images.py) for every metric and every algorithm pair
  * per-image results in the output JSON, so figures and further statistics
    never require a re-run

Development / hold-out discipline
    --subset dev      first --dev_n files of the seed-shuffled list: used to
                      choose ESMA v2 defaults (ablate_esma_v2.py)
    --subset holdout  the remaining files: used for REPORTED numbers
    --subset all      the full seed-shuffled list, i.e. exactly the order
                      compare_full_images.py uses, so

    python compare_three_way.py --images_dir <DENTEX xrays> --subset all --max_images 300 --seed 42

    runs on exactly the 300 DENTEX images of the original benchmark.

Extra ESMA v2 rows:   --variants fullT,polish
    fullT  : v2 with early_stop=False (equal iteration budget)
    polish : v2 with local_refine=True (opt-in integer-neighbourhood polish)
Reproduction mode:    --legacy_fitness
    runs Standard SMA and ESMA v1 through the original per-agent fitness loop
    (byte-for-byte legacy code path; ~9x slower).
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon

from bench_common import list_images, prepare_image, select_subset
from metrics import compute_psnr, compute_ssim
from sma_algorithms import (
    KapurEntropyTable,
    apply_thresholds,
    enhanced_sma,
    enhanced_sma_v2,
    kapur_optimal_thresholds,
    standard_sma,
)

LABELS = {
    "standard_sma": "Standard SMA",
    "esma_v1": "Original ESMA (v1)",
    "esma_v2": "ESMA v2",
    "esma_v2_fullT": "ESMA v2 (full T, no early stop)",
    "esma_v2_polish": "ESMA v2 + opt-in polish",
}

METRICS = [
    ("kapur", "Kapur's Entropy", False),
    ("psnr", "PSNR", False),
    ("ssim", "SSIM", False),
    ("runtime_sec", "Runtime (s)", True),
    ("iterations", "Iterations used", True),
]


def build_algorithms(N, T, d, seed, legacy_fitness, v2_kw, variants, skip_v1=False):
    ff = not legacy_fitness
    common = dict(d=d, N=N, T=T, lb=0, ub=255, seed=seed)
    algos = {
        "standard_sma": lambda prob: standard_sma(prob, fast_fitness=ff, **common),
        "esma_v2": lambda prob: enhanced_sma_v2(prob, **common, **v2_kw),
    }
    if not skip_v1:
        algos["esma_v1"] = lambda prob: enhanced_sma(prob, fast_fitness=ff, **common)
    if "fullT" in variants:
        kw = dict(v2_kw, early_stop=False)
        algos["esma_v2_fullT"] = lambda prob: enhanced_sma_v2(prob, **common, **kw)
    if "polish" in variants:
        kw2 = dict(v2_kw, local_refine=True)
        algos["esma_v2_polish"] = lambda prob: enhanced_sma_v2(prob, **common, **kw2)
    return algos


def paired_stats(a_vals, b_vals, lower_is_better=False):
    """b vs a (positive gain = b larger). Drops pairs with None/inf."""
    pairs = [(x, y) for x, y in zip(a_vals, b_vals)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if not pairs:
        return None
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    diff = b - a
    better = (diff < 0) if lower_is_better else (diff > 0)
    try:
        _, p_value = wilcoxon(a, b)
        p_value = float(p_value)
    except ValueError:      # e.g. all differences are exactly zero
        p_value = None
    return {
        "n": int(len(diff)),
        "a_mean": float(a.mean()), "a_std": float(a.std()),
        "b_mean": float(b.mean()), "b_std": float(b.std()),
        "mean_gain": float(diff.mean()), "gain_std": float(diff.std()),
        "n_b_better": int(better.sum()), "pct_b_better": float(100 * better.mean()),
        "n_ties": int(np.sum(diff == 0)),
        "wilcoxon_p_value": p_value,
    }


def fmt_p(p):
    if p is None:
        return "n/a (all differences zero)"
    return f"{p:.4f} -- {'SIGNIFICANT (p < 0.05)' if p < 0.05 else 'not significant (p >= 0.05)'}"


def run(args):
    paths = select_subset(list_images(args.images_dir), args.subset, seed=args.seed,
                          dev_n=args.dev_n, max_images=args.max_images)
    variants = [v for v in (args.variants.split(",") if args.variants else []) if v]
    v2_kw = json.loads(args.v2_kw) if args.v2_kw else {}
    algos = build_algorithms(args.N, args.T, args.d, args.seed, args.legacy_fitness, v2_kw, variants,
                              skip_v1=args.skip_v1)
    names = list(algos)
    print(f"Subset '{args.subset}': {len(paths)} images | N={args.N} T={args.T} d={args.d} seed={args.seed}"
          f" | fitness path: {'LEGACY per-agent loop' if args.legacy_fitness else 'shared entropy table'}")
    if v2_kw:
        print(f"ESMA v2 overrides: {v2_kw}")

    per_image = []
    t_start = time.perf_counter()
    for i, path in enumerate(paths):
        if i > 0 and i % 10 == 0:
            print(f"  ...{i}/{len(paths)} images done ({time.perf_counter() - t_start:.1f}s elapsed)", flush=True)
        image, prob = prepare_image(path)
        if image is None:
            print(f"  [skip] could not read '{os.path.basename(path)}'")
            continue

        table = KapurEntropyTable(prob)
        h_star, thr_star = kapur_optimal_thresholds(prob, args.d, table=table)
        seg_star = apply_thresholds(image, thr_star)
        psnr_star = compute_psnr(image, seg_star)
        rec = {
            "file": os.path.basename(path),
            "shape": list(image.shape),
            "optimum": {
                "kapur": h_star, "thresholds": [int(t) for t in thr_star],
                "psnr": None if psnr_star == float("inf") else psnr_star,
                "ssim": compute_ssim(image, seg_star),
            },
            "algorithms": {},
        }
        for name, fn in algos.items():
            r = fn(prob)
            seg = apply_thresholds(image, r["thresholds"])
            psnr = compute_psnr(image, seg)
            rec["algorithms"][name] = {
                "kapur": float(r["fitness"]),
                "thresholds": [int(t) for t in r["thresholds"]],
                "gap": float(h_star - r["fitness"]),
                "at_optimum": bool(h_star - r["fitness"] < 1e-9),
                "psnr": None if psnr == float("inf") else psnr,
                "ssim": compute_ssim(image, seg),
                "runtime_sec": float(r["runtime_sec"]),
                "iterations": int(r.get("iterations_used", args.T)),
                "n_reinitialized": int(r.get("n_reinitialized", 0)),
            }
        per_image.append(rec)

    n = len(per_image)
    print(f"\nProcessed {n} images successfully ({time.perf_counter() - t_start:.1f}s total).\n")

    def col(name, key):
        return [rec["algorithms"][name][key] for rec in per_image]

    # ---------------- per-algorithm summary ----------------
    summary = {}
    print("=== Per-algorithm summary (mean +/- std) ===")
    h_stars = np.array([rec["optimum"]["kapur"] for rec in per_image])
    for name in names:
        kap = np.array(col(name, "kapur")); ps = np.array([v for v in col(name, "psnr") if v is not None])
        ss = np.array(col(name, "ssim")); rt = np.array(col(name, "runtime_sec")) * 1e3
        it = np.array(col(name, "iterations")); gap = np.array(col(name, "gap"))
        at_opt = int(np.sum(gap < 1e-9))
        summary[name] = {
            "kapur_mean": float(kap.mean()), "kapur_std": float(kap.std()),
            "psnr_mean": float(ps.mean()) if ps.size else None, "psnr_std": float(ps.std()) if ps.size else None,
            "ssim_mean": float(ss.mean()), "ssim_std": float(ss.std()),
            "runtime_mean_sec": float(rt.mean() / 1e3), "runtime_std_sec": float(rt.std() / 1e3),
            "iterations_mean": float(it.mean()),
            "gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)), "gap_max": float(gap.max()),
            "n_at_optimum": at_opt, "pct_at_optimum": float(100 * at_opt / n),
        }
        print(f"{LABELS[name]}")
        print(f"  Kapur:    {kap.mean():.4f} +/- {kap.std():.4f}")
        print(f"  PSNR:     {ps.mean():.4f} +/- {ps.std():.4f}")
        print(f"  SSIM:     {ss.mean():.4f} +/- {ss.std():.4f}")
        print(f"  Runtime:  {rt.mean():.1f}ms +/- {rt.std():.1f}ms")
        print(f"  Iters:    {it.mean():.1f}")
        print(f"  Gap mean: {gap.mean():.4f}")
        print(f"  At optimum: {at_opt}/{n}")
        print()
    ps_star = np.array([rec["optimum"]["psnr"] for rec in per_image if rec["optimum"]["psnr"] is not None])
    ss_star = np.array([rec["optimum"]["ssim"] for rec in per_image])
    summary["exact_optimum"] = {
        "kapur_mean": float(h_stars.mean()), "psnr_mean": float(ps_star.mean()) if ps_star.size else None,
        "ssim_mean": float(ss_star.mean()),
    }
    print("Exact Kapur optimum (DP reference)")
    print(f"  Kapur: {h_stars.mean():.4f}")
    print(f"  PSNR:  {ps_star.mean():.4f}")
    print(f"  SSIM:  {ss_star.mean():.4f}")
    print()

    # ---------------- pairwise comparisons ----------------
    pairs = [("standard_sma", "esma_v2")]
    if not args.skip_v1:
        pairs = [("standard_sma", "esma_v1"), ("standard_sma", "esma_v2"), ("esma_v1", "esma_v2")]
    for v in ("esma_v2_fullT", "esma_v2_polish"):
        if v in algos:
            pairs.append(("esma_v2", v))
    pairwise = {}
    for a, b in pairs:
        key = f"{a}__vs__{b}"
        pairwise[key] = {}
        print(f"=== {LABELS[b]} vs {LABELS[a]} ===")
        for mkey, mlabel, lower in METRICS:
            st = paired_stats(col(a, mkey), col(b, mkey), lower_is_better=lower)
            pairwise[key][mkey] = st
            if st is None:
                continue
            direction = "lower is better" if lower else "higher is better"
            print(f"  {mlabel} ({direction})")
            print(f"    {LABELS[a]}: {st['a_mean']:.4f}")
            print(f"    {LABELS[b]}: {st['b_mean']:.4f}")
            print(f"    gain: {st['mean_gain']:+.4f} (std {st['gain_std']:.4f})")
            print(f"    {LABELS[b]} better: {st['n_b_better']}/{st['n']} ({st['pct_b_better']:.1f}%), ties {st['n_ties']}")
            print(f"    Wilcoxon p: {fmt_p(st['wilcoxon_p_value'])}")
            print()
        print()

    out = {
        "n_images": n,
        "subset": args.subset, "dev_n": args.dev_n, "images_dir": os.path.abspath(args.images_dir),
        "sma_params": {"N": args.N, "T": args.T, "d": args.d, "seed": args.seed},
        "fitness_path": "legacy" if args.legacy_fitness else "table",
        "v2_overrides": v2_kw,
        "labels": {k: LABELS[k] for k in names},
        "summary": summary,
        "pairwise": pairwise,
        "per_image": per_image,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved full results (summary + per-image) -> {args.out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--subset", choices=["dev", "holdout", "all"], default="holdout")
    parser.add_argument("--dev_n", type=int, default=40, help="size of the development split")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--sma_population", type=int, default=30, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=50, dest="T")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="algorithm seed AND the shuffle seed of the image list (as in compare_full_images.py)")
    parser.add_argument("--variants", default="fullT,polish",
                        help="comma list of extra ESMA v2 rows: fullT, polish (empty string for none)")
    parser.add_argument("--v2_kw", default=None, help="JSON dict of enhanced_sma_v2 keyword overrides")
    parser.add_argument("--skip_v1", action="store_true",
                        help="Exclude ESMA v1 from the run entirely -- Standard SMA vs ESMA v2 only "
                             "(v1 was already verified separately; use this to keep v2-branch runs focused)")
    parser.add_argument("--legacy_fitness", action="store_true",
                        help="run Standard SMA / ESMA v1 through the original per-agent fitness loop")
    parser.add_argument("--out", default="./three_way_comparison.json")
    run(parser.parse_args())