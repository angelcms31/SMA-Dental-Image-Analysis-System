"""
ablate_esma_v2.py
------------------
Component ablation of ESMA (enhanced_sma): switches each component off
(or swaps it for the Standard-SMA-style alternative) one at a time and
measures the effect against the EXACT Kapur optimum of every image
(kapur_optimal_thresholds), which is what makes the numbers interpretable:
"mean optimality gap" and "share of images where the run found the true
optimum" instead of raw entropy differences.

Only the histogram is needed per image, so this is fast (no PSNR/SSIM): use
several seeds (--seeds 1,2,3) to average out run-to-run randomness -- with a
single seed the ranking of similar configurations is noise.

Run it on the DEVELOPMENT subset (default) when choosing defaults; report
the hold-out subset with compare_three_way.py.

Usage:
    python ablate_esma_v2.py --images_dir ../Datasets/images --subset dev --seeds 1,2,3
"""

import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon

from bench_common import list_images, prepare_image, select_subset
from sma_algorithms import enhanced_sma, kapur_optimal_thresholds, standard_sma

# name -> keyword overrides of enhanced_sma (relative to its defaults)
ABLATIONS = {
    "ESMA (defaults)": {},
    "  - canonical ordering (unsorted agents)": {"canonical": False},
    "  - LHS init -> thesis-literal diagonal strata": {"init": "strata"},
    "  - LHS init -> uniform random init": {"init": "uniform"},
    "  - sampled leaders -> weighted centroid": {"leader_mode": "centroid"},
    "  - rank weights -> raw fitness weights": {"weight_mode": "fitness"},
    "  - a(t) adaptation (amplitude fixed at a0)": {"delta": 0.0, "gamma": 0.0},
    "  - early stopping (full T)": {"early_stop": False},
    "  + z(t) exploration channel (wide moves)": {"explore_channel": True, "explore_mode": "wide"},
    "  + z(t) exploration channel (random re-init)": {"explore_channel": True, "explore_mode": "reinit"},
    "  + opt-in local polish": {"local_refine": True},
    "  k=1 (single leader)": {"k": 1},
    "  k=3": {"k": 3},
}

# which ablations get a paired significance test against "ESMA (defaults)",
# and which Chapter 4 objective each corresponds to -- add/remove entries
# here to test other rows
SIGNIFICANCE_TESTS = {
    "  k=1 (single leader)": "Objective 1 (multi-leader guidance)",
    "  - LHS init -> uniform random init": "Objective 2 (quasi-uniform initialization)",
    "  - a(t) adaptation (amplitude fixed at a0)": "Objective 3 (adaptive control)",
}


def evaluate(fn, probs, h_stars, seeds, N, T, d, **kw):
    """Returns both the aggregate stats (as before) AND the raw per-run
    gap array, in the SAME order every time (seed outer loop, image inner
    loop) -- this lets two configurations' gap arrays be paired up
    image-for-image for a Wilcoxon signed-rank test, since both are run
    against the identical sequence of (seed, image) pairs."""
    gaps, iters, rts = [], [], []
    for s in seeds:
        for p, h in zip(probs, h_stars):
            r = fn(p, d=d, N=N, T=T, lb=0, ub=255, seed=s, **kw)
            gaps.append(h - r["fitness"])
            iters.append(r.get("iterations_used", T))
            rts.append(r["runtime_sec"])
    gaps = np.array(gaps)
    return {
        "kapur_mean": float(np.mean(h_stars) - gaps.mean()),
        "gap_mean": float(gaps.mean()), "gap_median": float(np.median(gaps)), "gap_max": float(gaps.max()),
        "pct_at_optimum": float(100 * np.mean(gaps < 1e-9)),
        "iterations_mean": float(np.mean(iters)), "runtime_ms_mean": float(np.mean(rts) * 1e3),
        "n_runs": int(len(gaps)),
        "_gaps": gaps,   # raw per-(seed,image) gaps, used only for significance testing below
    }


def main(args):
    paths = select_subset(list_images(args.images_dir), args.subset, seed=args.seed,
                          dev_n=args.dev_n, max_images=args.max_images)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    probs, h_stars = [], []
    for p in paths:
        image, prob = prepare_image(p)
        if image is None:
            continue
        probs.append(prob)
        h_stars.append(kapur_optimal_thresholds(prob, args.d)[0])
    print(f"Subset '{args.subset}': {len(probs)} images x {len(seeds)} seeds | N={args.N} T={args.T} d={args.d}")
    print(f"Exact-optimum mean Kapur: {np.mean(h_stars):.4f}\n")

    rows = {}
    hdr = f"{'configuration':48s} {'Kapur':>8s} {'gap mean':>9s} {'gap max':>8s} {'at opt':>7s} {'iters':>6s} {'ms':>7s}"
    print(hdr)

    def show(name, r):
        print(f"{name:48s} {r['kapur_mean']:8.4f} {r['gap_mean']:9.4f} {r['gap_max']:8.4f} "
              f"{r['pct_at_optimum']:6.1f}% {r['iterations_mean']:6.1f} {r['runtime_ms_mean']:7.1f}", flush=True)

    t0 = time.perf_counter()
    rows["Standard SMA"] = evaluate(standard_sma, probs, h_stars, seeds, args.N, args.T, args.d)
    show("Standard SMA", rows["Standard SMA"])
    for name, kw in ABLATIONS.items():
        rows[name] = dict(evaluate(enhanced_sma, probs, h_stars, seeds, args.N, args.T, args.d, **kw), overrides=kw)
        show(name, rows[name])
    print(f"\n({time.perf_counter() - t0:.0f}s)")

    # --- paired Wilcoxon signed-rank test: each ablation vs. "ESMA (defaults)" ---
    # valid because evaluate() runs every configuration against the exact same
    # (seed, image) sequence, so index i in both gap arrays refers to the same
    # (seed, image) pair -- this is what makes them a matched/paired sample.
    print("\nSignificance (paired Wilcoxon signed-rank test vs. 'ESMA (defaults)'):")
    print(f"{'objective':45s} {'ablation row':40s} {'p-value':>10s}  significant?")
    sig_results = {}
    base_gaps = rows["ESMA (defaults)"]["_gaps"]
    for ablation_name, objective_label in SIGNIFICANCE_TESTS.items():
        other_gaps = rows[ablation_name]["_gaps"]
        diffs = other_gaps - base_gaps
        if np.allclose(diffs, 0):
            p = 1.0  # identical arrays -- wilcoxon errors on all-zero differences
        else:
            _, p = wilcoxon(other_gaps, base_gaps)
        sig = "Yes (p < 0.05)" if p < 0.05 else "No"
        print(f"{objective_label:45s} {ablation_name.strip():40s} {p:10.4g}  {sig}")
        sig_results[ablation_name] = {"objective": objective_label, "p_value": float(p), "significant": bool(p < 0.05)}

    # strip the raw per-run gap arrays before saving -- json.dump can't
    # serialize numpy arrays, and the aggregate stats already summarize them
    for r in rows.values():
        r.pop("_gaps", None)

    with open(args.out, "w") as f:
        json.dump({"subset": args.subset, "n_images": len(probs), "seeds": seeds,
                   "sma_params": {"N": args.N, "T": args.T, "d": args.d},
                   "exact_optimum_kapur_mean": float(np.mean(h_stars)), "rows": rows,
                   "significance_vs_defaults": sig_results}, f, indent=1)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--subset", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--dev_n", type=int, default=40)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for the dev/holdout split")
    parser.add_argument("--sma_population", type=int, default=30, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=50, dest="T")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--out", default="./esma_v2_ablation.json")
    main(parser.parse_args())
