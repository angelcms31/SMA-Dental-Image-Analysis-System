"""
ablate_esma_v2.py
------------------
Component ablation of ESMA v2 (enhanced_sma_v2): switches each v2 component
off (or swaps it for the v1 / Standard-SMA alternative) one at a time and
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

from bench_common import list_images, prepare_image, select_subset
from sma_algorithms import enhanced_sma, enhanced_sma_v2, kapur_optimal_thresholds, standard_sma

# name -> keyword overrides of enhanced_sma_v2 (relative to its defaults)
ABLATIONS = {
    "ESMA v2 (defaults)": {},
    "  - canonical ordering (unsorted agents)": {"canonical": False},
    "  - LHS init -> thesis-literal diagonal strata": {"init": "strata"},
    "  - LHS init -> uniform random init": {"init": "uniform"},
    "  - sampled leaders -> weighted centroid": {"leader_mode": "centroid"},
    "  - rank weights -> raw fitness weights (v1)": {"weight_mode": "fitness"},
    "  - a(t) adaptation (amplitude fixed at a0)": {"delta": 0.0, "gamma": 0.0},
    "  - early stopping (full T)": {"early_stop": False},
    "  + z(t) exploration channel (wide moves)": {"explore_channel": True, "explore_mode": "wide"},
    "  + z(t) exploration channel (random re-init)": {"explore_channel": True, "explore_mode": "reinit"},
    "  + opt-in local polish": {"local_refine": True},
    "  k=1 (single leader)": {"k": 1},
    "  k=3": {"k": 3},
}


def evaluate(fn, probs, h_stars, seeds, N, T, d, **kw):
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
    for name, fn in (("Standard SMA", standard_sma), ("Original ESMA (v1)", enhanced_sma)):
        rows[name] = evaluate(fn, probs, h_stars, seeds, args.N, args.T, args.d)
        show(name, rows[name])
    for name, kw in ABLATIONS.items():
        rows[name] = dict(evaluate(enhanced_sma_v2, probs, h_stars, seeds, args.N, args.T, args.d, **kw), overrides=kw)
        show(name, rows[name])
    print(f"\n({time.perf_counter() - t0:.0f}s)")

    with open(args.out, "w") as f:
        json.dump({"subset": args.subset, "n_images": len(probs), "seeds": seeds,
                   "sma_params": {"N": args.N, "T": args.T, "d": args.d},
                   "exact_optimum_kapur_mean": float(np.mean(h_stars)), "rows": rows}, f, indent=1)
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
