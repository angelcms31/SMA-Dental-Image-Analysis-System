"""
run_dentex.py -- re-run every DENTEX result in Chapter 4 with the REVISED ESMA
(the current sma_algorithms.py). One command, three parts:

  Part 1  Table 4.1 / Figure 4.1   Standard SMA vs revised ESMA (and ESMA before
                                   revision), one run per image: Kapur, PSNR, SSIM,
                                   runtime, iterations, gap, exact-optimum hits,
                                   Wilcoxon signed-rank tests
  Part 2  Tables 4.2 to 4.9        ablation: full ESMA vs versions with one component
                                   changed, 20 runs per image
  Part 3  k sweep                  number of leaders k = 1, 2, 3, 4, 5, 7, 10, 15

Put this file in your backend folder (next to sma_algorithms.py, bench_common.py and
metrics.py) and run, in PowerShell:

    python run_dentex.py --images_dir "C:\\path\\to\\DENTEX\\xrays"

It selects exactly the same 300 images as before (sorted list shuffled with seed 42,
first 300), with the same border autocrop. Optional:
    --T 100 --N 30 --d 4      search settings (Chapter 3: N = 30, T = 100, d = 4)
    --seeds 20                runs per image for Parts 2 and 3
    --parts 1,2,3             run only some parts
    --workers 4               processes for Parts 2 and 3 (Part 1 runs one at a time
                              so runtimes are measured fairly)

Outputs (send these back): dentex_report.txt, dentex_table41.csv,
dentex_table41_per_image.csv, dentex_ablation.csv, dentex_ksweep.csv
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import rankdata, wilcoxon

from bench_common import list_images, prepare_image, select_subset
from sma_algorithms import (KapurEntropyTable, apply_thresholds, enhanced_sma,
                            kapur_optimal_thresholds, standard_sma)

try:
    from metrics import compute_psnr, compute_ssim
except ImportError:                                    # fallback if metrics.py is missing
    from skimage.metrics import structural_similarity

    def compute_psnr(a, b):
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    def compute_ssim(a, b):
        return float(structural_similarity(a, b, data_range=255))

HIT = 1e-9          # a run "reaches the optimum" when its gap is below this

OLD = dict(weight_mode="rank", adapt="cr_pd", accept_equal=False, kicks=0)
ABLATION = [
    # (objective, label, kwargs for enhanced_sma; None = standard_sma)
    ("Full", "Revised ESMA (all three objectives)", {}),
    ("Full", "ESMA before revision", OLD),
    ("Baseline", "Standard SMA", None),
    ("Objective 1", "Single leader (k = 1)", dict(k=1)),
    ("Objective 1", "Top-5, rank weights (before revision)", dict(weight_mode="rank")),
    ("Objective 1", "Averaged leaders (centroid)", dict(leader_mode="centroid")),
    ("Objective 2", "Random init, no sorting, strict acceptance", dict(init="uniform", canonical=False, accept_equal=False)),
    ("Objective 2", "LHS, no sorting, strict acceptance", dict(canonical=False, accept_equal=False)),
    ("Objective 2", "Random init + sorting + equal-fitness moves", dict(init="uniform")),
    ("Objective 2", "LHS + sorting, strict acceptance (before revision)", dict(accept_equal=False)),
    ("Objective 3", "Success rule, no early stopping (full T)", dict(early_stop=False)),
    ("Objective 3", "Success rule, early stopping without kick", dict(kicks=0)),
    ("Objective 3", "CR/PD rule + early stopping (before revision)", dict(adapt="cr_pd", kicks=0)),
    ("Objective 3", "Fixed a = 1, full T", dict(adapt="cr_pd", gamma=0.0, delta=0.0, early_stop=False)),
    ("Objective 3", "Fixed a = 1 + early stopping with kick", dict(adapt="cr_pd", gamma=0.0, delta=0.0)),
]
K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15]
ARGS = {}


def _init_worker(args):
    # Windows starts workers fresh (spawn), so each one must fill its own
    # module-level ARGS. Passing ARGS.update directly would only update a
    # pickled copy and leave ARGS empty in the worker.
    ARGS.update(args)


def _run(kw, prob, s):
    N, T, d = ARGS["N"], ARGS["T"], ARGS["d"]
    if kw is None:
        r = standard_sma(prob, d=d, N=N, T=T, seed=s)
        r["iterations_used"] = T
        return r
    return enhanced_sma(prob, d=d, N=N, T=T, seed=s, **kw)


def _work_multi(path):
    image, prob = prepare_image(path)
    if image is None:
        return None
    opt, _ = kapur_optimal_thresholds(prob, ARGS["d"], table=KapurEntropyTable(prob))
    out = {"file": os.path.basename(path), "opt": opt, "abl": [], "k": []}
    seeds = range(ARGS["seeds"])
    if 2 in ARGS["parts"]:
        for _, _, kw in ABLATION:
            rows = []
            for s in seeds:
                r = _run(kw, prob, s)
                rows.append((r["fitness"], opt - r["fitness"], r["iterations_used"], 1000 * r["runtime_sec"]))
            out["abl"].append(rows)
    if 3 in ARGS["parts"]:
        for k in K_VALUES:
            rows = []
            for s in seeds:
                r = _run(dict(k=k), prob, s)
                rows.append((r["fitness"], opt - r["fitness"], r["iterations_used"], 1000 * r["runtime_sec"]))
            out["k"].append(rows)
    return out


def _fmt_p(p):
    if p is None or p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.any(np.abs(a - b) > 1e-12):
        return None
    return float(wilcoxon(a, b).pvalue)


def _write(fn, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def part1(paths, log):
    """Table 4.1: one run per image (seed = 42), as in the original comparison."""
    algos = [("Standard SMA", None), ("ESMA (revised)", {}), ("ESMA before revision", OLD)]
    per_image = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        image, prob = prepare_image(path)
        if image is None:
            continue
        opt, thr = kapur_optimal_thresholds(prob, ARGS["d"], table=KapurEntropyTable(prob))
        seg = apply_thresholds(image, thr)
        rec = {"file": os.path.basename(path), "optimum_kapur": opt,
               "optimum_psnr": compute_psnr(image, seg), "optimum_ssim": compute_ssim(image, seg)}
        for name, kw in algos:
            r = _run(kw, prob, ARGS["seed"])
            seg = apply_thresholds(image, r["thresholds"])
            rec[f"{name} | kapur"] = r["fitness"]
            rec[f"{name} | psnr"] = compute_psnr(image, seg)
            rec[f"{name} | ssim"] = compute_ssim(image, seg)
            rec[f"{name} | runtime_ms"] = 1000 * r["runtime_sec"]
            rec[f"{name} | iterations"] = r["iterations_used"]
            rec[f"{name} | gap"] = opt - r["fitness"]
        per_image.append(rec)
        if i % 25 == 0:
            print(f"  Part 1: {i}/{len(paths)} images ({time.perf_counter() - t0:.0f} s)", flush=True)
    _write("dentex_table41_per_image.csv", per_image)
    n = len(per_image)
    rows = []
    log.append(f"\n=== PART 1: Table 4.1 (n = {n} images, one run each, seed {ARGS['seed']}) ===")
    log.append(f"{'algorithm':24s} {'Kapur':>16s} {'PSNR':>16s} {'SSIM':>16s} {'runtime ms':>14s} {'iters':>6s} {'gap':>9s} {'at optimum':>11s}")
    for name, _ in algos:
        g = lambda m: np.array([r[f"{name} | {m}"] for r in per_image], float)
        kap, ps, ss, rt, it, gap = g("kapur"), g("psnr"), g("ssim"), g("runtime_ms"), g("iterations"), g("gap")
        ps = ps[np.isfinite(ps)]
        hits = int(np.sum(gap < HIT))
        rows.append({"algorithm": name, "kapur_mean": kap.mean(), "kapur_std": kap.std(), "psnr_mean": ps.mean(),
                     "psnr_std": ps.std(), "ssim_mean": ss.mean(), "ssim_std": ss.std(), "runtime_ms_mean": rt.mean(),
                     "runtime_ms_std": rt.std(), "iterations_mean": it.mean(), "gap_mean": gap.mean(),
                     "images_at_optimum": hits, "pct_at_optimum": 100 * hits / n})
        log.append(f"{name:24s} {kap.mean():8.4f}+/-{kap.std():6.4f} {ps.mean():8.4f}+/-{ps.std():6.4f} "
                   f"{ss.mean():8.4f}+/-{ss.std():6.4f} {rt.mean():7.1f}+/-{rt.std():5.1f} {it.mean():6.1f} "
                   f"{gap.mean():9.5f} {hits:4d} ({100 * hits / n:.1f}%)")
    log.append("\nPaired Wilcoxon signed-rank tests (B vs A):")
    for a, b in (("Standard SMA", "ESMA (revised)"), ("ESMA before revision", "ESMA (revised)")):
        for m, lower in (("kapur", False), ("psnr", False), ("ssim", False), ("runtime_ms", True), ("iterations", True)):
            A = np.array([r[f"{a} | {m}"] for r in per_image], float)
            B = np.array([r[f"{b} | {m}"] for r in per_image], float)
            ok = np.isfinite(A) & np.isfinite(B)
            A, B = A[ok], B[ok]
            better = (B < A) if lower else (B > A)
            p = _wil(A, B)
            rows.append({"algorithm": f"{b} vs {a}", "metric": m, "a_mean": A.mean(), "b_mean": B.mean(),
                         "b_better": int(better.sum()), "ties": int(np.sum(A == B)), "n": len(A), "wilcoxon_p": p})
            log.append(f"  {b} vs {a} | {m:10s}: {A.mean():.4f} -> {B.mean():.4f} | {b} better in "
                       f"{int(better.sum())}/{len(A)} ({100 * better.mean():.1f}%), ties {int(np.sum(A == B))} | p {_fmt_p(p)}")
    _write("dentex_table41.csv", rows)


def summarize(res, key, labels, log, title, fn):
    M = np.array([[np.mean([x[0] for x in r[key][j]]) for j in range(len(labels))] for r in res])
    ranks = np.array([rankdata(-np.round(row, 9)) for row in M]).mean(0)
    rows = []
    log.append(f"\n=== {title} (n = {len(res)} images x {ARGS['seeds']} runs) ===")
    log.append(f"{'configuration':55s} {'hit %':>6s} {'gap':>9s} {'iters':>6s} {'ms':>6s} {'rank':>5s}  ref better/tie/worse   p")
    for j, lab in enumerate(labels):
        arr = np.array([x for r in res for x in r[key][j]], float)
        row = {"group": lab[0], "configuration": lab[1], "hit_pct": round(100 * float(np.mean(arr[:, 1] < HIT)), 1),
               "mean_gap": float(arr[:, 1].mean()), "mean_iterations": round(float(arr[:, 2].mean()), 1),
               "mean_runtime_ms": round(float(arr[:, 3].mean()), 2), "friedman_rank": round(float(ranks[j]), 2)}
        if j:
            diff = M[:, 0] - M[:, j]
            row.update(ref_better=int((diff > 1e-9).sum()), tie=int((np.abs(diff) <= 1e-9).sum()),
                       ref_worse=int((diff < -1e-9).sum()), wilcoxon_p=_wil(M[:, 0], M[:, j]))
        rows.append(row)
        wtl = f"{row['ref_better']}/{row['tie']}/{row['ref_worse']}" if j else ""
        log.append(f"{lab[1]:55s} {row['hit_pct']:6.1f} {row['mean_gap']:9.5f} {row['mean_iterations']:6.1f} "
                   f"{row['mean_runtime_ms']:6.1f} {row['friedman_rank']:5.2f}  {wtl:20s} {_fmt_p(row.get('wilcoxon_p')) if j else ''}")
    _write(fn, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="image shuffle seed and the Part 1 run seed")
    ap.add_argument("--seeds", type=int, default=20, help="runs per image in Parts 2 and 3")
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    parts = {int(x) for x in a.parts.split(",") if x}
    ARGS.update(N=a.N, T=a.T, d=a.d, seed=a.seed, seeds=a.seeds, parts=parts)
    paths = select_subset(list_images(a.images_dir), "all", seed=a.seed, max_images=a.max_images)
    log = [f"DENTEX re-run with the revised ESMA | {len(paths)} images | N={a.N} T={a.T} d={a.d}"]
    print(log[0])
    t0 = time.perf_counter()
    if 1 in parts:
        part1(paths, log)
    if parts & {2, 3}:
        with Pool(a.workers, initializer=_init_worker, initargs=(dict(ARGS),)) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_work_multi, paths), 1):
                if r is not None:
                    res.append(r)
                if i % 25 == 0:
                    print(f"  Parts 2-3: {i}/{len(paths)} images ({time.perf_counter() - t0:.0f} s)", flush=True)
        res.sort(key=lambda r: r["file"])
        if 2 in parts:
            summarize(res, "abl", [(g, l) for g, l, _ in ABLATION], log,
                      "PART 2: Ablation (reference = revised ESMA)", "dentex_ablation.csv")
        if 3 in parts:
            summarize(res, "k", [("k sweep", f"k = {k}") for k in K_VALUES], log,
                      "PART 3: k sweep (reference = k = 1)", "dentex_ksweep.csv")
    log.append(f"\nTotal time {time.perf_counter() - t0:.0f} s")
    text = "\n".join(log)
    print(text)
    with open("dentex_report.txt", "w") as f:
        f.write(text + "\n")
    print("\nSaved dentex_report.txt and the dentex_*.csv files")


if __name__ == "__main__":
    main()
