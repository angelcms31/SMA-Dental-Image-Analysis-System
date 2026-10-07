"""
Per-objective ablation for ESMA.

Put this file in the backend folder (same folder as sma_algorithms.py), then:

    python run_ablation.py "C:\\path\\to\\DENTEX\\xrays"

By default it uses the SAME 300 images as the Chapter 4 benchmark
(compare_three_way.py --subset all --max_images 300 --seed 42): the sorted
list of images in that folder, shuffled with seed 42, first 300.

Optional:  --seeds 20  --max_images 300  --seed 42  --d 4  --N 30  --T 100  --out ablation_results.csv

Each configuration changes ONE thing from the full ESMA, so the table shows
what each part contributes. "Hit rate" = % of runs that reach the exact
optimum (found by dynamic programming); "gap" = optimum - fitness reached.
"""
import argparse, csv, glob, os, random, sys, time
from multiprocessing import Pool

import cv2
import numpy as np

from sma_algorithms import (KapurEntropyTable, autocrop_black_borders, compute_histogram_prob,
                            enhanced_sma, kapur_optimal_thresholds, standard_sma)

CONFIGS = [
    ("Baseline", "Standard SMA", None),
    ("Full", "ESMA (all three objectives)", {}),
    ("Objective 1", "Single leader (k = 1)", dict(k=1)),
    ("Objective 1", "Averaged leaders (weighted centroid)", dict(leader_mode="centroid")),
    ("Objective 1", "Sampled leader, fitness weights", dict(weight_mode="fitness")),
    ("Objective 2", "Random init, no sorting", dict(init="uniform", canonical=False)),
    ("Objective 2", "LHS, no sorting", dict(canonical=False)),
    ("Objective 2", "Random init + sorting", dict(init="uniform")),
    ("Objective 3", "Fixed a, no early stop", dict(alpha=0, beta=0, gamma=0, delta=0, early_stop=False)),
    ("Objective 3", "Adaptive a, no early stop", dict(early_stop=False)),
    ("Objective 3", "Fixed a + early stop", dict(alpha=0, beta=0, gamma=0, delta=0)),
]
ARGS = {}


def _load(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return compute_histogram_prob(autocrop_black_borders(img))


def _work(path):
    prob = _load(path)
    if prob is None:
        return None
    d, N, T, seeds = ARGS["d"], ARGS["N"], ARGS["T"], ARGS["seeds"]
    best, _ = kapur_optimal_thresholds(prob, d, table=KapurEntropyTable(prob))
    out = []
    for _, name, kw in CONFIGS:
        rows = []
        for s in range(seeds):
            t0 = time.perf_counter()
            if kw is None:
                r = standard_sma(prob, d=d, N=N, T=T, seed=s)
            else:
                r = enhanced_sma(prob, d=d, N=N, T=T, seed=s, **kw)
            ms = (time.perf_counter() - t0) * 1000
            rows.append((best - r["fitness"], r.get("iterations_used", T), ms))
        out.append(rows)
    return out


def _init(a):
    ARGS.update(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for choosing the images")
    ap.add_argument("--out", default="ablation_results.csv")
    a = ap.parse_args()
    # Same listing and selection as compare_three_way.py / bench_common.py:
    # sorted file list of this folder (no subfolders), shuffled with
    # random.Random(seed), then the first --max_images files.
    files = sorted(f for ext in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(a.folder, "*." + ext)))
    if not files:
        sys.exit("No images found directly inside " + a.folder)
    random.Random(a.seed).shuffle(files)
    if a.max_images:
        files = files[:a.max_images]
    print(f"{len(files)} images x {a.seeds} seeds x {len(CONFIGS)} configurations")
    args = dict(d=a.d, N=a.N, T=a.T, seeds=a.seeds)
    with Pool(initializer=_init, initargs=(args,)) as pool:
        res = []
        for i, r in enumerate(pool.imap(_work, files), 1):
            if r is not None:
                res.append(r)
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)} images done", flush=True)
    full = np.array([[x[0] for x in r[1]] for r in res]).mean(1)
    table = []
    for c, (group, name, _) in enumerate(CONFIGS):
        arr = np.array([x for r in res for x in r[c]])
        per_img = np.array([[x[0] for x in r[c]] for r in res]).mean(1)
        table.append({
            "group": group, "configuration": name,
            "hit_rate_pct": round(100 * float(np.mean(arr[:, 0] <= 1e-6)), 1),
            "mean_gap": round(float(arr[:, 0].mean()), 5),
            "mean_iterations": round(float(arr[:, 1].mean()), 1),
            "mean_runtime_ms": round(float(arr[:, 2].mean()), 1),
            "images_full_esma_better": int((full < per_img - 1e-9).sum()),
            "images_full_esma_worse": int((full > per_img + 1e-9).sum()),
            "runs": len(arr),
        })
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)
    print()
    for t in table:
        print(f"{t['group']:12s} {t['configuration']:38s} hit {t['hit_rate_pct']:5.1f}%  gap {t['mean_gap']:.5f}"
              f"  iters {t['mean_iterations']:5.1f}  {t['mean_runtime_ms']:6.1f} ms")
    print("\nSaved:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()