"""Follow-up: hybrid (w=0.5, Kapur units) objective at d = 4, 8 and 10 on all 100 Mendeley OPGs.
Standard SMA vs ESMA v2 vs exact DP optimum; gaps over seeds 1,2,3; PSNR/SSIM (class-mean repaint) for seed 42."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)          # backend/
sys.path.insert(0, BACKEND); sys.path.insert(0, HERE)
import numpy as np, cv2
from multiprocessing import Pool
from fastmetrics import fast_psnr, fast_ssim
from bench_common import list_images, prepare_image, select_subset
from sma_algorithms import (KapurEntropyTable, BetweenClassVarianceTable, HybridObjectiveTable,
                            kapur_optimal_thresholds, apply_thresholds, standard_sma, enhanced_sma_v2)

IMAGES = os.path.join(BACKEND, "..", "Datasets", "images")
N, T, SEED = 30, 100, 42
DS = (4, 8, 10)
GAP_SEEDS = (1, 2, 3)


def metrics(img, thr):
    seg = apply_thresholds(img, thr, "mean")
    return {"psnr": fast_psnr(img, seg), "ssim": fast_ssim(img, seg)}


def work(path):
    cv2.setNumThreads(1)
    img, prob = prepare_image(path)
    if img is None:
        return None
    rec = {"file": os.path.basename(path)}
    kt = KapurEntropyTable(prob); ot = BetweenClassVarianceTable(prob)
    for d in DS:
        tab = HybridObjectiveTable(prob, d, 0.5, kt, ot)
        fstar, thr_star = kapur_optimal_thresholds(prob, d, table=tab)
        row = {"fstar": float(fstar), "norms": list(tab.norms),
               "dp": {"thr": [int(t) for t in thr_star], "kapur": float(kt.evaluate_one(thr_star)),
                      "sigma_B2": float(ot.evaluate_one(thr_star)), **metrics(img, thr_star)}}
        for name, fn in (("std", standard_sma), ("v2", enhanced_sma_v2)):
            gaps, its = [], []
            for s in GAP_SEEDS:
                r = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=s, objective=tab)
                gaps.append(float(fstar - r["fitness"])); its.append(int(r.get("iterations_used", T)))
            r = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=SEED, objective=tab)
            thr = [int(t) for t in r["thresholds"]]
            row[name] = {"gaps": gaps, "iters": its, "thr": thr, "F": float(r["fitness"]), "gap42": float(fstar - r["fitness"]),
                         "kapur": float(kt.evaluate_one(thr)), "sigma_B2": float(ot.evaluate_one(thr)),
                         "it42": int(r.get("iterations_used", T)), "rt42": float(r["runtime_sec"]), **metrics(img, thr)}
        rec[str(d)] = row
    return rec


if __name__ == "__main__":
    paths = list_images(IMAGES)
    dev, hold = select_subset(paths, "dev"), select_subset(paths, "holdout")
    jobs = dev + hold
    t0 = time.perf_counter(); out = []
    with Pool(4) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, jobs), 1):
            if rec is not None:
                out.append(rec)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    out.sort(key=lambda r: r["file"])
    res = {"n": len(out), "N": N, "T": T, "seed": SEED, "ds": list(DS), "gap_seeds": list(GAP_SEEDS),
           "holdout_files": [os.path.basename(p) for p in hold], "per_image": out}
    with open(os.path.join(HERE, "exp_hybrid_d.json"), "w") as f:
        json.dump(res, f)
    print(f"saved exp_hybrid_d.json ({time.perf_counter() - t0:.0f}s)")
