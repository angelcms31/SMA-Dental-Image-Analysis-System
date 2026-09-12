"""
exp_metrics.py -- three experiments on the 100 Mendeley OPGs:
  A. d=4 protocol (N=30, T=100, seed 42): Standard SMA / ESMA v2 / exact DP optimum under
     three objectives (kapur, otsu, hybrid w=0.5); PSNR+SSIM under even and mean repaint.
  B. d sweep (4..12), kapur objective: gap-to-optimum / at-optimum over seeds 1,2,3 and
     PSNR/SSIM (mean repaint, seed 42) for Standard, ESMA v2 and the DP optimum.
  C. non-separable SSIM-aware objective (w*kapur_norm + (1-w)*SSIM_small) on the first
     n hold-out images: Standard vs ESMA v2, full-resolution metrics of the result.
"""
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
D_SWEEP = (4, 6, 8, 10, 12)
GAP_SEEDS = (1, 2, 3)
C_WEIGHTS = (0.5, 0.8)
C_SCALE = 8


class KapurSSIMObjective:
    """F = w * H_K / H_K* + (1 - w) * SSIM(downsampled original, downsampled class-mean repaint).
    Non-separable (SSIM is spatial) -> no DP optimum; evaluations memoised per integer vector."""
    name = "kapur+ssim"

    def __init__(self, img, prob, d, weight, scale=C_SCALE):
        self.kt = KapurEntropyTable(prob)
        self.hk = kapur_optimal_thresholds(prob, d, table=self.kt)[0]
        h, w = img.shape
        self.small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        self.w = float(weight); self.cache = {}; self.L = 256; self.n_ssim = 0

    def _ssim(self, key):
        v = self.cache.get(key)
        if v is None:
            v = fast_ssim(self.small, apply_thresholds(self.small, key, "mean"))
            self.cache[key] = v; self.n_ssim += 1
        return v

    def evaluate(self, X, assume_sorted=False):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Ti = np.clip(np.rint(X), 0, 255).astype(np.intp)
        if not assume_sorted:
            Ti.sort(axis=1)
        k = self.kt.evaluate(Ti, assume_sorted=True) / self.hk
        s = np.array([self._ssim(tuple(int(v) for v in row)) for row in Ti])
        return self.w * k + (1.0 - self.w) * s

    def evaluate_one(self, thr):
        return float(self.evaluate(np.asarray(thr, dtype=np.float64))[0])


def metrics(img, thr, mode):
    seg = apply_thresholds(img, thr, mode)
    return {"psnr": fast_psnr(img, seg), "ssim": fast_ssim(img, seg)}


def work(args):
    path, do_c = args
    cv2.setNumThreads(1)
    img, prob = prepare_image(path)
    if img is None:
        return None
    rec = {"file": os.path.basename(path), "shape": list(img.shape)}
    kt = KapurEntropyTable(prob); ot = BetweenClassVarianceTable(prob)
    lv = np.arange(256.0); mu = float((prob * lv).sum())
    rec["sigma_T2"] = float((prob * lv ** 2).sum() - mu ** 2)
    # ---------------- A ----------------
    A = {}
    tables = {"kapur": kt, "otsu": ot, "hybrid": HybridObjectiveTable(prob, 4, 0.5, kt, ot)}
    for obj, tab in tables.items():
        fstar, thr_star = kapur_optimal_thresholds(prob, 4, table=tab)
        runs = {"dp": {"thr": [int(t) for t in thr_star], "fit": float(fstar), "rt": 0.0, "it": 0}}
        for name, fn in (("std", standard_sma), ("v2", enhanced_sma_v2)):
            r = fn(prob, d=4, N=N, T=T, lb=0, ub=255, seed=SEED, objective=tab)
            runs[name] = {"thr": [int(t) for t in r["thresholds"]], "fit": float(r["fitness"]),
                          "rt": float(r["runtime_sec"]), "it": int(r.get("iterations_used", T))}
        for name, rr in runs.items():
            thr = rr["thr"]
            rr["gap"] = float(fstar - rr["fit"]); rr["at_opt"] = bool(rr["gap"] < 1e-9)
            rr["kapur"] = float(kt.evaluate_one(thr)); rr["sigma_B2"] = float(ot.evaluate_one(thr))
            rr["mean"] = metrics(img, thr, "mean")
            if obj == "kapur":
                rr["even"] = metrics(img, thr, "even")
        A[obj] = runs
    rec["A"] = A
    # ---------------- B ----------------
    B = {}
    for d in D_SWEEP:
        hstar, thr_star = kapur_optimal_thresholds(prob, d, table=kt)
        row = {"hstar": float(hstar), "dp_thr": [int(t) for t in thr_star], "dp": metrics(img, thr_star, "mean")}
        for name, fn in (("std", standard_sma), ("v2", enhanced_sma_v2)):
            gaps, its, rts = [], [], []
            for s in GAP_SEEDS:
                r = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=s)
                gaps.append(float(hstar - r["fitness"])); its.append(int(r.get("iterations_used", T))); rts.append(float(r["runtime_sec"]))
            r = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=SEED)
            thr = [int(t) for t in r["thresholds"]]
            row[name] = {"gaps": gaps, "iters": its, "rts": rts, "thr": thr, "kapur": float(r["fitness"]),
                         "gap42": float(hstar - r["fitness"]), "it42": int(r.get("iterations_used", T)),
                         "rt42": float(r["runtime_sec"]), **metrics(img, thr, "mean")}
        B[str(d)] = row
    rec["B"] = B
    # ---------------- C ----------------
    if do_c:
        C = {}
        for w in C_WEIGHTS:
            rowc = {}
            for name, fn in (("std", standard_sma), ("v2", enhanced_sma_v2)):
                objc = KapurSSIMObjective(img, prob, 4, w)
                t0 = time.perf_counter()
                r = fn(prob, d=4, N=N, T=T, lb=0, ub=255, seed=SEED, objective=objc)
                thr = [int(t) for t in r["thresholds"]]
                rowc[name] = {"thr": thr, "F": float(r["fitness"]), "kapur": float(kt.evaluate_one(thr)),
                              "it": int(r.get("iterations_used", T)), "wall": time.perf_counter() - t0,
                              "n_ssim_evals": objc.n_ssim, **metrics(img, thr, "mean")}
            C[str(w)] = rowc
        rec["C"] = C
    return rec


if __name__ == "__main__":
    n_c = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    paths = list_images(IMAGES)
    dev, hold = select_subset(paths, "dev"), select_subset(paths, "holdout")
    c_set = set(hold[:n_c])
    jobs = [(p, p in c_set) for p in dev + hold]
    print(f"{len(jobs)} images ({len(dev)} dev + {len(hold)} holdout); experiment C on {len(c_set)} holdout images", flush=True)
    t0 = time.perf_counter(); out = []
    with Pool(4) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, jobs), 1):
            if rec is not None:
                out.append(rec)
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    out.sort(key=lambda r: r["file"])
    res = {"n": len(out), "N": N, "T": T, "seed": SEED, "d_sweep": list(D_SWEEP), "gap_seeds": list(GAP_SEEDS),
           "dev_files": [os.path.basename(p) for p in dev], "holdout_files": [os.path.basename(p) for p in hold],
           "c_files": sorted(os.path.basename(p) for p in c_set), "per_image": out}
    with open(os.path.join(HERE, "exp_results.json"), "w") as f:
        json.dump(res, f)
    print(f"saved exp_results.json ({time.perf_counter() - t0:.0f}s)")
