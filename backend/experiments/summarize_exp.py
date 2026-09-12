"""Summarize exp_results.json (experiments A, B, C). Usage: python summarize_exp.py [all|holdout]"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "exp_results.json")))
subset = sys.argv[1] if len(sys.argv) > 1 else "all"
keep = set(R["holdout_files"]) if subset == "holdout" else None
P = [r for r in R["per_image"] if keep is None or r["file"] in keep]
n = len(P)
print(f"===== subset={subset}: n={n} images | N={R['N']} T={R['T']} seed={R['seed']} =====\n")


def arr(f):
    return np.array([f(r) for r in P], dtype=float)


def paired(a, b):
    d = b - a
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return d.mean(), int((d > 0).sum()), int((d == 0).sum()), p


def fmt_p(p):
    return "n/a" if np.isnan(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


# ---------------------------------------------------------------- A
print("A. d=4 -- objective x algorithm (means over images)")
print(f"{'objective':8s} {'algo':4s} {'Kapur':>8s} {'sigmaB2':>8s} {'gap':>8s} {'at opt':>7s} {'iters':>6s} "
      f"{'PSNR even':>9s} {'SSIM even':>9s} {'PSNR mean':>9s} {'SSIM mean':>9s}")
for obj in ("kapur", "otsu", "hybrid"):
    for alg in ("std", "v2", "dp"):
        g = lambda k: arr(lambda r: r["A"][obj][alg][k])
        pe = arr(lambda r: r["A"][obj][alg]["even"]["psnr"]) if obj == "kapur" else None
        se = arr(lambda r: r["A"][obj][alg]["even"]["ssim"]) if obj == "kapur" else None
        pm = arr(lambda r: r["A"][obj][alg]["mean"]["psnr"]); sm = arr(lambda r: r["A"][obj][alg]["mean"]["ssim"])
        at = arr(lambda r: r["A"][obj][alg]["at_opt"])
        print(f"{obj:8s} {alg:4s} {g('kapur').mean():8.4f} {g('sigma_B2').mean():8.1f} {g('gap').mean():8.5f} "
              f"{100*at.mean():6.1f}% {g('it').mean():6.1f} "
              f"{(pe.mean() if pe is not None else float('nan')):9.4f} {(se.mean() if se is not None else float('nan')):9.4f} "
              f"{pm.mean():9.4f} {sm.mean():9.4f}")
print()
print("A1. Repaint effect (mean - even), same thresholds, kapur objective  [gain, #better, ties, Wilcoxon p]")
for alg in ("std", "v2", "dp"):
    for m in ("psnr", "ssim"):
        a = arr(lambda r: r["A"]["kapur"][alg]["even"][m]); b = arr(lambda r: r["A"]["kapur"][alg]["mean"][m])
        g, nb, nt, p = paired(a, b)
        print(f"  {alg:4s} {m}: {g:+.4f}  better {nb}/{n} ties {nt}  p={fmt_p(p)}")
print()
print("A2. ESMA v2 vs Standard SMA, per objective  [gain, #v2 better, ties, p]")
for obj in ("kapur", "otsu", "hybrid"):
    for label, f in (("Kapur", lambda r, a: r["A"][obj][a]["kapur"]),
                     ("objective", lambda r, a: r["A"][obj][a]["fit"]),
                     ("PSNR(mean)", lambda r, a: r["A"][obj][a]["mean"]["psnr"]),
                     ("SSIM(mean)", lambda r, a: r["A"][obj][a]["mean"]["ssim"])):
        a = arr(lambda r: f(r, "std")); b = arr(lambda r: f(r, "v2"))
        g, nb, nt, p = paired(a, b)
        print(f"  {obj:7s} {label:11s}: {g:+.5f}  v2 better {nb}/{n} ties {nt}  p={fmt_p(p)}")
    if obj == "kapur":
        a = arr(lambda r: r["A"][obj]["std"]["even"]["psnr"]); b = arr(lambda r: r["A"][obj]["v2"]["even"]["psnr"])
        g, nb, nt, p = paired(a, b); print(f"  {obj:7s} {'PSNR(even)':11s}: {g:+.5f}  v2 better {nb}/{n} ties {nt}  p={fmt_p(p)}")
        a = arr(lambda r: r["A"][obj]["std"]["even"]["ssim"]); b = arr(lambda r: r["A"][obj]["v2"]["even"]["ssim"])
        g, nb, nt, p = paired(a, b); print(f"  {obj:7s} {'SSIM(even)':11s}: {g:+.5f}  v2 better {nb}/{n} ties {nt}  p={fmt_p(p)}")
print()
print("A3. Objective effect at the EXACT optimum (DP), mean repaint: otsu/hybrid vs kapur  [gain, #better, ties, p]")
for obj in ("otsu", "hybrid"):
    for m in ("psnr", "ssim"):
        a = arr(lambda r: r["A"]["kapur"]["dp"]["mean"][m]); b = arr(lambda r: r["A"][obj]["dp"]["mean"][m])
        g, nb, nt, p = paired(a, b); print(f"  {obj:7s} {m}: {g:+.4f}  better {nb}/{n} ties {nt}  p={fmt_p(p)}")
    a = arr(lambda r: r["A"]["kapur"]["dp"]["kapur"]); b = arr(lambda r: r["A"][obj]["dp"]["kapur"])
    print(f"  {obj:7s} Kapur cost: {(b - a).mean():+.4f} ({100 * (b - a).mean() / a.mean():+.2f}%)")
print()
print("A4. Same for ESMA v2 (what the user would actually run): otsu/hybrid vs kapur objective")
for obj in ("otsu", "hybrid"):
    for m in ("psnr", "ssim"):
        a = arr(lambda r: r["A"]["kapur"]["v2"]["mean"][m]); b = arr(lambda r: r["A"][obj]["v2"]["mean"][m])
        g, nb, nt, p = paired(a, b); print(f"  {obj:7s} {m}: {g:+.4f}  better {nb}/{n} ties {nt}  p={fmt_p(p)}")
print()
# correlation: Kapur gain vs PSNR/SSIM gain (v2 - std), kapur objective
dk = arr(lambda r: r["A"]["kapur"]["v2"]["kapur"] - r["A"]["kapur"]["std"]["kapur"])
for m in ("psnr", "ssim"):
    for rep in ("even", "mean"):
        dm = arr(lambda r: r["A"]["kapur"]["v2"][rep][m] - r["A"]["kapur"]["std"][rep][m])
        rho = spearmanr(dk, dm).correlation if np.std(dk) > 0 and np.std(dm) > 0 else float("nan")
        print(f"A5. Spearman(Kapur gain, {m} {rep} gain) = {rho:+.3f}")
print()

# ---------------------------------------------------------------- B
print("B. d sweep, kapur objective (gap/at-opt over 3 seeds x images; PSNR/SSIM seed 42, mean repaint)")
print(f"{'d':>2s} {'H*':>8s} | {'std gap':>8s} {'std@opt':>7s} | {'v2 gap':>8s} {'v2@opt':>7s} {'v2 it':>5s} | "
      f"{'PSNR std':>8s} {'PSNR v2':>8s} {'PSNR dp':>8s} | {'SSIM std':>8s} {'SSIM v2':>8s} {'SSIM dp':>8s} | "
      f"{'dKapur':>8s} {'p':>7s} {'dPSNR':>7s} {'p':>7s} {'dSSIM':>7s} {'p':>7s}")
for d in R["d_sweep"]:
    k = str(d)
    hs = arr(lambda r: r["B"][k]["hstar"])
    gs = np.concatenate([r["B"][k]["std"]["gaps"] for r in P]); gv = np.concatenate([r["B"][k]["v2"]["gaps"] for r in P])
    itv = np.concatenate([r["B"][k]["v2"]["iters"] for r in P])
    ps = arr(lambda r: r["B"][k]["std"]["psnr"]); pv = arr(lambda r: r["B"][k]["v2"]["psnr"]); pd = arr(lambda r: r["B"][k]["dp"]["psnr"])
    ss = arr(lambda r: r["B"][k]["std"]["ssim"]); sv = arr(lambda r: r["B"][k]["v2"]["ssim"]); sd = arr(lambda r: r["B"][k]["dp"]["ssim"])
    ks = arr(lambda r: r["B"][k]["std"]["kapur"]); kv = arr(lambda r: r["B"][k]["v2"]["kapur"])
    gk, _, _, pk = paired(ks, kv); gp, _, _, pp = paired(ps, pv); gsm, _, _, psm = paired(ss, sv)
    print(f"{d:2d} {hs.mean():8.4f} | {gs.mean():8.4f} {100*np.mean(gs < 1e-9):6.1f}% | {gv.mean():8.4f} {100*np.mean(gv < 1e-9):6.1f}% {itv.mean():5.1f} | "
          f"{ps.mean():8.3f} {pv.mean():8.3f} {pd.mean():8.3f} | {ss.mean():8.4f} {sv.mean():8.4f} {sd.mean():8.4f} | "
          f"{gk:+8.4f} {fmt_p(pk):>7s} {gp:+7.3f} {fmt_p(pp):>7s} {gsm:+7.4f} {fmt_p(psm):>7s}")
print()
print("B1. share of images where v2 beats std (seed 42) on PSNR / SSIM per d")
for d in R["d_sweep"]:
    k = str(d)
    ps = arr(lambda r: r["B"][k]["std"]["psnr"]); pv = arr(lambda r: r["B"][k]["v2"]["psnr"])
    ss = arr(lambda r: r["B"][k]["std"]["ssim"]); sv = arr(lambda r: r["B"][k]["v2"]["ssim"])
    print(f"  d={d:2d}: PSNR v2 better {int((pv > ps).sum())}/{n} ties {int((pv == ps).sum())} | SSIM v2 better {int((sv > ss).sum())}/{n} ties {int((sv == ss).sum())}"
          f" | worst std gap {np.max(np.concatenate([r['B'][k]['std']['gaps'] for r in P])):.4f} worst v2 gap {np.max(np.concatenate([r['B'][k]['v2']['gaps'] for r in P])):.4f}")
print()

# ---------------------------------------------------------------- C
PC = [r for r in P if "C" in r]
if PC:
    nc = len(PC)
    print(f"C. SSIM-aware objective on {nc} hold-out images (d=4, mean repaint). Reference = same image's kapur-objective v2 result from A.")
    for w in sorted(next(iter(PC))["C"].keys(), key=float):
        print(f"  weight w={w} (F = w*Kapur/Kapur* + (1-w)*SSIM_small)")
        ref_k = np.array([r["A"]["kapur"]["v2"]["kapur"] for r in PC]); ref_p = np.array([r["A"]["kapur"]["v2"]["mean"]["psnr"] for r in PC]); ref_s = np.array([r["A"]["kapur"]["v2"]["mean"]["ssim"] for r in PC])
        for alg in ("std", "v2"):
            kk = np.array([r["C"][w][alg]["kapur"] for r in PC]); pp_ = np.array([r["C"][w][alg]["psnr"] for r in PC]); ss_ = np.array([r["C"][w][alg]["ssim"] for r in PC])
            F = np.array([r["C"][w][alg]["F"] for r in PC]); wall = np.array([r["C"][w][alg]["wall"] for r in PC]); ne = np.array([r["C"][w][alg]["n_ssim_evals"] for r in PC]); it = np.array([r["C"][w][alg]["it"] for r in PC])
            print(f"    {alg:3s}: F={F.mean():.5f} Kapur={kk.mean():.4f} ({(kk-ref_k).mean():+.4f} vs kapur-v2) PSNR={pp_.mean():.3f} ({(pp_-ref_p).mean():+.3f}) "
                  f"SSIM={ss_.mean():.4f} ({(ss_-ref_s).mean():+.4f}) | iters {it.mean():.1f} ssim-evals {ne.mean():.0f} wall {wall.mean():.1f}s")
        for label, f in (("F", lambda r, a: r["C"][w][a]["F"]), ("Kapur", lambda r, a: r["C"][w][a]["kapur"]),
                         ("PSNR", lambda r, a: r["C"][w][a]["psnr"]), ("SSIM", lambda r, a: r["C"][w][a]["ssim"])):
            a = np.array([f(r, "std") for r in PC]); b = np.array([f(r, "v2") for r in PC])
            g, nb, nt, p = paired(a, b); print(f"    v2 vs std {label:5s}: {g:+.5f}  v2 better {nb}/{nc} ties {nt}  p={fmt_p(p)}")
        s_ref = ref_s; s_new = np.array([r["C"][w]["v2"]["ssim"] for r in PC]); g, nb, nt, p = paired(s_ref, s_new)
        print(f"    v2(SSIM-aware) vs v2(kapur) SSIM: {g:+.4f} better {nb}/{nc} p={fmt_p(p)}")
        p_ref = ref_p; p_new = np.array([r["C"][w]["v2"]["psnr"] for r in PC]); g, nb, nt, p = paired(p_ref, p_new)
        print(f"    v2(SSIM-aware) vs v2(kapur) PSNR: {g:+.3f} better {nb}/{nc} p={fmt_p(p)}")
