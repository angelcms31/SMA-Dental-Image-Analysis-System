"""Summarize exp_hybrid_d.json (hybrid objective in Kapur units, d = 4/8/10) and join it with the
Kapur-objective runs of exp_results.json (experiment B) on the same images.
Usage: python summarize_hybrid.py [exp_hybrid_d.json]"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
fn = sys.argv[1] if len(sys.argv) > 1 else "exp_hybrid_d.json"
H = json.load(open(os.path.join(HERE, fn)))
R = json.load(open(os.path.join(HERE, "exp_results.json")))
byfile = {r["file"]: r for r in R["per_image"]}
P = [r for r in H["per_image"] if r["file"] in byfile]
n = len(P)


def pw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return d.mean(), int((d > 0).sum()), int((d == 0).sum()), p


def fp(p):
    return "n/a" if np.isnan(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


print(f"===== {fn}: hybrid objective, n={n} images, N={H['N']} T={H['T']} seed={H['seed']}, gap seeds {H['gap_seeds']} =====")
for d in H["ds"]:
    k = str(d)
    print(f"\n--- d = {d} ---")
    fstar = np.array([r[k]["fstar"] for r in P])
    print(f"  exact hybrid optimum F* mean {fstar.mean():.4f} (Kapur units); norms H_K* {np.mean([r[k]['norms'][0] for r in P]):.4f}, sigma_B2* {np.mean([r[k]['norms'][1] for r in P]):.1f}")
    print(f"  {'algo':4s} {'Kapur':>8s} {'sigmaB2':>8s} {'PSNR':>7s} {'SSIM':>7s} {'gap(3 seeds)':>12s} {'at opt':>7s} {'iters':>6s}")
    for alg in ("std", "v2", "dp"):
        kap = np.array([r[k][alg]["kapur"] for r in P]); sb = np.array([r[k][alg]["sigma_B2"] for r in P])
        ps = np.array([r[k][alg]["psnr"] for r in P]); ss = np.array([r[k][alg]["ssim"] for r in P])
        if alg == "dp":
            print(f"  {alg:4s} {kap.mean():8.4f} {sb.mean():8.1f} {ps.mean():7.3f} {ss.mean():7.4f} {'0':>12s} {'100%':>7s} {'-':>6s}")
        else:
            g = np.concatenate([r[k][alg]["gaps"] for r in P]); it = np.concatenate([r[k][alg]["iters"] for r in P])
            print(f"  {alg:4s} {kap.mean():8.4f} {sb.mean():8.1f} {ps.mean():7.3f} {ss.mean():7.4f} {g.mean():12.5f} {100*np.mean(g < 1e-9):6.1f}% {it.mean():6.1f}")
    print("  ESMA v2 vs Standard SMA (paired, seed 42):")
    for label, key in (("F (hybrid)", "F"), ("Kapur", "kapur"), ("PSNR", "psnr"), ("SSIM", "ssim")):
        g, nb, nt, p = pw([r[k]["std"][key] for r in P], [r[k]["v2"][key] for r in P])
        print(f"    {label:11s}: {g:+.4f}  v2 better {nb}/{n} ties {nt}  p={fp(p)}")
    # join with the Kapur-objective runs at the same d (experiment B of exp_results.json)
    if k in byfile[P[0]["file"]]["B"]:
        print("  Hybrid objective vs Kapur objective, same algorithm, same images (class-mean paint):")
        for alg in ("v2", "std"):
            for label, key in (("PSNR", "psnr"), ("SSIM", "ssim"), ("Kapur", "kapur")):
                a = [byfile[r["file"]]["B"][k][alg][key] for r in P]; b = [r[k][alg][key] for r in P]
                g, nb, nt, p = pw(a, b)
                print(f"    {alg:3s} {label:5s}: hybrid - kapur = {g:+.4f}  better {nb}/{n}  p={fp(p)}")
        # the headline: ESMA v2 (hybrid) vs Standard SMA (Kapur, thesis baseline)
        print("  ESMA v2 with hybrid objective vs Standard SMA with Kapur objective (the thesis baseline):")
        for label, key in (("Kapur", "kapur"), ("PSNR", "psnr"), ("SSIM", "ssim")):
            a = [byfile[r["file"]]["B"][k]["std"][key] for r in P]; b = [r[k]["v2"][key] for r in P]
            g, nb, nt, p = pw(a, b)
            print(f"    {label:5s}: {g:+.4f}  v2(hybrid) better {nb}/{n}  p={fp(p)}")
