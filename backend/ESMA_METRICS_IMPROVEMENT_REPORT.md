# Enhanced SMA — why the Kapur / PSNR / SSIM gains are small, and what actually moves them

13 September 2026. Code: `backend/sma_algorithms.py` (class-mean repaint, Otsu/hybrid objective tables, `objective`
keyword on all optimizers), `backend/compare_three_way.py` (`--recon`, `--objective`), `backend/main.py` (`levels`,
`objective` form fields), `backend/tests/test_objectives_repaint.py` (5 new tests; 16/16 pass),
`backend/experiments/` (scripts + raw per-image JSON for every number below). Companion to `ESMA_V2_REPORT.md`.

## 0. Summary

The question was whether Kapur's entropy, PSNR and SSIM can be improved substantially beyond the small gains the
Enhanced SMA (ESMA v2) currently shows over Standard SMA. The answer has three parts.

1. **The small Kapur gain is a ceiling, not a weakness.** At d = 4 the exact global maximum of Kapur's entropy is
   only 0.003 above Standard SMA (19.2600 vs 19.2569 on the 300 DENTEX images) and ESMA v2 already takes 94% of that
   headroom. No optimizer can do better; the defensible statement is the hit rate (82–92% of images at the exact
   optimum vs 1–4%). The advantage becomes large only on a harder problem: at d = 10–12 thresholds it is 60–100× bigger
   (+0.11 to +0.17 Kapur, p < 1e-12).
2. **PSNR and SSIM never responded to Kapur optimization because Kapur's entropy does not target them.** The exact
   Kapur optimum has the same PSNR/SSIM as Standard SMA (20.015 dB vs 20.003; 0.4975 vs 0.4959 on DENTEX). Otsu's
   criterion, by contrast, is *identical* to maximizing PSNR of the class-mean segmented image (σ_W² = MSE), which the
   literature confirms empirically (Otsu–PSNR correlation 0.99, Kapur–PSNR 0.68).
3. **Three levers move PSNR/SSIM, and all three were implemented and measured on 100 OPGs:**

| Lever | Effect on PSNR | Effect on SSIM | Effect on Kapur | Applies to |
|---|---|---|---|---|
| Class-mean repaint instead of fixed levels 0/64/128/191/255 (identical thresholds) | **+4.8 dB**, 100/100 images | **+0.113**, 100/100 images | none | every algorithm |
| Hybrid Kapur/Otsu objective (w = 0.5) / pure Otsu | **+0.36 dB / +0.47 dB** | +0.003 / +0.004 | −0.16% / −0.54% | every algorithm |
| d = 8–12 thresholds instead of 4 | +5 to +8 dB absolute | 0.73 → 0.82 absolute | (not comparable across d) | every algorithm |
| **ESMA v2 over Standard SMA, Kapur objective** | d = 4: +0.001 (n.s.) → **d = 10: +0.32 dB (75/100, p = 1e-9)** | d = 4: n.s. → **d = 10: +0.0021 (68/100, p = 5e-4)** | +0.0018 → **+0.114 (p = 1e-13)** | the thesis comparison |
| **ESMA v2 with hybrid objective vs the Kapur Standard-SMA baseline** | **+0.30 dB (d = 8), +0.45 dB (d = 10)**, 84–87/100 | **+0.0019 (d = 8), +0.0037 (d = 10)**, 63–65/100 | **+0.047 (d = 8), +0.061 (d = 10)**, 70–83/100 | recommended protocol |
| SSIM-aware non-separable objective (d = 4, 20 images) | +0.19 to +0.25 dB vs pure Kapur | **+0.006 to +0.009** vs pure Kapur (19–20/20) | −0.2% to −0.6% | ESMA v2 better than Standard on 19–20/20 at 40% of the time |

A new finding surfaced on the way: **Standard SMA's exploitation gate `p = tanh|S(i) − DF|` makes its performance
depend on the numeric scale of the objective** (hit rate 1% on Kapur's entropy at its natural scale ≈ 19, 69% when the
same objective is multiplied by 3500, 0% when divided by 19). ESMA v2 is exactly scale-invariant (84% at every scale).
This is a fourth structural weakness of Standard SMA the thesis can name, and it is why the hybrid objective is
expressed in Kapur units before comparing the two algorithms.

Sections: 1 current numbers · 2 problems · 3 code changes · 4 experiments · 5 answers and recommended protocol ·
6 literature.

---

## 1. Starting point: what the current numbers actually say

`backend/three_way_300.json` (300 DENTEX OPGs, N = 30, T = 100, d = 4, seed 42, `apply_thresholds` with the original
fixed band levels) is the benchmark behind the "large runtime gain, small Kapur/PSNR/SSIM gain" observation. The one
column that is usually left out of that table is the **exact Kapur optimum** that the same file already contains
(`kapur_optimal_thresholds`, dynamic programming, the true global maximum for d = 4):

| Metric | Standard SMA | ESMA v2 | **Exact Kapur optimum (DP)** | ESMA v2 vs Standard (paired Wilcoxon) |
|---|---|---|---|---|
| Kapur's entropy | 19.2569 | 19.2598 | **19.2600** | +0.0030, better in 292/300 (97.3%), p = 2e-49 |
| Images at the exact optimum | 4 / 300 (1.3%) | 245 / 300 (81.7%) | 300 / 300 | — |
| Mean gap to optimum | 0.00317 | 0.00020 | 0 | — |
| PSNR (dB) | 20.003 | 20.024 | **20.015** | +0.021, better in 151/300 (50.3%), p = 0.41 |
| SSIM | 0.4959 | 0.4974 | **0.4975** | +0.0015, better in 172/300 (57.3%), p = 0.002 |
| Runtime (ms) | 31.7 | 18.0 | ≈ 7 (DP) | −13.7 ms, faster in 295/300, p = 1e-48 |

Three facts in this table explain the whole "small improvement" pattern:

* **The Kapur ceiling is 0.003 above Standard SMA.** No algorithm can score above 19.2600 at d = 4; ESMA v2 already
  captures 94% of the distance between Standard SMA and that ceiling (0.0030 of 0.0032). The Kapur gain cannot be made
  "substantial" by any optimizer change whatsoever.
* **A perfect Kapur optimizer has the same PSNR/SSIM as Standard SMA.** The exact optimum scores 20.015 dB / 0.4975,
  i.e. within noise of Standard SMA (20.003 / 0.4959) and actually *below* ESMA v2 on PSNR. Kapur's entropy is not a
  proxy for PSNR or SSIM on these images; the 60-image hold-out study in `ESMA_V2_REPORT.md` measured the per-image
  correlation between Kapur gain and PSNR gain at r = +0.006.
* **The per-image PSNR difference is pure noise around a zero effect.** The paired PSNR gain has mean +0.021 dB and
  standard deviation 0.319 dB (signal-to-noise 0.07). Detecting an effect of that size at p < 0.05 would need on
  the order of 900 paired images, and the live frontend runs are unseeded (the ranking document measured a 0.006–0.023
  Kapur spread for the *same* algorithm on the *same* image, larger than the median per-image gain of 0.0026).

## 2. The specific problems behind the small gains

### P1 — Saturation: Kapur's entropy is solved at d = 4

Kapur's objective is a sum of independent per-class entropies, so for d thresholds on 256 levels its global maximum is
computable exactly in O(d·L²) (Luessi et al. 2006/2009; `kapur_optimal_thresholds`). With N = 30 agents and
T = 100 iterations both optimizers sit within 0.02% of that maximum. The remaining error of Standard SMA is a
*polishing* error of 1–12 gray levels inside the right basin (median L∞ threshold error 3 levels, never ≥ 20). ESMA v2
removes most of it, which is exactly the "+0.003, 97% of images, p = 1e-49" result — statistically decisive, numerically
invisible. Presenting the improvement as raw Kapur units hides it; presenting it as **gap to the exact optimum** and
**hit rate** (1.3% → 81.7% of images at the true optimum) shows it.

### P2 — Objective–metric mismatch: Kapur's entropy does not target PSNR or SSIM

PSNR of a thresholded image is determined by the mean-squared error between the original and the repainted bands.
For fixed thresholds the MSE-minimizing repaint is the class mean, and its MSE is Otsu's within-class variance:

    MSE_mean-repaint = Σ_k ω_k σ_k² = σ_W² = σ_T² − σ_B²      →      PSNR = 10·log10(255² / σ_W²)

So **maximizing Otsu's between-class variance σ_B² is literally maximizing the PSNR of the class-mean segmented image.**
Kapur's criterion Σ_k H_k maximizes the information content of the class histograms instead; it has no algebraic
relation to reconstruction error, which is why the exact Kapur optimum does not beat Standard SMA on PSNR. The
literature measures the same thing empirically: on BSDS500, Otsu's criterion correlates with PSNR at r = 0.987 ± 0.008
and Kapur's at 0.68 ± 0.19 (Otsu higher on 100% of images); for SSIM 0.896 vs 0.674 (Otsu higher on 91% of images)
[arXiv 2605.27132]. SSIM additionally depends on *spatial* structure (local means, variances and covariances in 7×7
windows), which no histogram-only criterion — Kapur, Otsu or any entropy — can see at all.

### P3 — The repaint convention throws away ~4 dB for everyone

`apply_thresholds` painted the five bands with the fixed levels 0, 64, 128, 191, 255 regardless of the image. Those
values are chosen for on-screen contrast, not fidelity: a band whose pixels average 26 is painted 0, a band averaging
225 is painted 255. Under this convention PSNR depends on where the band edges fall relative to five arbitrary
constants — a quantity Kapur's criterion does not target and that moves by ±0.5 dB when a threshold shifts by a few
levels (hence the 0.32 dB noise). The convention in the multilevel-thresholding literature is the class mean ("each
pixel is assigned the mean gray value of its class", Arora et al. 2008), which is the minimum-MSE repaint for any
threshold vector and leaves the thresholds, the fitness and every pixel's class membership untouched. On a single OPG
(Mendeley 1008) the same Kapur-optimal thresholds score 20.06 dB / 0.573 with fixed levels and 24.36 dB / 0.698 with
class means (+4.3 dB, +0.125 SSIM). Section 4 quantifies this on all 100 images.

### P4 — d = 4 is too easy a problem to separate two good optimizers

Every SMA thresholding paper in the reference folder observes the same pattern: algorithms are indistinguishable at 4–6
thresholds and separate at 8–18. The Lévy/QOBL ESMA paper reports "only small differences between the ESMA and other
compared algorithms in threshold values 4 and 6 … the PSNR values significantly increase when the threshold values are
increasing" and "when the threshold is equal to 4, the SSIM results of each algorithm are roughly the same"; SMA-MLS
reports "the larger the number of thresholds, the more obvious the effect is" with the Friedman χ² rising from 18.4
(TH = 2) to 28.6 (TH = 14); Hosny et al. 2022 benchmark at K = 6…26. With d = 4 and the Standard SMA already at
gap 0.003, there is no room for *any* enhanced optimizer to show a large gain in *any* metric. Section 4 measures how
the gap, the hit rate and the PSNR/SSIM gains evolve for d = 4…12 on the OPG data.

### P5 — Evaluation noise dominates the effect sizes being measured

A single seeded run per image and per algorithm is the protocol of `compare_three_way.py`; the frontend comparison is
unseeded. With PSNR gain σ = 0.32 dB and SSIM gain σ = 0.008 against mean effects of 0.02 dB and 0.0015, the per-image
signs are coin flips (50.3% / 57.3%). Nothing algorithmic fixes this; only (a) a larger true effect (P2–P4) or (b)
averaging several seeds per image and reporting paired statistics with confidence intervals does.

### What the problems are *not*

* They are not caused by the ESMA update rules being weak. The optimality-gap analysis shows ESMA v2 reaches the exact
  Kapur optimum on 82–88% of images and is within 0.0002 of it on average; the remaining 18% are basin-level misses
  worth ≈ 0.001 Kapur each.
* They are not caused by the fitness implementation (table evaluation agrees with the scalar Kapur function to 1e-14).
* They are not fixable by more iterations or agents: the equal-budget run (`esma_v2_fullT`, T = 100) moves the mean
  Kapur by 0.00003 and PSNR by −0.002 dB.

## 3. What was changed in the code (all backward compatible; 16/16 tests pass)

Every default is unchanged: `standard_sma`, `enhanced_sma`, `enhanced_sma_v2` and `apply_thresholds` return
byte-identical results to the previous code unless one of the new options is used (verified against a frozen copy of
the module on the 100 reference histograms and in `tests/test_objectives_repaint.py`).

| ID | Change | Where | Addresses |
|---|---|---|---|
| S1 | **Class-mean repaint** `apply_thresholds(img, thr, levels="mean")` (+ `band_levels`). Each band is painted with the mean intensity of its own pixels — the minimum-MSE repaint and the literature's convention for PSNR/SSIM. `levels="even"` (default) is the original behaviour. | `sma_algorithms.py` | P3 |
| S2 | **Objective choice.** `BetweenClassVarianceTable` (Otsu's σ_B², same interface as `KapurEntropyTable`), `HybridObjectiveTable` (w·Kapur/Kapur* + (1−w)·σ_B²/σ_B²*, both terms normalized by their own exact optimum so the weight is meaningful), `make_objective_table`, `optimal_thresholds` (exact DP optimum of *any* separable table). All three optimizers take `objective="kapur"|"otsu"|"hybrid"` or **any object with `evaluate(X, assume_sorted)`**, so a custom, non-separable objective can be plugged in without touching the SMA code. | `sma_algorithms.py` | P2, P4 |
| S3 | **Benchmark protocol flags.** `compare_three_way.py --recon even|mean --objective kapur|otsu|hybrid`; the JSON stores `objective_value` and the true Kapur value of every threshold vector separately, and the DP reference is the optimum of the chosen objective. | `compare_three_way.py` | P3, P5 |
| S4 | **API pass-through.** `/analyze/standard|enhanced|improved|compare` accept `levels` and `objective` form fields; responses gain `objective`, `objective_value`, `band_levels`; `kapur_entropy_fitness` is always Kapur's entropy of the returned thresholds. Frontend unchanged (defaults). | `main.py` | — |
| S5 | **SSIM-aware objective prototype** `KapurSSIMObjective` (w·Kapur/Kapur* + (1−w)·SSIM on an 8× down-sampled class-mean repaint, memoised per integer threshold vector). Non-separable, so no DP optimum exists and the metaheuristic is genuinely required. Kept in the experiment script pending the results in Section 4. | `exp_metrics.py` (scratch) | P2 (SSIM) |
| S6 | **Tests** for S1–S2: even repaint == legacy, mean repaint == band means and never worse MSE, Otsu table == brute force, DP optimum == exhaustive maximum (d = 2), hybrid == normalized weighted sum, objective defaults leave all optimizers unchanged, legacy path rejects non-Kapur objectives. | `tests/test_objectives_repaint.py` | — |

Why S2 is sound as an *SMA* objective: Otsu's criterion is a sum over classes just like Kapur's (class k contributes
P_k(μ_k − μ)²), so the whole evaluation infrastructure — the 257×257 table, one vectorized evaluation per generation,
the exact DP reference, the optimality-gap reporting — carries over unchanged. The hybrid is a linear combination of
two such tables and therefore also separable. The maximize-convention is preserved (σ_B² larger is better), so the
greedy selection, leader ranking and the CR(t)/PD(t) feedback signals of ESMA v2 work exactly as before.

Usage:

```
# class-mean repaint, Kapur objective (thesis objective, literature evaluation convention)
python compare_three_way.py --images_dir <xrays> --subset all --max_images 300 --seed 42 --skip_v1 --recon mean

# hybrid Kapur/Otsu objective, class-mean repaint, 8 thresholds
python compare_three_way.py --images_dir <xrays> --subset all --max_images 300 --seed 42 --skip_v1 \
    --recon mean --objective hybrid --d 8 --out three_way_300_hybrid_d8.json

# in code
from sma_algorithms import enhanced_sma_v2, apply_thresholds, optimal_thresholds
r = enhanced_sma_v2(prob, d=8, N=30, T=100, seed=42, objective="hybrid")
seg = apply_thresholds(image, r["thresholds"], levels="mean")
f_star, thr_star = optimal_thresholds(prob, 8, "hybrid")        # exact ceiling for the gap/hit-rate report
```

## 4. Experiments

All runs on this machine: the 100 Mendeley panoramic OPGs in `Datasets/images` (the DENTEX folder is not present
here), N = 30, T = 100, d = 4 unless stated, seed 42 for the metric runs, seeds 1–3 for optimality-gap statistics;
ESMA v2 with its defaults (adaptive termination on). PSNR/SSIM are computed with a vectorized re-implementation of
scikit-image's defaults that agrees with `metrics.py` to 1e-14 (checked on 12 image/threshold pairs). Scripts:
`backend/experiments/exp_metrics.py`, `exp_hybrid_d.py`, `summarize_exp.py`; raw per-image results in the JSON files
next to them. Where the 60-image hold-out split differs materially from the all-100 numbers it is stated.

### 4.1 Repaint (S1): identical thresholds, class-mean paint instead of fixed levels

| Thresholds from | PSNR, fixed levels | PSNR, class means | Δ PSNR | SSIM, fixed levels | SSIM, class means | Δ SSIM |
|---|---|---|---|---|---|---|
| Standard SMA | 19.913 | 24.707 | **+4.79 dB** (100/100) | 0.6126 | 0.7255 | **+0.113** (100/100) |
| ESMA v2 | 19.894 | 24.708 | **+4.81 dB** (100/100) | 0.6122 | 0.7256 | **+0.113** (100/100) |
| Exact Kapur optimum | 19.907 | 24.696 | +4.79 dB (100/100) | 0.6132 | 0.7256 | +0.112 (100/100) |

Wilcoxon p = 3.9e-18 for every row. The repaint is worth 5 dB — two orders of magnitude more than any difference
between optimizers — and it applies to every algorithm equally, so it raises the absolute PSNR/SSIM the thesis reports
without changing the Standard-vs-Enhanced comparison. Under the class-mean paint the spurious ±0.5 dB dependence on
fixed levels disappears; what is left (ESMA v2 − Standard = +0.0006 dB, 45/100 images) is the true zero effect of P2.
The Otsu identity of Section 2 (PSNR of the class-mean image = 10·log10(255²/(σ_T² − σ_B²))) holds numerically to
0.002 dB (residual = rounding of the band means to integers).

### 4.2 Objective (S2) at d = 4: what PSNR/SSIM the three criteria buy, and what Kapur they cost

Exact optimum of each objective (DP), class-mean paint, 100 images:

| Objective optimized | Kapur's entropy of the result | PSNR | SSIM | vs Kapur objective |
|---|---|---|---|---|
| Kapur (thesis) | 19.0750 | 24.696 | 0.7256 | — |
| Otsu (σ_B²) | 18.9721 (−0.54%) | 25.161 | 0.7296 | **+0.465 dB** (100/100, p = 4e-18), **+0.0040 SSIM** (71/100, p = 6e-9) |
| Hybrid w = 0.5 | 19.0441 (−0.16%) | 25.069 | 0.7284 | **+0.372 dB** (99/100, p = 6e-18), **+0.0028 SSIM** (73/100, p = 4e-9) |

ESMA v2 run with each objective reproduces these deltas to within 0.01 dB (+0.454 / +0.0040 with Otsu, +0.361 /
+0.0028 with the hybrid). The hybrid gives 80% of Otsu's PSNR gain for 30% of its Kapur cost, which is why it is the
recommended compromise.

ESMA v2 versus Standard SMA under each objective (100 images, seed 42):

| Objective | At exact optimum: Standard / ESMA v2 | Δ objective | Δ Kapur | Δ PSNR (class means) | Δ SSIM (class means) |
|---|---|---|---|---|---|
| Kapur | 1% / 84% | +0.0018 (97/100, p = 4e-16) | +0.0018 | +0.0006 (45/100, p = 0.25) | +0.0001 (50/100, p = 0.74) |
| Otsu | 67% / 86% | +9.7 (31/100, 60 ties, p = 0.001) | +0.0008 (n.s.) | +0.0002 (27/100, 62 ties, p = 0.02) | −0.00001 (n.s.) |
| Hybrid, normalized [0, 1] form | **0% / 92%** | +0.00088 (**100/100**, p = 4e-18) | **+0.0058** (60/100, p = 0.047) | **+0.099 dB** (82/100, p = 2e-11) | **+0.0015** (68/100, p = 4e-6) |

Spearman correlation between the per-image Kapur gain and the PSNR gain (Kapur objective): −0.14 (class means), +0.04
(fixed levels); with SSIM: −0.04 / +0.20. Better Kapur optimization is uncorrelated with PSNR/SSIM (P2 again).

### 4.3 Why the hybrid separates the two algorithms — the numeric-scale sensitivity of Standard SMA

The hybrid row above is the first configuration in which ESMA v2 beats Standard SMA on all three metrics at d = 4.
Before recommending it, the reason had to be understood. Standard SMA decides per agent between exploiting
(`Xb + vb·(W·XA − XB)`) and contracting (`vc·X`) with probability `p = tanh|S(i) − DF|` computed on **raw** fitness
values. With Kapur's entropy (≈ 19, differences 0.01–1) p is 0.01–0.76; with the normalized hybrid (≈ 1, differences
0.001–0.02) p is ≈ 0.001–0.02, so almost every agent proposes a contraction, greedy selection rejects it, and the
population barely moves. ESMA v2 has no such gate (every agent takes the multi-leader move; weights are rank-based), so
it is invariant to the objective's scale. Measured (d = 4, 100 images, seed 42, same optimum, same thresholds space):

| Objective × scale factor | Standard SMA: mean gap / at optimum | ESMA v2: mean gap / at optimum |
|---|---|---|
| Kapur × 1/19 (values ≈ 1) | 0.0140 / 0% | 0.0007 / 84% |
| Kapur × 1 (natural, ≈ 19) | 0.0025 / 1% | 0.0007 / 84% |
| Kapur × 3500 (≈ Otsu's scale) | 0.0004 / 69% | 0.0007 / 84% |
| Hybrid × 1 (normalized) | 0.00088 / 0% | 0 / 92% |
| Hybrid × 20 (≈ Kapur's scale) | 0.00016 / 1% | 0 / 92% |
| Hybrid × 3500 | 0.00001 / 61% | 0 / 92% |

(gaps in the objective's own units, divided back by the scale factor.) Two consequences:

* **A fourth structural weakness of Standard SMA that the thesis can name and ESMA removes.** Most of Standard
  SMA's polishing failure on Kapur's entropy is the scale-dependent `tanh` gate, not the single leader or the random
  initialization: multiplying the *same* objective by 3500 lifts its hit rate from 1% to 69%. Jiang et al. 2023 fix
  the same defect in their improved SMA by normalizing the gate to `tanh(10·|bF − S(i)| / |bF − wF|)`. ESMA v2's
  unconditional multi-leader move and rank weights make it scale-free, which is a defensible, testable claim.
* **Fairness.** A normalized hybrid (values ≈ 1) handicaps the baseline in a way the thesis's own Kapur protocol does
  not. The hybrid objective is therefore expressed **in Kapur units** by default (`HybridObjectiveTable(scale="kapur")`,
  F* ≈ H_K* ≈ 19), so Standard SMA's gate sees the same numeric scale as with Kapur's entropy. Section 4.5 re-measures
  the hybrid in that form. Even at Otsu's scale (× 3500) Standard SMA reaches the hybrid optimum on 61% of images
  against ESMA v2's 92%, so the algorithmic advantage survives the fairness correction.

### 4.4 Number of thresholds (S3), Kapur objective, class-mean paint

Gap and hit rate over 3 seeds × 100 images; PSNR/SSIM for seed 42; Δ = ESMA v2 − Standard SMA (paired Wilcoxon):

| d | H* (exact) | Standard SMA gap / at opt | ESMA v2 gap / at opt / iterations | PSNR: Standard / ESMA v2 / exact | SSIM: Standard / ESMA v2 / exact | Δ Kapur (p) | Δ PSNR (p) | Δ SSIM (p) |
|---|---|---|---|---|---|---|---|---|
| 4 | 19.075 | 0.0042 / 3.7% | 0.0001 / 89.3% / 40 | 24.707 / 24.708 / 24.696 | 0.7255 / 0.7256 / 0.7256 | +0.0018 (4e-16) | +0.001 (0.25) | +0.0001 (0.74) |
| 6 | 24.411 | 0.0214 / 0% | 0.0021 / 38.7% / 57 | 27.499 / 27.569 / 27.537 | 0.7479 / 0.7478 / 0.7476 | +0.023 (2e-17) | +0.070 (0.13) | −0.0001 (0.81) |
| 8 | 29.209 | 0.0740 / 0% | 0.0077 / 6.7% / 76 | 29.594 / 29.687 / 29.673 | 0.7723 / 0.7723 / 0.7726 | **+0.055 (7e-15)** | **+0.093 (0.002)**, 63/100 | +0.0000 (0.86) |
| 10 | 33.573 | 0.1731 / 0% | 0.0459 / 1.3% / 87 | 31.125 / 31.444 / 31.484 | 0.7967 / 0.7988 / 0.7984 | **+0.114 (1e-13)** | **+0.319 (1e-9)**, 75/100 | **+0.0021 (5e-4)**, 68/100 |
| 12 | 37.587 | 0.2809 / 0% | 0.1306 / 0% / 87 | 32.581 / 32.828 / 32.966 | 0.8203 / 0.8226 / 0.8229 | **+0.167 (2e-12)** | **+0.247 (1e-5)**, 68/100 | **+0.0023 (0.03)**, 57/100 |

Reading:

* The Kapur gain grows from +0.0018 (d = 4) to +0.167 (d = 12) — **~100×** — because Standard SMA's gap grows 70×
  while ESMA v2's grows less. Hit rates drop for both (the exact optimum is a needle in 255^d/d! at d = 12), but ESMA v2
  keeps less than half the gap of Standard SMA at every d.
* PSNR and SSIM become *significantly* better for ESMA v2 from d = 8 (PSNR) and d = 10 (SSIM), with 63–75% of images
  won; at d = 10 the PSNR gain is +0.32 dB. Absolute PSNR rises from 24.7 dB (d = 4) to 32.6–33.0 dB (d = 12), SSIM from
  0.73 to 0.82, which is the "substantial improvement" of these metrics the literature reports at high K.
* At d ≥ 10 ESMA v2 itself is no longer near-optimal (gap 0.05–0.13, hit rate ≤ 1%, and the patience-20 stop fires
  at ≈ 87 of 100 iterations). This is the regime where the SMA enhancements — and the currently disabled z(t)
  exploration channel and the opt-in polish — can be expected to matter and should be re-ablated.

### 4.5 Hybrid objective in Kapur units (the recommended form of S2), d = 4, 8, 10

Same protocol as 4.4, hybrid w = 0.5 expressed in Kapur units (`HybridObjectiveTable(scale="kapur")`, F* ≈ H_K*), so
Standard SMA's exploitation gate sees the same numeric scale as with the thesis objective. Class-mean paint.

| d | Algorithm | Kapur's entropy | σ_B² | PSNR | SSIM | Gap to exact hybrid optimum (3 seeds) | At optimum | Iterations |
|---|---|---|---|---|---|---|---|---|
| 4 | Standard SMA | 19.0425 | 3428.9 | 25.053 | 0.7278 | 0.0032 | 3.0% | 100 |
| 4 | **ESMA v2** | 19.0442 | 3429.6 | 25.068 | 0.7284 | **0.00002** | **90.3%** | 40 |
| 4 | exact optimum | 19.0441 | 3429.6 | 25.069 | 0.7284 | 0 | 100% | — |
| 8 | Standard SMA | 29.0927 | 3565.0 | 29.795 | 0.7749 | 0.0545 | 0% | 100 |
| 8 | **ESMA v2** | 29.1843 | 3566.7 | 29.893 | 0.7742 | **0.0070** | 5.7% | 75 |
| 8 | exact optimum | 29.1955 | 3567.0 | 29.910 | 0.7745 | 0 | 100% | — |
| 10 | Standard SMA | 33.3545 | 3584.2 | 31.219 | 0.7987 | 0.1039 | 0% | 100 |
| 10 | **ESMA v2** | 33.4707 | 3588.3 | 31.578 | 0.8004 | **0.0206** | 1.0% | 84 |
| 10 | exact optimum | 33.5658 | 3588.9 | 31.635 | 0.7999 | 0 | 100% | — |

ESMA v2 versus Standard SMA, both on the hybrid objective (paired, seed 42, 100 images):

| d | Δ hybrid F | Δ Kapur | Δ PSNR | Δ SSIM |
|---|---|---|---|---|
| 4 | +0.0029 (97/100, p = 1e-17) | +0.0017 (56/100, p = 0.085) | **+0.015 dB (60/100, p = 0.002)** | **+0.0005 (61/100, p = 0.001)** |
| 8 | +0.052 (96/100, p = 9e-18) | **+0.092 (92/100, p = 2e-16)** | **+0.098 dB (65/100, p = 3e-4)** | −0.0007 (43/100, p = 0.11) |
| 10 | +0.077 (85/100, p = 5e-13) | **+0.116 (84/100, p = 3e-11)** | **+0.359 dB (84/100, p = 3e-11)** | +0.0017 (57/100, p = 0.19) |

ESMA v2 with the hybrid objective versus **Standard SMA with Kapur's entropy — the thesis baseline** (both with
class-mean paint), i.e. the comparison a reader of Chapter 4 would actually make:

| d | Δ Kapur | Δ PSNR | Δ SSIM |
|---|---|---|---|
| 4 | −0.028 (7/100, p = 2e-17) — the hybrid's Kapur cost exceeds the optimizer's Kapur gain at d = 4 | **+0.361 dB (100/100, p = 4e-18)** | **+0.0028 (74/100, p = 2e-8)** |
| 8 | **+0.047 (83/100, p = 6e-12)** | **+0.299 dB (87/100, p = 2e-13)** | **+0.0019 (65/100, p = 2e-4)** |
| 10 | **+0.061 (70/100, p = 9e-6)** | **+0.453 dB (84/100, p = 2e-12)** | **+0.0037 (63/100, p = 9e-5)** |

Reading:

* In Kapur units the d = 4 hybrid advantage of ESMA v2 over Standard SMA is real but small (+0.015 dB, +0.0005 SSIM,
  60–61% of images). The +0.099 dB / 82% of Section 4.2 was mostly Standard SMA's scale sensitivity, not ESMA's
  merit; the Kapur-unit numbers are the ones to cite. (Check: in the normalized form Standard SMA's PSNR at d = 8/10 was
  0.6–1.5 dB *below* its own Kapur-objective result; in Kapur units it is 0.1–0.2 dB above it, as it should be.)
* From d = 8 the hybrid-objective ESMA v2 beats the thesis baseline on **all three metrics simultaneously**, with
  63–87% of images won and p ≤ 2e-4 everywhere; at d = 10 the margins are +0.06 Kapur, +0.45 dB, +0.0037 SSIM.
* Switching the objective changes the *thresholds*, not the optimizer: ESMA v2's hit rate on the hybrid optimum
  (90% at d = 4) matches its hit rate on the Kapur optimum (84–89%), while Standard SMA stays at 3% / 0% / 0%.

### 4.6 SSIM-aware, non-separable objective (S5) — 20 hold-out images, d = 4, class-mean paint

`F = w·H_K/H_K* + (1 − w)·SSIM(original, class-mean repaint)`, SSIM evaluated on an 8× down-sampled image and memoised
per integer threshold vector (≈ 1 100–2 800 SSIM evaluations per run). No exact optimum exists for this objective, so the
metaheuristic is genuinely required. Reference = the same image's ESMA v2 result with the pure Kapur objective.

| w | Algorithm | F | Kapur's entropy (vs Kapur-v2) | PSNR (vs Kapur-v2) | SSIM (vs Kapur-v2) | Iterations | SSIM evals | Wall time |
|---|---|---|---|---|---|---|---|---|
| 0.5 | Standard SMA | 0.84945 | 18.959 (−0.111) | 24.516 (+0.138) | 0.7231 (+0.0081) | 100 | 2 833 | 37.3 s |
| 0.5 | **ESMA v2** | **0.85182** | 18.961 (−0.108) | 24.565 (+0.187) | **0.7242 (+0.0091)** | 43 | 1 186 | **14.8 s** |
| 0.8 | Standard SMA | 0.93746 | 19.033 (−0.037) | 24.520 (+0.142) | 0.7187 (+0.0037) | 100 | 2 833 | 32.2 s |
| 0.8 | **ESMA v2** | **0.93881** | 19.030 (−0.039) | 24.631 (+0.254) | **0.7210 (+0.0060)** | 44 | 1 112 | **13.1 s** |

Paired tests (n = 20): ESMA v2 beats Standard SMA on the objective F in 20/20 (w = 0.5, p = 2e-6) and 19/20 (w = 0.8,
p = 8e-5) images at 40% of the wall time; on SSIM itself +0.0011 (13/20, p = 0.15) and +0.0023 (13/20, p = 0.044).
Relative to the pure-Kapur ESMA v2 result the SSIM-aware run raises SSIM on 19/20 (w = 0.5, +0.0091, p = 4e-6) and
20/20 images (w = 0.8, +0.0060, p = 2e-6) and PSNR on 16/20 (+0.19 dB) and 18/20 (+0.25 dB), at a Kapur cost of
0.2–0.6%. The SSIM gain is bounded by the 5-band quantization itself (SSIM 0.72–0.73 at d = 4 versus 0.82 at d = 12);
an SSIM claim at d = 4 is limited by d, not by the optimizer.

## 5. Answers, recommended protocol, and what each change does

### 5.1 Can Kapur's entropy, PSNR and SSIM be substantially improved further for the Enhanced SMA?

| Metric | Further improvement possible? | Lever | Measured (100 Mendeley OPGs) |
|---|---|---|---|
| **Kapur's entropy** | **Not as a raw value at d = 4** — ESMA v2 is within 0.0007 of the exact global maximum (0.0002 on DENTEX), so there is nothing left to gain. **Yes as an advantage over Standard SMA**, by measuring on a problem hard enough to separate the algorithms. | d ≥ 8; report gap-to-optimum and hit rate, not raw entropy | ESMA v2 − Standard: +0.0018 (d = 4) → +0.055 (d = 8) → +0.114 (d = 10) → +0.167 (d = 12), all p < 1e-12, 30–100× larger; at d = 4 the defensible statement is 84–89% vs 1–4% of images at the exact optimum |
| **PSNR** | **Yes, substantially** — but mostly through the evaluation protocol and the objective, not the optimizer | (1) class-mean repaint, (2) Otsu or hybrid objective, (3) d ≥ 8 | (1) **+4.8 dB** for every algorithm, 100/100 images; (2) **+0.36 dB** (hybrid) / **+0.47 dB** (Otsu) at a Kapur cost of 0.16% / 0.54%; (3) ESMA v2 − Standard becomes significant: +0.10 dB (d = 8, 63–65%), **+0.32–0.36 dB (d = 10, 75–84%, p ≤ 1e-9)**; hybrid ESMA v2 vs the Kapur baseline: **+0.30 dB (d = 8), +0.45 dB (d = 10)** |
| **SSIM** | **Yes, moderately** — SSIM is capped by the number of bands (5 bands ≈ 0.73, 13 bands ≈ 0.82); within a given d the optimizer can move it by thousandths | (1) class-mean repaint, (2) more thresholds, (3) hybrid or SSIM-aware objective | (1) **+0.113** for every algorithm, 100/100; (2) 0.726 → 0.823 from d = 4 to 12; (3) hybrid +0.0028, SSIM-aware +0.006–0.009 over pure Kapur (19–20/20 images); ESMA v2 − Standard: **+0.0021 at d = 10 (68%, p = 5e-4)**; hybrid ESMA v2 vs Kapur baseline **+0.0037 at d = 10 (63%, p = 9e-5)** |

Bottom line: the "small improvement" is a property of the d = 4 Kapur protocol, not of the Enhanced SMA. Under a
protocol that (a) paints bands with their means, (b) uses 8–10 thresholds and (c) optionally targets a
Kapur/Otsu hybrid, ESMA v2 beats Standard SMA on all three metrics at once with 63–92% of images won, while staying
2–2.5× faster.

### 5.2 Recommended Chapter 4 protocol

1. **Class-mean repaint** for PSNR/SSIM (`--recon mean`; API `levels=mean`). Report the fixed-level numbers once, for
   continuity with the existing tables, and state the convention (Arora et al. 2008).
2. **Threshold counts d = 4, 8, 10** (or 4, 6, 8, 10, 12). The d = 4 row carries the exactness claim (hit rate,
   worst-case gap); the d ≥ 8 rows carry the metric gains. Every SMA thresholding paper in the reference folder does
   this.
3. **Objective.** Keep Kapur's entropy as the thesis objective and make *gap to the exact optimum* and *hit rate* the
   primary evidence (already produced by `compare_three_way.py`). Add the **hybrid objective in Kapur units** (w = 0.5)
   as the segmentation-quality-oriented variant and compare it to the Kapur baseline on PSNR/SSIM (Section 4.5). Do
   not use the normalized [0, 1] hybrid for Standard-vs-Enhanced comparisons (Section 4.3).
4. **≥ 3 seeds per image**, paired Wilcoxon signed-rank tests, ties reported; the per-image ranking from unseeded
   frontend runs should not be cited for margins (its own caveat already says so).
5. **Re-measure on DENTEX** with the commands in Section 3 (`--subset all --max_images 300 --seed 42` reproduces the
   original selection). The Mendeley/DENTEX structure matched in the earlier audit, but the numbers must be measured,
   not assumed.
6. **Chapter 3 addition.** Name Standard SMA's scale-dependent exploitation gate `p = tanh|S(i) − DF|` as a fourth
   structural weakness and ESMA's gate-free, rank-weighted multi-leader move as the remedy (Section 4.3 is the
   evidence; Jiang et al. 2023 is the precedent).

### 5.3 Why each change helps

* **Class-mean repaint (S1).** For fixed thresholds the band value that minimizes the squared error is the band mean,
  so no other repaint can have a higher PSNR; SSIM's luminance and contrast terms improve for the same reason. It does
  not touch thresholds, fitness or class membership, so the Standard-vs-Enhanced comparison is unchanged in structure
  while both absolute metrics rise by 5 dB / 0.11. It also removes the ±0.5 dB noise that fixed levels inject
  (Section 4.1).
* **Otsu / hybrid objective (S2).** Otsu's criterion is *identical* to maximizing the PSNR of the class-mean image
  (Section 2, verified to 0.002 dB in 4.1), so any optimizer that targets it will show PSNR gains that Kapur-only
  optimization cannot. The hybrid keeps 99.8% of Kapur's entropy while taking 80% of Otsu's PSNR gain. Because both
  criteria are sums over classes, the table evaluation, the exact DP reference and the gap/hit-rate reporting carry
  over unchanged, and ESMA v2's feedback signals work as before. Expressing it in Kapur units keeps the baseline's gate
  behaving as it does in the thesis.
* **More thresholds (S3).** The search space grows as 255^d/d! and Standard SMA's per-coordinate polishing error
  compounds (its gap grows 70× from d = 4 to 12); ESMA v2's sorted representation and sampled multi-leader guidance
  degrade much more slowly (gap < half of Standard SMA's at every d). The metric gains follow directly, and the
  absolute PSNR/SSIM (29–33 dB, 0.77–0.82) are the values the literature reports as "good segmentation".
* **SSIM-aware objective (S5).** SSIM depends on local structure that no histogram criterion can see; only a spatial
  term in the objective can target it. The objective is non-separable, so the exact DP no longer applies and a
  metaheuristic is genuinely needed — and ESMA v2 finds a better optimum than Standard SMA on 19–20 of 20 images at
  40% of the wall time. Its gains are bounded by d (Section 4.6); it is the right tool once d ≥ 8 makes SSIM
  headroom available.

### 5.4 What would not help (measured or implied by the ceiling)

* More iterations or agents at d = 4: the equal-budget ESMA v2 run moves mean Kapur by 0.00003 and PSNR by −0.002 dB.
* Additional SMA operators at d = 4 (Lévy flights, opposition-based learning, chaotic initialization, mutation): they
  cannot raise a value that is already at the global maximum on 84–92% of images. They belong in the d ≥ 8 regime, where
  ESMA v2's own hit rate falls to ≤ 6% and the z(t) exploration channel / opt-in polish should be re-ablated.
* Tuning ESMA's control parameters for PSNR/SSIM under the Kapur objective: the correlation between Kapur gain and
  PSNR/SSIM gain is ≈ 0 (Section 4.2).
* Citing the normalized-hybrid margins (+0.10 dB at d = 4) as ESMA merit: they are Standard SMA's scale artifact.

### 5.5 Limitations of this study

* Data: the 100 Mendeley OPGs on this machine, not the 300 DENTEX images; the DENTEX d = 4 numbers in Section 1 show
  the same structure (ceiling, PSNR/SSIM at the optimum ≈ Standard SMA), but Sections 4.1–4.6 must be re-run there.
* Seeds: PSNR/SSIM from one seed per image (42); optimality gaps from three. Per-image PSNR/SSIM signs at d = 4 remain
  noisy; the paired tests over 100 images are what carry the conclusions.
* The SSIM-aware objective was run on 20 hold-out images with SSIM evaluated on an 8× down-sampled proxy; an in-loop
  SSIM evaluation costs ≈ 1 000× a table lookup (13 s vs 18 ms per run).
* The hybrid's normalization constants are the two exact optima of the image (values only, never the thresholds).
* At d ≥ 10 ESMA v2's patience-20 termination fires at ≈ 85 of 100 iterations and its hit rate is ≤ 1%; equal-budget,
  larger-T and exploration-channel ablations at high d were not part of this study.

## 6. What the literature says (external sources and the reference folder)

### 6.1 Which objective functions track PSNR and SSIM

| Source | Finding relevant here |
|---|---|
| *Image Thresholding: Understanding Bias of Evaluation Metrics towards Specific Evaluation Functions*, arXiv 2605.27132 (BSDS500, 500 images) | Correlation with PSNR: Otsu 0.987 ± 0.008 vs Kapur 0.682 ± 0.189 (Otsu higher on 100% of images). Correlation with SSIM: Otsu 0.896 ± 0.131 vs Kapur 0.674 ± 0.172 (Otsu higher on 91% of images). The authors call this an "inherent metric–objective-function bias" that exists regardless of the optimizer. |
| Kalyani, Sathya & Sakthivel 2021, IOP Conf. Ser. Mater. Sci. Eng. 1119 (cuckoo search, 4–7 thresholds) | "The Otsu based cuckoo search algorithm outperform[s] Kapur and MCE" in PSNR, SSIM and CPU time with the *same* optimizer. |
| Jiang et al. 2023, *Entropy* 25(1) (improved SMA + symmetric cross-entropy, d = 2–5) | "Kapur's entropy … at the low threshold (d = 2, 3) are the lowest" on PSNR/SSIM/FSIM; at d = 4–5 symmetric cross-entropy wins (Lena, d = 5: SSIM 0.680 vs Otsu 0.581 vs MCE 0.612). Chosen because it "takes into account both gray-level and neighborhood average gray-level information". |
| Hosny et al. 2022, *Sci. Rep.* / PMC9510310 (hybrid fitness a·Otsu + b·Kapur, a = b = 0.5; K = 6…26) | Hybrid beats both single objectives: PSNR 23.49 vs 22.61 (Kapur) vs 23.21 (Otsu) at K = 6; 33.79 vs 30.77 vs 33.47 at K = 26; SSIM 0.6864 vs 0.6831 vs 0.6822 at K = 6. Equal weights "yield better results". |
| Khairuzzaman & Chaudhury 2019, *Multimedia Tools Appl.* (Masi entropy) | Masi-entropy thresholding obtains superior ME, MSE and PSNR versus Kapur, Rényi and Tsallis "for almost all images at each threshold level". |
| Boubechal, Seghir & Benzid 2019, *Applied Artificial Intelligence* 33 | Generalize SSIM itself into the objective function for multilevel thresholding (swarm-optimized, parallelized); reported better thresholds than Otsu- and PSNR-based objectives. |
| Brajević & Ignjatović 2025, PMC12214501 | Use Otsu as objective and cite prior work "combining Kapur's entropy with structural similarity index" — hybrid Kapur/SSIM objectives exist in the literature. |

Reading: PSNR is *by construction* the Otsu criterion of the class-mean segmented image (Section 2, P2); SSIM adds
spatial structure that no histogram criterion encodes. Any thesis that reports PSNR/SSIM as quality evidence while
optimizing pure Kapur's entropy is measuring an objective it is not optimizing.

### 6.2 Number of thresholds

| Source | Finding |
|---|---|
| Lévy-flight + quasi-opposition ESMA, *Entropy* 2021, 23, 1700 (reference folder; MCE objective, nTh = 4, 6, 8, 10; 30 runs, N = 30, T = 500) | "Only small differences between the ESMA and other compared algorithms in threshold values 4 and 6. However, the PSNR values significantly increase when the threshold values are increasing." "When the threshold is equal to 4, the SSIM results of each algorithm are roughly the same." Baboon SSIM 0.8041 (4 thresholds) → 0.9395 (10). |
| SMA with leadership and self-phagocytosis (SMA-MLS), *Expert Syst. Appl.* 2024 (reference folder; Kapur; TH = 2, 6, 10, 14, 18) | "The larger the number of thresholds, the more obvious the effect is"; Friedman χ² over algorithms grows 18.37 → 24.38 → 27.88 → 28.62 for TH = 2, 6, 10, 14; "as the number of thresholds increases, the algorithm segments images more accurately, and the difference with other algorithms increases." |
| Diffusion-association SMA (DASMA) + Rényi entropy, *Comput. Biol. Med.* (reference folder; K = 5–8, COPD CT) | "As the threshold level increases, the accuracy of segmentation also increases" for PSNR, SSIM and FSIM; differences between algorithms are established by Friedman ranks at K = 6–8, not below. |
| Hosny et al. 2022 | Evaluate at K = 6, 10, 14, 18, 22, 26; PSNR rises from ≈ 23 dB to ≈ 34 dB. |

Reading: the community demonstrates optimizer improvements at 8–18 thresholds because that is where the search
problem is hard enough for optimizers to differ. The thesis protocol (d = 4) is below the range in which SMA variants
are ever reported to separate on PSNR/SSIM.

### 6.3 Exact methods for separable criteria

| Source | Finding |
|---|---|
| Luessi, Eichmann, Schuster & Katsaggelos, ICIP 2006; *J. Electron. Imaging* 18(1) 2009 | Dynamic programming finds the exact optimum of Otsu, Kapur and Kittler–Illingworth criteria in O((K−1)L²); the framework covers any criterion that is a sum over classes. |
| Hegazy & Gabr 2026, arXiv 2605.27287 | DP framework for Otsu/Kapur/Kittler (O(L³) time, O(L²) space) plus a criterion that chooses the number of thresholds; argues exact methods are preferable to stochastic metaheuristics whose "results are inconsistent across runs". |
| *Efficient solution of Otsu multilevel image thresholding: a comparative study*, *Expert Syst. Appl.* 2018; adaptive Masi-entropy DP, *J. Vis. Commun. Image Represent.* 2023 | Same conclusion for Otsu and Masi entropies: exact solutions are cheap for small K. |

Reading: for Kapur, Otsu and any hybrid of them the ceiling is known exactly; a metaheuristic's contribution must be
stated as gap and hit rate against that ceiling (as `compare_three_way.py` already does). A metaheuristic is *needed*
only when the objective is non-separable — SSIM-aware terms, 2D/neighbourhood histograms, spatial regularizers — which
is also where PSNR/SSIM gains become available.

### 6.4 Spatial information (2D histograms) for SSIM/FSIM

| Source | Finding |
|---|---|
| Mittal & Saraswat 2018, *Eng. Appl. Artif. Intell.* 71 | Non-local-means 2D histogram + exponential-Kbest gravitational search: improvements in SSIM, FSIM and RMSE over 1D-histogram thresholding. |
| DASMA (reference folder) | Applies Rényi entropy to a non-local-means 2D histogram for exactly this reason. |
| Morphologically reconstructed 2D histogram with opposition learning, *Cluster Computing* 2024 | Same direction: spatial context in the histogram raises structural metrics. |
| Abualigah, Almotairi & Abd Elaziz 2022/2023, *Applied Intelligence* (survey) | Lists the objective–metric mismatch, limited use of 2D histograms, absence of hybrid multi-objective formulations and lack of comparison with exact methods among the open challenges of metaheuristic multilevel thresholding. |

### 6.5 The repaint convention

| Source | Finding |
|---|---|
| Arora, Acharya, Verma & Panigrahi 2008, *Pattern Recognit. Lett.* 29(2) | For PSNR, "each pixel is assigned the mean gray value of its class." |
| Hosny et al. 2022; Brajević & Ignjatović 2025 | Segmented image built by region membership of the threshold vector, scored against the original with PSNR/SSIM per the standard formulas. |

Reading: the fixed-level repaint in the original `apply_thresholds` is a visualization choice, not the literature's
evaluation convention, and it costs every algorithm several dB.

### 6.6 SMA-specific enhancements in the reference folder and what they can and cannot do here

The folder documents the mechanisms other authors bolted onto SMA: Lévy flights + quasi-opposition learning (ESMA,
*Entropy* 2021), diffusion mechanism + association strategy (DASMA), leadership update + adaptive combined mutation +
self-phagocytosis + Logistic–Tent chaotic initialization (SMA-MLS), elite opposition-based learning + historical
leader (ISMA, Jiang 2023), differential evolution + Powell local search (PSMADE), adaptive grouping, elitist archives,
Bloch-sphere elite initialization (QCMSMA), local dimensional mutation (feature selection), and several adaptive
parameter-control schemes (JADE-style, closed-loop data-driven control). All of them report PSNR/SSIM gains over the
base SMA *at high threshold counts* and most of them with a non-Kapur objective (MCE, Rényi, symmetric cross-entropy).
None of them can raise Kapur's entropy above 19.2600 at d = 4, and none of them changes PSNR/SSIM when the objective
does not target those metrics — which is exactly the situation the current benchmark is in. They become relevant again
once the objective is non-separable (Section 5, S4) or d is large (S3), where the search is genuinely hard.

### Sources

* arXiv 2605.27132 — https://arxiv.org/abs/2605.27132
* Hosny et al. 2022 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9510310/
* Jiang et al. 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9858507/
* Lévy/QOBL ESMA 2021 — https://pmc.ncbi.nlm.nih.gov/articles/PMC8700578/ (also in `Datasets/SMA MD File for AI/`)
* Boubechal, Seghir & Benzid 2019 — https://www.tandfonline.com/doi/full/10.1080/08839514.2019.1683986
* Kalyani, Sathya & Sakthivel 2021 — https://iopscience.iop.org/article/10.1088/1757-899X/1119/1/012019
* Khairuzzaman & Chaudhury 2019 (Masi entropy) — https://link.springer.com/article/10.1007/s11042-019-08117-8
* Mittal & Saraswat 2018 — https://www.sciencedirect.com/science/article/abs/pii/S0952197618300496
* Hegazy & Gabr 2026 — https://arxiv.org/html/2605.27287
* Efficient solution of Otsu multilevel thresholding (ESWA 2018) — https://www.sciencedirect.com/science/article/abs/pii/S0957417418305797
* Adaptive Masi entropy by DP (2023) — https://www.sciencedirect.com/science/article/abs/pii/S1047320323002584
* Arora et al. 2008 — https://people.ece.cornell.edu/acharya/papers/mlt_thr_img.pdf
* Abualigah et al. survey — https://link.springer.com/article/10.1007/s10489-022-04064-4
* Brajević & Ignjatović 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12214501/
* Morphologically reconstructed 2D histogram (2024) — https://dl.acm.org/doi/abs/10.1007/s10586-024-05027-9
* Comprehensive survey of multilevel thresholding (2024) — https://link.springer.com/article/10.1007/s11831-024-10093-8
* SMA-MLS, DASMA, PSMADE, HKTSMA, QCMSMA, ISMA papers — `Datasets/SMA MD File for AI/`
