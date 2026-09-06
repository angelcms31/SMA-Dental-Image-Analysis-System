# Enhanced Slime Mould Algorithm v2 — audit, diagnosis, improvements and benchmark report

Branch `ImprovementTest`, September 2026. Code: `backend/sma_algorithms.py` (Standard SMA, ESMA v1, ESMA v2,
shared fitness infrastructure), `backend/compare_three_way.py` (three-way benchmark), `backend/ablate_esma_v2.py`
(component ablation), `backend/tests/` (pytest suite, 11 tests). All numbers in this report were produced by
those scripts; nothing was transcribed from memory.

**Data note.** The 300-image DENTEX x-ray folder used for the original benchmark screenshot is not present on
this machine (the cached bytecode in `backend/__pycache__` is CPython 3.14, i.e. that run happened elsewhere).
Everything here was measured on the 100 Mendeley panoramic OPG images in `Datasets/images` (2943 × 1435,
8-bit grayscale), split deterministically into a **40-image development set** (used only to choose ESMA v2's
defaults) and a **60-image hold-out set** (reported results). `compare_three_way.py --subset all --max_images 300
--seed 42` reproduces the exact 300-image DENTEX selection on any machine that has the folder.

---

## 1. Current ESMA objectives and implemented enhancements

The original implementation (`enhanced_sma`, kept unchanged and referred to as **ESMA v1**) follows the thesis'
Algorithm 3.1 literally. What it adds on top of `standard_sma`:

| # | Enhancement in code | Thesis basis | Effect vs Standard SMA |
|---|---|---|---|
| E1 | Fitness-weighted multi-leader guidance `X_i(t+1) = Σ_j w_j (L_j + vb·(W·XA − XB))`, `w_j = S(L_j)/Σ S`, top-k leaders (k = 3) | Objective 1, §3.2.1 | Replaces the single leader `Xb`; applied to **every** agent, **unconditionally** (no `rand < z` branch, no `vc·X` branch) |
| E2 | Quasi-uniform initialization `X[i,j] = lb + (i/N + rand/N)(ub − lb)` | Objective 2 (iSMA, Conteh & Du 2025) | Replaces random uniform initialization |
| E3 | Performance-feedback control: `CR(t)` over a window h = 5, `PD(t)` = mean pairwise distance / D, update rules for `z(t)` and `a(t)` (α = β = γ = δ = 0.10, z ∈ [0.01, 0.5], τ_CR = 1e-6, τ_PD = 0.10, a₀ = 1) | Objective 3, §3.2.1 | Replaces fixed z = 0.03 and `a = arctanh(1 − t/T)` |
| E4 | `adaptive_k` (leader count decays with PD) | documented extension, **off** in the benchmark | Not active in reported results |
| E5 | Greedy per-agent selection (keep the better of old/new position) | "standard convention" | Added to **both** SMA and ESMA (not in Li et al. 2020) |
| E6 | Vectorized W, position update (`tensordot`), clash-free XA/XB sampling | implementation | Same mathematics, no Python agent loop |
| E7 | `autocrop_black_borders` (black/white letterbox removal) before the histogram | preprocessing | Identical for all algorithms |

Benchmark settings used throughout (identical to `compare_full_images.py`): N = 30 agents, T = 50 iterations,
d = 4 thresholds, seed 42 for every algorithm on every image.

### What each enhancement does, and where it stands

* **E1** is the source of ESMA v1's entropy advantage: anchoring every agent at the top-3 centroid with a
  perturbation amplitude that stays ≈ 1 all run long keeps sampling the neighbourhood of good basins until the
  last iteration, whereas Standard SMA's amplitude decays to zero. It is, however, applied to unsorted agents
  (see D5) and with weights that are effectively uniform (D6).
* **E2** is implemented as written in the thesis, and as written it does not achieve its stated purpose (D4).
* **E3** is only half alive: `z(t)` is computed but never consumed (D1) and `a(t)` cannot contract (D2) or stop
  growing (D3). The measurable effect of Objective 3 in v1 is therefore limited to occasional amplitude growth.
* **E5** (greedy selection) is the single most important reason both implementations are strong: it turns the
  population into an elitist archive so no good threshold vector is ever lost.
* **E6/E7** are correct and were kept.

## 2. Analysis of current strengths and weaknesses

### 2.1 Baseline numbers

DENTEX, 300 images (user's screenshot, original code):

| Metric | Standard SMA | ESMA v1 | Gain | ESMA better | Wilcoxon p |
|---|---|---|---|---|---|
| Kapur's entropy | 19.2105 | 19.2155 | +0.0050 | 275/300 (91.7%) | < 0.0001 (significant) |
| PSNR (dB) | 19.9492 | 19.9302 | −0.0190 | 165/300 (55.0%) | 0.5426 (not significant) |
| SSIM | 0.4986 | 0.5000 | +0.0014 | 191/300 (63.7%) | 0.0013 (significant) |
| Runtime (s) | 0.2161 | 0.2186 | +0.0025 | 102/300 (34.0%) | < 0.0001 (significant, slower) |

Mendeley, 100 images (this machine, **untouched original code**, `compare_full_images.py`):

| Metric | Standard SMA | ESMA v1 | Gain | ESMA better | Wilcoxon p |
|---|---|---|---|---|---|
| Kapur's entropy | 19.0697 | 19.0722 | +0.0025 | 87/100 | < 0.0001 (significant) |
| PSNR (dB) | 19.9341 | 19.9063 | −0.0277 | 47/100 | 0.5451 (not significant) |
| SSIM | 0.6120 | 0.6134 | +0.0014 | 53/100 | 0.5917 (not significant) |
| Runtime (s) | 0.2078 | 0.2015 | −0.0063 | 56/100 | 0.5988 (not significant) |

The pattern is the same on both datasets: a tiny but very consistent entropy edge, PSNR slightly worse with
higher variance, SSIM marginal, runtime equal within noise.

### 2.2 The decisive diagnostic: distance to the exact optimum

Kapur's objective is a sum of independent per-class entropies, so the **exact global maximum** for d thresholds
can be computed by dynamic programming in a few milliseconds (`kapur_optimal_thresholds`; see §5.1). Measuring
every run against it (100 Mendeley images, original code, seed 42) explains the baseline numbers:

| Algorithm | Reached the exact optimum | Mean gap | Median gap | Max gap | Median L∞ threshold error | Max L∞ error |
|---|---|---|---|---|---|---|
| Standard SMA | **0 / 100** | 0.0053 | 0.0042 | 0.0220 | 3 levels | 12 levels |
| ESMA v1 | **70 / 100** | 0.0028 | 0.0000 | 0.0481 | 0 levels | 14 levels |
| Exact optimum (mean Kapur 19.0750) | 100 / 100 | 0 | 0 | 0 | 0 | 0 |

* **Why ESMA v1 wins on entropy:** Standard SMA never lands exactly on the optimum — it always ends 1–12 gray
  levels away (a *polishing* failure, not a basin failure: no image has an L∞ error ≥ 20). ESMA v1's persistent
  moderate perturbation around the top-3 centroid polishes better and reaches the exact optimum in 70% of images.
* **Why the gain is tiny:** the *entire* headroom above Standard SMA on this set is +0.0053 (mean Kapur 19.0750
  vs 19.0697); v1 captures about half of it, and its failures are worse than SMA's (max gap 0.048 vs 0.022),
  which raises its variance.
* **Why SSIM barely moves:** both segmentations replace the image by five flat gray bands (SSIM ≈ 0.5–0.6);
  threshold shifts of a few levels move SSIM in the third decimal, in the same direction as entropy.
* **Why PSNR is slightly worse:** PSNR is not the optimized objective. `apply_thresholds` paints the bands with
  fixed evenly spaced levels (0, 64, 128, 191, 255), so PSNR depends on where the *band edges* fall relative to
  the image's intensity mass, which Kapur's criterion does not target. Section 7 measures PSNR *at the exact
  optimum* to separate "the optimizer" from "the objective".
* **Why ESMA v1 is slower:** not the fitness function (identical for both) but ≈ 50 µs of extra numpy work per
  iteration (pairwise-distance PD over all agents, a `(k, N, d)` tensordot, history bookkeeping).

## 3. Problems identified in the current implementation

Verified by reading and instrumenting `enhanced_sma` (v1):

* **D1 — `z(t)` is dead code.** It is updated every iteration (step 5.6) but never used: the position update has
  no `rand < z` branch. Half of Objective 3 has no effect on the search.
* **D2 — `a(t)` cannot contract.** `CR(t)` is a *relative* entropy change (≈ 1e-5 … 1e-3 for bF ≈ 19), so
  `a -= δ·CR` and `z -= β·CR` move the parameters by ≈ 1e-5 per iteration. `a` stays at 1.0 during convergence;
  v1 has no fine-exploitation phase (Standard SMA gets one from `arctanh(1 − t/T) → 0`).
* **D3 — `a(t)` grows without bound during stagnation** (`a += γ(1 − CR)` ≈ +0.1 per iteration, no cap). Under
  greedy selection the population stays collapsed, stagnation persists and `a` keeps growing, so late proposals
  degenerate into clipped random jumps.
* **D4 — The quasi-uniform initialization collapses onto the diagonal.** The same stratum index `i` is used in
  every dimension, so agent i has all four thresholds inside one 8.5-level band (t1 ≈ t2 ≈ t3 ≈ t4). Coverage of
  the *threshold-configuration* space is worse than random initialization; three of the five classes start empty.
* **D5 — The d!-fold permutation symmetry is ignored.** Fitness depends only on `sorted(thresholds)`, but agents
  are not kept sorted, so coordinate j means different things in different agents. The leader centroid
  `Σ w_j L_j` and the difference vector `W·XA − XB` mix unrelated thresholds (the centroid of [40, 90, 140, 200]
  and [200, 40, 140, 90] is [120, 65, 140, 145], a point in no basin).
* **D6 — The fitness weights are degenerate.** All Kapur values are ≈ 19.2 ± 0.1, so `w_j = S(L_j)/Σ S` ≈ 1/k for
  every leader: "fitness-weighted" only in name.
* **D7 — Fitness evaluation dominates runtime for every algorithm.** `kapurs_entropy_fitness` is called N times
  per iteration in a Python list comprehension (≈ 140–190 µs per call, ≈ 5.8 ms per generation of 30 agents;
  ≈ 0.2 s per run of 1 530 evaluations).
* **D8 — Benchmark script bug.** `compare_full_images.py` ended with a leftover second
  `args = parser.parse_args(); run_comparison(...)`: every reported run executed the whole benchmark **twice**,
  the second time ignoring all `--esma_*` overrides and overwriting the JSON. Fixed (single call).
* **Thesis-text inconsistency to flag to the adviser:** §3.2.1 / Algorithm 3.1 (unconditional multi-leader move,
  no z branch, k = 3 in code) versus §3.4.1–3.4.2 (X_guide replaces `Xb` only in the exploitation branch, `vc` and
  z-reinitialization branches kept, k = 5, look-back δ = 10, z ∈ [0.01, 0.30],
  `z(t) = z_max − (z_max − z_min)·CR/CR_max·PD/PD_max`, PD as distance to the centroid). The code follows §3.2.1.

## 4. New improvement objectives

| ID | Objective | Fixes | Measurable target |
|---|---|---|---|
| N1 | Exact class-entropy table + population-level vectorized evaluation, shared by all algorithms | D7 | identical fitness values (< 1e-9); ≥ 5× lower runtime for every algorithm |
| N2 | Canonical ordered representation (agents kept sorted) | D5 | lower mean gap to the exact optimum, higher share of exact hits |
| N3 | De-correlated (Latin-hypercube) quasi-uniform initialization | D4 | same stratification guarantee, no diagonal collapse, lower gap |
| N4 | Scale-free multi-leader weights and stochastic (sampled) multi-leader guidance | D6, D5 | lower gap, higher hit rate than the weighted centroid |
| N5 | Bounded, normalized feedback control: `CR/CR_max`, `PD/PD_0`, multiplicative contraction with floor, additive expansion with cap, `z(t)` connected to a real exploration branch | D1, D2, D3 | a(t) measurably tracks progress; contribution quantified by ablation |
| N6 | Convergence-based adaptive termination (patience) | runtime | fewer iterations at unchanged mean Kapur; quality cost measured against the equal-budget run |
| N7 | Opt-in integer-neighbourhood polish of the final best (`local_refine`, default **off**) | — | reported separately; an operator beyond SMA's components, left to the adviser |
| N8 | Optimality-gap evaluation with the exact DP optimum, per-image JSON, all-pairs Wilcoxon | evaluation | rigorous, reproducible Chapter 4 evidence |

Every default of ESMA v2 was chosen on the **development split only**, with 3–8 seeds per configuration, and the
hold-out set was touched exactly once, for the final benchmark.

## 5. Detailed algorithmic improvements (ESMA v2 = `enhanced_sma_v2`)

### 5.1 Exact evaluation infrastructure (N1, N8 — implementation-level, objective unchanged)

For a class covering bins [lo, hi) with P = Σ p_i and S = Σ p_i ln p_i,
`H[lo, hi) = −Σ (p_i/P) ln(p_i/P) = ln P − S/P`. `KapurEntropyTable` tabulates all 257 × 257 class entropies once
per image (≈ 2.5 ms, 0.5 MB; partial sums start fresh at `lo` so small classes lose no precision), after which a
whole population is evaluated with one fancy-indexing call (≈ 20 µs for 30 agents versus ≈ 5.8 ms for 30 scalar
calls). Agreement with `kapurs_entropy_fitness`: max |Δ| = 1.4e-14 over 18 400 random, duplicate, out-of-range and
half-integer threshold vectors; Standard SMA and ESMA v1 produce **identical thresholds on all 100 reference
images** with the fast path (test `test_legacy_algorithms_unchanged_against_frozen_reference`).

`kapur_optimal_thresholds` runs the dynamic programme `g_j[t] = max_{s ≤ t} g_{j−1}[s] + H[s, t]` (duplicates
allowed, exactly the space the SMA variants search) in O(d·L²) ≈ 7 ms including the table. It is used **only** as
an evaluation reference, never inside any optimizer (the way known optima of benchmark functions are used in the
metaheuristics literature). The report is explicit that for fixed small d this also makes the exact method the
practical choice in production; the research object here is the metaheuristic's behaviour.

### 5.2 Canonical ordered representation (N2)

After initialization and after every position update, each agent's vector is sorted ascending. Fitness is
invariant (`f(x) = f(sort(x))`), so the objective is untouched; what changes is that coordinate j now means "the
j-th smallest threshold" in every agent, so leaders, centroids and difference vectors act on aligned coordinates
and the 24 redundant orderings of every solution collapse into one. Effect (development split, 200 random
configurations × 3 seeds): mean gap 0.0010 with canonical ordering vs 0.0075 without; the five worst
configurations of the whole search were all unsorted.

### 5.3 Latin-hypercube quasi-uniform initialization (N3)

`X[i, j] = lb + (π_j(i) + rand) / N · (ub − lb)`, with an independent random permutation π_j per dimension: every
dimension is still split into N equal strata each holding exactly one agent (the guarantee Objective 2 asks
for), but agents cover the hypercube instead of its diagonal. The thesis-literal formula is kept as
`init="strata"` for the ablation (gap 0.0009 vs 0.0001 for LHS, 80.8% vs 85.0% exact hits).

### 5.4 Sampled fitness-weighted multi-leader guidance (N4)

Because Σ w_j = 1, the thesis update `Σ_j w_j [L_j + vb (W·XA − XB)]` equals `X_guide + vb (W·XA − XB)` with
`X_guide = Σ_j w_j L_j` (v1's `(k, N, d)` tensordot computed exactly this at k times the cost). v2 realizes the
same guidance **stochastically**: each agent follows one leader L_j drawn with probability w_j, so
E[anchor] = X_guide while different agents are pulled toward different leaders — agents can exploit a second
basin instead of being averaged into the space between basins. Weights are rank-based (w_j ∝ k − j + 1,
scale-free); the raw-fitness weights of v1 (`weight_mode="fitness"`) and advantage weights remain available.
k = 5 (the value §3.4.1 of the thesis uses). Effect: sampled leaders 0.0011 vs centroid 0.0071 (marginal means
over 200 configurations); in the ablation the centroid drops exact hits from 85.0% to 65.0%.

### 5.5 Bounded, normalized performance-feedback control (N5)

Signals (as in the thesis, made dimensionless):
`CR(t) = (bF(t) − bF(t − h)) / (|bF(t)| + ε)`, `CRn = CR / max_{τ ≤ t} CR(τ)` (§3.4.1's CR/CR_max),
`PD(t)` = mean pairwise Euclidean distance of the fitter half of the population divided by the search-space
diameter, `PDn = PD / PD(0)`. Update (h = 5, α = β = γ = δ = 0.10):

* stagnating (`CR ≤ τ_CR` and `PDn < τ_PD = 0.10`): `z ← min(z + α(1 − PDn)(1 − CRn), z_max)`,
  `a ← min(a + γ(1 − CRn), a_max = 1.0)`
* converging (`CR > 0`): `z ← max(z − β·CRn, z_min)`, `a ← max(a·(1 − δ·CRn), a_min = 0.05)`

`a` therefore contracts multiplicatively while progress is being made and re-expands under stagnation, within
[0.05, 1.0]; `z(t)` follows the same logic within [0.01, 0.10]. A mild contraction (δ = 0.1) is best: δ = 0.5
was the worst setting in the random search (0.0093 vs 0.0016), because the population still needs
amplitude ≈ 0.5–1 to keep finding improvements — the fine polishing actually comes from the aligned difference
vectors `XA − XB` of a converged, sorted population (whose magnitude shrinks with the population spread), not
from a small `a`. Removing the amplitude adaptation (a fixed at 1) raises the mean gap 6× (0.0006 vs 0.0001).

**The z(t) exploration branch.** v2 connects `z(t)` to a real branch: with probability z each non-leader agent
makes an exploration move — either the Standard-SMA random re-initialization (`explore_mode="reinit"`, accepted
unconditionally so that diversity actually survives greedy selection) or an SMA move at the full amplitude
`a_max` (`explore_mode="wide"`, greedy-selected). Every variant tried (z₀ ∈ {0, 0.01, 0.03, 0.1}, z_max ∈ {0.05,
0.1, 0.3}, both modes, 8 seeds × 40 images) was neutral-to-harmful at T = 50 (exact hits 81–88% vs 88.1% without;
mean gap 0.0004–0.0009 vs 0.00015) and neutral at T = 100. The reason is visible in the DP diagnostic: on this
problem essentially all remaining error is polishing error inside the right basin, so evaluations spent on
restarts are evaluations not spent on polishing. The branch is therefore **off by default** (`explore_channel=False`),
fully implemented, switchable, and `z(t)` is still computed and returned in `history`. This is a finding for
Chapter 4, not a removal of Objective 3.

### 5.6 Adaptive termination (N6)

The run stops when the best fitness has not improved for `patience = 20` iterations (by then the feedback loop
has already re-expanded `a`). On the development split this keeps the mean gap of the full-T run (0.00015) at
39.8 instead of 50 iterations (exact hits 85.9% vs 88.1%); at T = 100 it stops after ≈ 40 iterations
(86.2% vs 90.6%). The equal-budget run is always reported next to it.

### 5.7 Opt-in local polish (N7)

`local_refine=True` adds a coordinate-wise ±1 hill climb on the integer grid around the final best (2d table
lookups per round). It is an operator outside SMA's own components, so it is off by default. With v2 it is
nearly redundant: the remaining v2 failures are basin-level (max gap 0.0316 both with and without polish) and
the hit rate moves from 85.9% to 86.2% on the development split.

### 5.8 How the defaults were chosen (development split only)

1. 203 random configurations × 40 images × 3 seeds (structure search): canonical ordering and sampled leaders are
   the two large effects; δ = 0.1 and a₀ = 1 beat larger values; `z_max` = 0.1 beats 0.3.
2. 384-configuration full factorial over the remaining options × 3 fresh seeds: `a_min`, `γ`, `pd_low_thresh`,
   `weight_mode` and `init` (LHS vs uniform) change results by less than seed noise; rank weights, "prop"
   contraction, k = 5 and h = 5–10 are marginally better.
3. 8 finalists × 8 seeds, plus a patience sweep and an exploration-channel study: the default configuration
   above (mean gap 0.00015, 88.1% exact hits at full T; 0.00015 / 85.9% / 39.8 iterations with patience 20)
   against 0.00364 / 62.8% for ESMA v1 and 0.00812 / 0.3% for Standard SMA (320 runs each).

Component ablation with the final defaults (development split, 40 images × 3 seeds, N = 30, T = 50, d = 4,
exact-optimum mean Kapur 19.1010; `esma_v2_ablation.json`):

| Configuration | Mean Kapur | Gap mean | Gap max | Exact optimum | Iterations |
|---|---|---|---|---|---|
| Standard SMA | 19.0939 | 0.0071 | 0.0718 | 0.8% | 50.0 |
| Original ESMA (v1) | 19.0971 | 0.0039 | 0.0672 | 61.7% | 50.0 |
| **ESMA v2 (defaults)** | **19.1010** | **0.0001** | **0.0035** | **85.0%** | **39.8** |
| − canonical ordering (unsorted agents) | 19.1004 | 0.0006 | 0.0321 | 83.3% | 42.5 |
| − LHS init → thesis-literal diagonal strata | 19.1002 | 0.0009 | 0.0672 | 80.8% | 42.6 |
| − LHS init → uniform random init | 19.1004 | 0.0006 | 0.0672 | 85.0% | 37.7 |
| − sampled leaders → weighted centroid | 19.0999 | 0.0011 | 0.0672 | 65.0% | 35.3 |
| − rank weights → raw fitness weights (v1) | 19.1003 | 0.0007 | 0.0672 | 85.0% | 41.6 |
| − a(t) adaptation (amplitude fixed at a₀) | 19.1004 | 0.0006 | 0.0672 | 85.0% | 43.8 |
| − early stopping (full T) | 19.1010 | 0.0001 | 0.0035 | 87.5% | 50.0 |
| + z(t) exploration channel (wide moves) | 19.1000 | 0.0010 | 0.0672 | 85.8% | 39.3 |
| + z(t) exploration channel (random re-init) | 19.1001 | 0.0009 | 0.0672 | 77.5% | 37.8 |
| + opt-in local polish | 19.1010 | 0.0001 | 0.0035 | 85.0% | 39.8 |
| k = 1 (single leader) | 19.1001 | 0.0010 | 0.0672 | 76.7% | 36.6 |
| k = 3 | 19.1001 | 0.0009 | 0.0672 | 78.3% | 38.4 |

## 6. Runtime optimization techniques

Measured on this machine (4 cores), 10 images, N = 30, T = 50, d = 4, minimum of 7 repetitions per run:

| Implementation | Per run | Per iteration | Note |
|---|---|---|---|
| Standard SMA, legacy path (`fast_fitness=False`, original code) | 170.8 ms | ≈ 3.4 ms | 30 scalar `kapurs_entropy_fitness` calls per generation |
| Standard SMA, table path (default) | 11.75 ms | 0.19 ms | ≈ 14.5× faster, identical thresholds on 100/100 images |
| ESMA v1, table path | 15.27 ms | 0.26 ms | extra: all-agent pairwise PD, `(k, N, d)` tensordot |
| **ESMA v2, defaults (adaptive termination)** | **11.42 ms** | 0.25 ms × ≈ 40 iterations | faster than Standard SMA per run |
| ESMA v2, full T | 13.16 ms | 0.25 ms | equal budget |

What was changed, and what it is:

1. **Exact class-entropy table + one vectorized evaluation per generation** (implementation-level; N1). The
   values are the same to 1e-14; this is where ≈ 93% of the legacy runtime went. Applies to all three algorithms
   so that the comparison isolates algorithmic overhead; `fast_fitness=False` keeps the legacy loop available.
2. **`compute_histogram_prob` via `np.bincount`** for 8-bit input (exact; ≈ 10× faster on 4-megapixel images;
   outside the timed region).
3. **Multi-leader update in closed form** `X_guide + vb (W·XA − XB)` instead of a `(k, N, d)` tensordot
   (mathematically identical because Σ w_j = 1).
4. **Diversity on the fitter half with numpy broadcasting** (14 µs) instead of scipy `pdist` (≈ 115 µs per call
   including wrapper overhead) or all-agent broadcasting; also removes a scipy import from the core module.
5. **Inverse-CDF leader sampling** (`searchsorted` on the cumulative weights) instead of `rng.choice(p=…)`
   (≈ 4× cheaper per iteration).
6. **Sorted evaluation** (`assume_sorted=True`): canonical agents skip the per-row sort inside the table lookup.
7. **Adaptive termination** (algorithmic; N6): ≈ 20% fewer iterations at unchanged mean Kapur — the only change
   that reduces search effort, and it is reported against the equal-budget run.
8. **Benchmark script fix** (D8): every run of `compare_full_images.py` used to execute twice.

Not done on purpose: no reduction of image resolution, population size or iteration count; `metrics.py` and
`apply_thresholds` untouched; Standard SMA's update rules untouched.

## 7. Before-and-after benchmark comparison

`python compare_three_way.py --images_dir ../Datasets/images --subset holdout --sma_population 30
--sma_iterations 50 --d 4 --seed 42` — 60 hold-out Mendeley OPG images never used while designing v2; every
algorithm sees the same image, the same histogram, the same seed and the same evaluation infrastructure.
PSNR/SSIM are computed exactly as before (`metrics.py`, scikit-image, on the `apply_thresholds` segmentation).

| Algorithm | Kapur's entropy | PSNR (dB) | SSIM | Runtime (ms) | Iterations | Gap to exact optimum (mean / max) | Exact optimum reached |
|---|---|---|---|---|---|---|---|
| Standard SMA | 19.0534 ± 0.3500 | 19.8818 ± 0.9403 | 0.6109 ± 0.0437 | 13.7 ± 3.1 | 50.0 | 0.0043 / 0.0138 | 0/60 (0.0%) |
| Original ESMA (v1) | 19.0552 ± 0.3493 | 19.8952 ± 0.9158 | 0.6130 ± 0.0430 | 17.4 ± 3.7 | 50.0 | 0.0025 / 0.0481 | 45/60 (75.0%) |
| **ESMA v2 (defaults)** | **19.0576 ± 0.3506** | 19.8650 ± 0.9425 | 0.6118 ± 0.0437 | **12.6 ± 2.6** | **40.3** | **0.0001 / 0.0018** | **52/60 (86.7%)** |
| ESMA v2, full T (no early stop) | 19.0576 ± 0.3506 | 19.8653 ± 0.9443 | 0.6118 ± 0.0437 | 14.6 ± 2.8 | 50.0 | 0.0001 / 0.0018 | 53/60 (88.3%) |
| ESMA v2 + opt-in polish | 19.0576 ± 0.3506 | 19.8663 ± 0.9436 | 0.6118 ± 0.0438 | 13.6 ± 4.2 | 40.3 | 0.0001 / 0.0018 | 53/60 (88.3%) |
| *Exact Kapur optimum (DP reference)* | 19.0577 | 19.8924 | 0.6128 | – | – | 0 | 60/60 |

Reading the table:

* **Kapur's entropy.** v2's mean (19.0576) is within 0.0001 of the exact optimum's mean (19.0577); its worst
  image is 0.0018 below the optimum, versus 0.0138 for Standard SMA and 0.0481 for v1. It ends within one gray
  level of the optimal thresholds in 53/60 images (Standard SMA: 18/60; v1: 48/60).
* **Runtime.** With the same evaluation infrastructure for everyone, v2 is the fastest of the three
  (12.6 ms vs 13.7 ms for Standard SMA and 17.4 ms for v1) because adaptive termination stops it after
  40 iterations on average; at equal budget (full T) it costs 0.9 ms more than Standard SMA. Against the
  legacy implementation the user measured (≈ 0.21 s per run) every algorithm is now 12–17× faster.
* **PSNR and SSIM.** All three algorithms — and the exact Kapur optimum itself — are statistically
  indistinguishable (Section 8). The PSNR of the *exact* optimum beats Standard SMA's in 29/60 images (a coin
  flip), and the per-image correlation between Kapur gain and PSNR gain (v2 vs SMA) is r = +0.006. In other
  words, better Kapur optimization neither helps nor hurts PSNR; the −0.017 dB mean difference is noise around
  an objective that does not target PSNR.
* **Early stop cost.** Full T changes v2's result on 2/60 images (one more exact hit); the mean Kapur is
  identical to four decimals.
* **Opt-in polish.** Changes 1/60 images; v2 already polishes.

**All 100 Mendeley images** (`--subset all`; reference only — it includes the 40 development images on which
v2's defaults were chosen, so it is not a clean hold-out):

| Algorithm | Kapur's entropy | PSNR (dB) | SSIM | Runtime (ms) | Iterations | Gap (mean / max) | Exact optimum reached |
|---|---|---|---|---|---|---|---|
| Standard SMA | 19.0697 ± 0.3304 | 19.9341 ± 0.9024 | 0.6120 ± 0.0454 | 12.6 ± 2.5 | 50.0 | 0.0053 / 0.0220 | 0/100 |
| Original ESMA (v1) | 19.0722 ± 0.3305 | 19.9063 ± 0.9157 | 0.6134 ± 0.0452 | 16.3 ± 2.9 | 50.0 | 0.0028 / 0.0481 | 70/100 |
| **ESMA v2 (defaults)** | **19.0743 ± 0.3309** | 19.8942 ± 0.9105 | 0.6122 ± 0.0452 | **12.4 ± 2.7** | **41.2** | **0.0007 / 0.0672** | **84/100** |
| ESMA v2, full T | 19.0743 ± 0.3309 | 19.8937 ± 0.9111 | 0.6123 ± 0.0452 | 14.3 ± 2.9 | 50.0 | 0.0007 / 0.0672 | 86/100 |
| ESMA v2 + opt-in polish | 19.0743 ± 0.3309 | 19.8950 ± 0.9112 | 0.6122 ± 0.0452 | 12.1 ± 2.9 | 41.2 | 0.0007 / 0.0672 | 85/100 |
| *Exact Kapur optimum* | 19.0750 | 19.9065 | 0.6132 | – | – | 0 | 100/100 |

Pairwise on all 100: v2 vs SMA — Kapur +0.0046, better in 98/100, p = 7.8e-17; PSNR −0.040, p = 0.17 (n.s.);
SSIM +0.0002, p = 0.75 (n.s.); runtime −0.3 ms, 55/100, p = 0.41 (n.s.). v2 vs v1 — Kapur +0.0021, better in
26/100 with 66 ties and 8 worse, p = 0.0003; runtime −4.0 ms, 90/100, p = 3.6e-14; PSNR/SSIM n.s. The exact
optimum's PSNR (19.9065) is *below* Standard SMA's (19.9341) on this set (better in 44/100), which again shows
that PSNR differences are not a property of the optimizer. v2's one large miss (gap 0.0672, thresholds 19 levels
off in one coordinate) is a basin-level failure on a development image; on the hold-out set its worst gap is
0.0018.

Reference point on the other dataset (user's screenshot, 300 DENTEX images, legacy code): Standard SMA 19.2105,
ESMA v1 19.2155 Kapur; PSNR 19.9492 / 19.9302; SSIM 0.4986 / 0.5000; runtime 0.2161 s / 0.2186 s. The
three-way script reproduces that selection exactly with `--subset all --max_images 300 --seed 42` on the
machine that has the folder (add `--legacy_fitness` to reproduce the legacy runtimes first).

## 8. Statistical significance comparison

Paired Wilcoxon signed-rank tests over the 60 hold-out images (same test and "better in x/y images" convention
as `compare_full_images.py`; ties are reported because v1 and v2 reach the same thresholds on 43 images).

| Comparison | Metric | Baseline mean | Other mean | Mean gain | Other better (ties) | Wilcoxon p | Significant (α = 0.05) |
|---|---|---|---|---|---|---|---|
| ESMA v1 vs Standard SMA | Kapur's entropy | 19.0534 | 19.0552 | +0.0018 | 53/60 (88.3%) | 2.1e-05 | yes |
| ESMA v1 vs Standard SMA | PSNR | 19.8818 | 19.8952 | +0.0134 | 31/60 (51.7%) | 0.8309 | no |
| ESMA v1 vs Standard SMA | SSIM | 0.6109 | 0.6130 | +0.0022 | 33/60 (55.0%) | 0.6012 | no |
| ESMA v1 vs Standard SMA | Runtime (s) | 0.0137 | 0.0174 | +0.0037 | 9/60 (15.0%) | 1.3e-07 | yes (slower) |
| **ESMA v2 vs Standard SMA** | **Kapur's entropy** | 19.0534 | 19.0576 | **+0.0042** | **59/60 (98.3%)** | **1.7e-11** | **yes** |
| ESMA v2 vs Standard SMA | PSNR | 19.8818 | 19.8650 | −0.0168 | 28/60 (46.7%) | 0.5658 | no |
| ESMA v2 vs Standard SMA | SSIM | 0.6109 | 0.6118 | +0.0010 | 32/60 (53.3%) | 0.8252 | no |
| **ESMA v2 vs Standard SMA** | **Runtime (s)** | 0.0137 | 0.0126 | **−0.0011** | **38/60 (63.3%)** | **0.0316** | **yes (faster)** |
| ESMA v2 vs Standard SMA | Iterations | 50.0 | 40.3 | −9.7 | 51/60 (9 ties) | 4.9e-10 | yes |
| **ESMA v2 vs ESMA v1** | **Kapur's entropy** | 19.0552 | 19.0576 | **+0.0024** | 13/60 (43 ties, 4 worse) | **0.0113** | **yes** |
| ESMA v2 vs ESMA v1 | PSNR | 19.8952 | 19.8650 | −0.0302 | 7/60 (43 ties) | 0.1773 | no |
| ESMA v2 vs ESMA v1 | SSIM | 0.6130 | 0.6118 | −0.0012 | 7/60 (43 ties) | 0.4631 | no |
| **ESMA v2 vs ESMA v1** | **Runtime (s)** | 0.0174 | 0.0126 | **−0.0048** | **55/60 (91.7%)** | **1.9e-10** | **yes (faster)** |
| ESMA v2 full T vs ESMA v2 | Kapur's entropy | 19.0576 | 19.0576 | +0.0000 | 2/60 (58 ties) | 0.1797 | no |
| ESMA v2 full T vs ESMA v2 | Runtime (s) | 0.0126 | 0.0146 | +0.0020 | 16/60 (26.7%) | 6.6e-05 | yes (slower) |
| ESMA v2 + polish vs ESMA v2 | Kapur's entropy | 19.0576 | 19.0576 | +0.0000 | 1/60 (59 ties) | 0.3173 | no |

Note on the "v2 vs v1" Kapur row: on 43/60 images both reach the same (optimal) thresholds; of the 17 that
differ, v2 is better in 13 and v1 in 4, hence the significant but modest p = 0.011. The large-sample multi-seed
comparison on the development split (320 runs per algorithm: mean gap 0.00015 vs 0.00364, exact hits 88.1% vs
62.8%) tells the same story with much more power.

## 9. Trade-offs and limitations

* **Dataset.** Reported numbers come from 60 hold-out Mendeley OPGs (plus the all-100 reference table in
  Section 7, which includes the 40 development images), not from the 300 DENTEX images of the original
  screenshot, which are not on this machine. The three-way script is ready to
  run there; conclusions about *relative* behaviour are expected to transfer (the optimality-gap structure of
  Standard SMA and v1 was identical on both datasets), but the DENTEX numbers must be re-measured, not assumed.
* **Adaptive termination is a real trade-off.** It is the only change that reduces search effort. On the
  hold-out set it costs one exact hit in 60 and no change in mean Kapur; on the development split (320 runs)
  it costs 2.2 points of exact-hit rate (85.9% vs 88.1%) at 20% fewer iterations, and at T = 100 it stops after
  ≈ 40 iterations (86.2% vs 90.6%). The equal-budget variant is one flag away (`early_stop=False`) and is
  reported in every table.
* **PSNR cannot be improved by optimizing Kapur's entropy better.** The exact optimum's PSNR is indistinguishable
  from Standard SMA's. Any PSNR gain would require changing the objective (e.g. a Kapur/PSNR compromise) or the
  band-level assignment in `apply_thresholds` (class means instead of evenly spaced levels — a change to the
  evaluation, deliberately not made). The user's target "no longer worse than Standard SMA" is met only in the
  statistical sense (p = 0.57; 46.7% of images better), not as a mean improvement.
* **The z(t) branch is switched off by default** because every configuration tried spent budget that Kapur
  thresholding rewards more when spent on polishing (Section 5.5). Objective 3 is realized through a(t) and the
  adaptive termination; the thesis text should describe the exploration branch as implemented, evaluated and
  optional.
* **Small absolute effects.** On these images the whole headroom above Standard SMA is ≈ 0.004 Kapur units
  (0.02%). The meaningful claims are the ones the DP reference makes precise: v2 reaches the true optimum in
  87–88% of images with a worst-case gap of 0.0018, versus 0% / 0.0138 (SMA) and 75% / 0.0481 (v1).
* **Single benchmark seed.** The hold-out benchmark follows the original protocol (seed 42 for every image);
  the development-split studies used 3–8 seeds and agree with it.
* **Exact ties at float precision.** The table path reproduces the legacy trajectories exactly on all 100
  reference images, but two threshold vectors with exactly equal true entropy could in principle be ordered
  differently by the two implementations (differences ≈ 1e-14); `fast_fitness=False` remains available.
* **Implication of the exact optimum.** Because Kapur's criterion is separable, the DP finds the optimum for
  fixed small d in milliseconds. This report uses it only as a reference; the thesis should acknowledge it when
  discussing practical deployment.
* **Downstream components.** `train_dentex_classifier.py`, `sma_preprocess.py` and the YOLO pipeline still call
  `enhanced_sma` (v1); nothing they depend on changed (v1 output is byte-identical on the reference images).
  Switching them to v2 is a one-line change but would require retraining the Random-Forest models whose
  features came from v1 segmentations.

## 10. Final assessment

| Metric | Target | Standard SMA → ESMA v1 (before) | Standard SMA → ESMA v2 (after) | Verdict |
|---|---|---|---|---|
| Kapur's entropy | maintain/improve the advantage | +0.0018, 88.3% of images, p = 2e-5 (hold-out); +0.0050, 91.7% (DENTEX) | **+0.0042, 98.3% of images, p = 1.7e-11**; mean within 0.0001 of the exact optimum; worst gap 0.0018 vs 0.0481 (v1) | **Improved** (also vs v1: +0.0024, p = 0.011) |
| PSNR | not worse than SMA | −0.028 (Mendeley) / −0.019 (DENTEX), not significant | −0.017, 46.7%, p = 0.57, not significant; exact optimum also indistinguishable | **Similar** (statistically indistinguishable; cannot be improved through the Kapur objective) |
| SSIM | maintain/improve | +0.0014, not significant on Mendeley (significant on DENTEX) | +0.0010, p = 0.83, not significant | **Similar** |
| Runtime | ≤ Standard SMA, remove v1's overhead | v1 slower (+2.5 ms DENTEX; +3.7 ms hold-out, p = 1e-7) | **v2 faster than SMA (−1.1 ms, 63.3%, p = 0.03) and than v1 (−4.8 ms, 91.7%, p = 2e-10)**; all algorithms ≈ 15× faster than the legacy code | **Improved** |
| Stability | fewer bad outcomes | v1 max gap 0.0481, 75% exact | v2 max gap 0.0018, 87–88% exact | **Improved** |

Deliverables (all under `backend/`):

* `sma_algorithms.py` — `standard_sma` (unchanged rules; `fast_fitness` flag), `enhanced_sma` (v1, unchanged
  rules; `fast_fitness` flag), `enhanced_sma_v2`, `KapurEntropyTable`, `kapur_optimal_thresholds`,
  `_initial_population`, `_integer_polish`; module docstring documents D1–D6.
* `compare_three_way.py` (three-way benchmark with exact-optimum reference, all-pairs Wilcoxon, per-image JSON),
  `ablate_esma_v2.py` (component ablation), `bench_common.py` (shared loading and the dev/hold-out split).
* `compare_full_images.py` — duplicate-run bug fixed; CLI and output unchanged.
* `tests/test_sma_algorithms.py` + `tests/fixtures/mendeley_reference.npz` — 11 tests: table = scalar fitness,
  DP optimum is global (exhaustive check for d = 2), legacy = fast paths, legacy = frozen reference on 100 images,
  v2 schema/validity/determinism, every component switch, LHS stratification, polish monotonicity, v2 ≥ v1.
* `main.py` — additive `POST /analyze/improved/`; `/analyze/compare/` now also returns `improved`,
  `improvement_v2`, `improvement_v2_vs_v1`; `iterations_used` added to every packaged result. Existing
  endpoints and keys unchanged (the frontend needs no change).
* Result files: `three_way_comparison_holdout.json`, `three_way_comparison_all.json`, `esma_v2_ablation.json`
  (git-ignored, regenerable).

How to reproduce:

```
cd backend
python -m pytest tests -q
python ablate_esma_v2.py     --images_dir ../Datasets/images --subset dev --seeds 1,2,3
python compare_three_way.py  --images_dir ../Datasets/images --subset holdout --sma_population 30 --sma_iterations 50 --d 4 --seed 42
python compare_three_way.py  --images_dir <DENTEX xrays> --subset all --max_images 300 --seed 42      # original 300-image selection
python compare_three_way.py  --images_dir <DENTEX xrays> --subset all --max_images 300 --seed 42 --legacy_fitness --variants ""   # legacy-runtime reproduction
```
