# Runtime after the Kapur/PSNR/SSIM metric improvements — what changed, what was fixed, and what's a real trade-off

13 September 2026. Companion to `ESMA_METRICS_IMPROVEMENT_REPORT.md` (the metric-improving levers this report
explains the cost of) and `ESMA_V2_REPORT.md`. Code touched: `backend/metrics.py`, `backend/main.py`,
`backend/sma_algorithms.py` (`band_levels`/`apply_thresholds` gain an optional `hist=` param), new tests
`backend/tests/test_metrics_fast.py`, `backend/tests/test_shared_tables.py` (38/38 pass, up from 30).

## 0. Summary

Three things were true at once, and they needed to be separated before "fix the runtime" meant anything:

1. **A real, measured, ~1.4–2.8x slowdown of the algorithm's own internal runtime** — the number the app calls
   "Runtime (s)" — comes directly from adopting the protocol that produces the improved Kapur/PSNR/SSIM numbers
   (`objective="hybrid"`, `d=8–10`, `levels="mean"`). Two-thirds of that was pure waste (rebuilding the same
   fitness table redundantly, recomputing a histogram that was already sitting in memory) and has been fixed with
   **zero effect on any reported value** (verified by new tests). The remaining third is not waste — it is ESMA v2
   legitimately spending more iterations on a harder search space, which is the same mechanism that produces the
   metric gains in the first place, and removing it would give the gains back.
2. **A separate, much larger cost dominates total request time regardless of the protocol**: SSIM computation on
   the full-resolution OPG (~4.1 Mpx) took 0.7–1.4 s per request on this machine — 10–40x everything else in the
   pipeline *combined*, including the SMA/ESMA optimizer loop itself — and this cost is identical whether the
   request uses the old or the new protocol. It was not caused by the metric-improving work, but it is the single
   biggest lever available, so it is fixed too (cv2-based reimplementation, verified numerically equivalent).
3. **A profiling artifact that must not be mistaken for either of the above**: my own Python test harness measured
   full requests to `http://localhost:8000` as ~2 s slower than identical requests to `http://127.0.0.1:8000`, on
   this Windows machine, using Python's `requests` library. Verified directly in Chrome via `fetch()` against the
   running frontend: **the browser shows no such gap** (48 ms first request, 3 ms after, identical to `127.0.0.1`) —
   Chrome's Happy-Eyeballs connection logic avoids exactly the sequential IPv6-then-IPv4 fallback that `requests`
   does not. This does not affect real users of the app and is noted only so it is not mistaken for a code issue if
   re-discovered later.

## 1. What specifically caused the "Runtime (s)" slowdown

Measured with `N=30, T=100, seed=42` (the live app's own values), on the 100 Mendeley OPGs already in
`Datasets/images`, using this repo's own existing experiment data (`backend/experiments/exp_hybrid_d.json`,
generated for `ESMA_METRICS_IMPROVEMENT_REPORT.md` section 4.4) plus new isolated timing:

| Cause | Where | Measured cost | Fixed? |
|---|---|---|---|
| `objective="hybrid"` builds a `KapurEntropyTable` **and** a `BetweenClassVarianceTable` and runs the exact-DP optimum **twice** (once per table) just to get the two normalization constants | `HybridObjectiveTable.__init__` | Algorithm-internal runtime at d=4: 18.0 ms (`kapur`) → 35.1 ms (`hybrid`), **~1.95x**, on identical images (DENTEX-300 baseline vs. Mendeley hybrid run) | Partly — see §3.1 |
| At d=8–10 ESMA v2's own adaptive early-stop (`patience=20`) fires much later, because the harder landscape keeps improving for longer | `enhanced_sma_v2`'s convergence loop | Mean iterations used: 41.5 (d=4) → 77.7 (d=8) → 81.5 (d=10), essentially **2x**; runtime scales in lockstep (35.1 ms → 61.3 ms → 70.4 ms) | No — see §4, this is the metric gain's own mechanism |
| `levels="mean"` repaint recomputes a full 256-bin histogram of the whole image from scratch | `band_levels(..., levels="mean")` | 28.1 ms (`even`) → 64.2 ms (`mean`) on a real OPG, **2.3x**, even though the identical histogram was already computed one line earlier for the population fitness | Yes — see §3.2 |
| Standard SMA's own per-iteration cost grows mildly with `d` (larger position arrays), but it always runs the fixed `T=100` regardless of objective/d | `standard_sma` | 47.7 ms → 67.0 ms, d=4→10, **1.4x** — the smallest of the four effects, expected and benign | N/A, not fixable without changing `N`/`T` |

Net effect on the number the UI actually displays, measured live end-to-end through the real endpoints (via
`127.0.0.1`, avoiding the artifact in §5):

| Endpoint | Baseline (d=4, kapur, even) `Runtime (s)` | Protocol (d=10, hybrid, mean) `Runtime (s)` | Ratio |
|---|---|---|---|
| `/analyze/standard/` | 0.0187 s | 0.0258 s | 1.4x |
| `/analyze/improved/` (ESMA v2) | 0.0119 s | 0.0334 s | 2.8x |

Both numbers are, and remain, tiny in absolute terms (tens of milliseconds) — the "much slower" observation the
task described is real but was never about a multi-second regression; see §2 for what actually takes seconds.

## 2. The much bigger, unrelated cost that was actually dominating total request time

Full pipeline profiling (`apply_thresholds` → `compute_psnr` → `compute_ssim` → `generate_annotated_overlay` →
image encoding), 15 real OPGs, before any fix:

| Stage | Mean time | Share of a single-algorithm request |
|---|---|---|
| SMA/ESMA optimizer loop | 30–70 ms | ~2% |
| Band repaint (`apply_thresholds`) | 28–64 ms | ~2% |
| PSNR (skimage) | ~100 ms | ~5% |
| **SSIM (skimage `structural_similarity`)** | **700–1400 ms** | **~65–85%** |
| Annotated overlay generation | ~650 ms | ~30% |
| PNG + base64 encoding (×2 images) | ~450 ms | ~20% |

(Percentages don't sum to 100 because they're each measured against different full-request totals across
configurations; the point is SSIM and overlay generation together account for the large majority of every request,
**identically** whether the request uses `objective="kapur"/d=4` or `objective="hybrid"/d=10`.) Confirmed with
`cProfile`: skimage's `structural_similarity` spends 320 ms of its own time on elementwise array arithmetic over
~4.1M-element float64 arrays, plus 330 ms in 10 calls to `scipy.ndimage.uniform_filter1d`, on top of the array
conversions — none of it touches `d`, `objective`, or `levels`.

**This is not something the recent Kapur/PSNR/SSIM work caused.** It has been true since `metrics.py` was written
(its own docstring explicitly chose skimage "so the numbers are citable / reproducible" — a reasonable choice that
just turned out to be slow on this machine). It is flagged here because it is by far the largest lever available if
the goal is a faster app, separate from the specific regression the task asked about.

## 3. Fixes applied (zero effect on any reported value — verified by tests)

### 3.1 Share one fitness table across `/analyze/compare/`'s three algorithm runs

`/analyze/compare/` runs Standard SMA, ESMA and ESMA v2 on the **same image, same `d`, same `objective`** — but
each of the three calls independently rebuilt its own `HybridObjectiveTable` (2 tables + 2 exact-DP passes) or
`KapurEntropyTable`. That's 3x the objective-table work of a single `/analyze/` call for no reason. `main.py` now
has `_shared_tables(prob, d, objective)`, called once per `/analyze/compare/` request; its result is passed through
`_run_and_package`'s new `table=`/`kapur_table=` parameters, which — when provided — are used for fitness
evaluation instead of rebuilding. `objective="kapur"` is left completely untouched (`_shared_tables` returns
`(None, None)` and `_run_and_package` falls back to its original per-call behaviour exactly), since building a
single `KapurEntropyTable` is already cheap and this keeps the default/thesis code path unchanged.

`/analyze/standard/`, `/analyze/enhanced/`, `/analyze/improved/` each still build their own table once per request
— correctly, since each is an independent request for one algorithm; there is nothing to share there.

**Verified a pure implementation change**: `backend/tests/test_shared_tables.py` calls `_run_and_package` with and
without a shared table (for `objective="otsu"` and `"hybrid"`, and across two different algorithms sharing the same
table) and asserts identical thresholds, `objective_value`, `kapur_entropy_fitness`, PSNR and SSIM in every case.

### 3.2 Stop recomputing the histogram for `levels="mean"`

`compute_histogram_prob(image)` is already computed once per request (it's what the SMA population fitness runs
on). `band_levels(..., levels="mean")` used to recompute a second, independent full-image histogram just to get
the per-band means. The per-band mean is a ratio (`sum(hist[lo:hi] * levels) / sum(hist[lo:hi])`), which is
invariant to any positive scaling of `hist` — so the already-normalized probability array works exactly as well as
raw counts. `band_levels`/`apply_thresholds` gained an optional `hist=` parameter; `main.py` now passes its
existing `prob` through on every call. Zero behavior change (same tests in `test_objectives_repaint.py` still pass
unmodified), one fewer full-image pass per request.

### 3.3 Replace skimage's PSNR/SSIM with an already-proven, faster equivalent

`backend/experiments/fastmetrics.py` already existed — built for `ESMA_METRICS_IMPROVEMENT_REPORT.md`'s own
100-image experiments, and *already documented there* as agreeing with `metrics.py` to 1e-14, because those
experiments would otherwise have taken far too long to run with skimage. It uses `cv2.blur` (SIMD-accelerated)
instead of `scipy.ndimage`'s separable filter for the exact same 7x7-window, sample-covariance SSIM formula. The
production API never adopted it. `metrics.py`'s `compute_psnr`/`compute_ssim` are now this implementation;
the original skimage versions are kept as `compute_psnr_skimage`/`compute_ssim_skimage` for exactly the
citability reason the module originally chose skimage — either can be checked against the other at any time.

**Verified**: `backend/tests/test_metrics_fast.py`, 14 new tests, comparing the two implementations on synthetic
random images and on realistic threshold-segmented images (large flat bands + sharp edges — the actual shape
`apply_thresholds` produces, not just noise), asserting agreement to 1e-9. Measured on real OPGs: PSNR unchanged in
speed (both ~50–100 ms; PSNR's formula has one correct implementation), SSIM ~1.6x faster on this machine
(1289 ms → 812 ms mean, 15 images) with **max observed difference 3e-14** — i.e. the same numbers.

*Not touched, out of scope for this pass*: annotated-overlay generation (~650 ms, ~30% of a request) was profiled
but not modified — it wasn't implicated in the Kapur/PSNR/SSIM regression and changing it risks the detected-region
boxes shown in the Findings panel, which deserves its own review rather than a change bundled into a runtime report.

## 4. The trade-off: what's left is not a bug

The remaining ~2x growth in ESMA v2's own iteration count from d=4 to d=10 (§1, row 2) is the *mechanism* by which
the metric gains happen, not overhead alongside it. `ESMA_METRICS_IMPROVEMENT_REPORT.md` §4.4 already measured why:
the search space grows as `255^d/d!`, Standard SMA's optimality gap grows ~70x from d=4 to d=12, and ESMA v2's own
gap grows more slowly but still grows — it needs more of its iteration budget to keep polishing a harder problem,
and `patience=20` correctly lets it use that budget instead of stopping early on a landscape that is still
improving. Cutting iterations back down would directly reduce the Kapur/PSNR/SSIM separation between Standard SMA
and ESMA v2 that the whole protocol change exists to produce — this is a dial, not a defect, and `esma-v2-project-
state`/the report's own §5.5 already flag "equal-budget, larger-T ablations at high d" as future work, not a
decision to make silently inside a runtime-optimization pass.

The two structural fixes in §3.1–3.2 have no such trade-off (verified byte-for-byte/1e-12 identical outputs) and
were applied outright. §3.3 has a documented, checked equivalence (1e-9, tests included) rather than a silent
swap, given the module's own stated reason for choosing skimage originally.

### 4.1 A lever that was tested and rejected: downsampling before PSNR/SSIM

Downsampling before computing SSIM is a real, precedented technique — `ESMA_METRICS_IMPROVEMENT_REPORT.md` §4.6
already uses an 8x-downsampled SSIM *inside* the optimization loop for its SSIM-aware objective experiment.
Applying it to the *final, reported* PSNR/SSIM (computed once per request, not per iteration) was measured instead
of assumed safe, because these OPGs are large uniform threshold bands separated by sharp edges — exactly the
content where SSIM's local-structure term is most sensitive to resampling:

| Downsample factor | Pixels | PSNR time | SSIM time | mean \|PSNR error\| | mean \|SSIM error\| |
|---|---|---|---|---|---|
| 1x (none) | 4.11 M | 51.4 ms | 466.7 ms | 0.0000 dB | 0.00000 |
| 2x | 1.03 M | 13.7 ms | 115.8 ms | 0.25 dB | **0.062** |
| 4x | 0.26 M | 4.1 ms | 30.5 ms | 0.66 dB | **0.094** |

The mean SSIM error at 2x downsampling (0.062) is **15–30x larger** than the actual ESMA-v2-vs-Standard-SMA effect
sizes this whole line of work is trying to measure (+0.002 to +0.004 SSIM at d=10, per the metrics report). Using
it would not speed up a real bottleneck safely — it would silently erase the very signal being reported.
**Rejected; not applied.** This is the kind of trade-off the task's constraint ("do not sacrifice the metric gains
to improve runtime") rules out, and it is included here specifically because it looked promising before measurement.

## 5. A profiling artifact, not a runtime finding — noted so it isn't rediscovered as a "fix"

Using Python's `requests` library against `http://localhost:8000` measured as ~2 s slower per request than the
identical request against `http://127.0.0.1:8000`, consistently, regardless of payload size (reproduced with a 4 KB
synthetic image and with a real 1.5 MB OPG). Diagnosis: `getaddrinfo('localhost', ...)` on this machine returns the
IPv6 loopback (`::1`) before the IPv4 one, nothing listens on `[::1]:8000` (uvicorn binds `127.0.0.1` only), and
`requests`/`urllib3` — unlike a browser — does not implement Happy Eyeballs (RFC 8305): it tries the first address
sequentially and only falls back after a real delay. Reusing a `requests.Session` (keep-alive) made every request
*after* the first as fast as `127.0.0.1` (confirming it's a per-new-connection cost, not per-request compute).

**Verified this does not reach real users**: the same `fetch()` calls the frontend actually makes, run directly
in Chrome against the live app, showed 48 ms (first request) then 3 ms (subsequent) for `localhost` — indistin-
guishable from `127.0.0.1` (6.5 ms / 3.7 ms). Chrome's connection logic avoids the slow path entirely. Every
wall-clock number reported elsewhere in this report was measured via `127.0.0.1` (or a reused `Session`) to avoid
this artifact contaminating the real findings.

## 6. What to do next, in priority order

1. Nothing further is required for §3's fixes — they're in, tested, and safe by construction.
2. If total request latency (not just the `Runtime (s)` field) needs to come down further, profile and optimize
   `generate_annotated_overlay` (overlay.py) next — it is now the second-largest cost after SSIM (~650 ms,
   unaffected by this report's fixes) and was out of scope here.
3. If the metric-improving protocol's ~2x ESMA v2 iteration growth at d≥8 needs to be smaller, that is a genuine
   quality/speed decision (re-ablate `patience`/`T` at high d, as §4's own limitations section already flags) —
   not something to change without re-measuring the Kapur/PSNR/SSIM gap it would give back.
4. Do not downsample before the reported PSNR/SSIM (§4.1) — measured and rejected for this image content.

## 7. Literature

* Search-space growth with `d` and its effect on optimizer difficulty (not a runtime bug, an expected metaheuristic
  property): `ESMA_METRICS_IMPROVEMENT_REPORT.md` §6.1, already citing the SMA-MLS (2024) and Lévy-flight ESMA
  (Entropy, 2021) results that thresholding algorithms separate more, and need more iterations to do so, as
  threshold count increases.
* Precomputed/tabulated separable objectives (`KapurEntropyTable`, `BetweenClassVarianceTable`) turning an
  O(population x d x bins) per-iteration cost into O(population x d) table lookups after one O(bins²) setup pass
  is the standard implementation trick for multilevel Kapur/Otsu thresholding once the objective is recognized as a
  sum over classes — the same structure that makes the exact DP optimum (`kapur_optimal_thresholds`) tractable at
  all (documented in `sma_algorithms.py`'s own `KapurEntropyTable` docstring).
* Adaptive/early termination trading a small solution-quality cost for a large iteration-count saving is a standard
  metaheuristics technique; `esma-v2-project-state` already documents this project's own ablation of it
  (`patience=20`: ~20% fewer iterations than full-T for a 0.00015→0.00015 mean-gap change at d=4, T=50).
* SSIM's per-window filtering cost on multi-megapixel images, and downsampling as a common but content-dependent
  mitigation, is well documented in image-quality-assessment practice (e.g. video-quality-metric literature
  routinely downsamples before SSIM) — but is only safe when the downsampled image still contains the structure
  being compared; §4.1 measured, rather than assumed, that this OPG/threshold-band content does not meet that bar
  at the effect sizes in play here.
