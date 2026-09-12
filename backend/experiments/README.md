# Metric experiments (Kapur / PSNR / SSIM) — September 2026

Scripts and raw results behind `../ESMA_METRICS_IMPROVEMENT_REPORT.md`. Everything runs from this folder on the
100 Mendeley OPGs in `../../Datasets/images` (N = 30, T = 100, seed 42; optimality gaps over seeds 1–3) and uses
4 worker processes. Run times on a 4-core laptop are given per script.

| File | What it does | Output |
|---|---|---|
| `fastmetrics.py` | PSNR / SSIM reproducing scikit-image's defaults (7×7 uniform window, sample covariance, K1 = 0.01, K2 = 0.03, border crop) with `cv2.blur`; agrees with `metrics.py` to 1e-14. Also `reconstruct()` (even / mean repaint, same as `sma_algorithms.apply_thresholds`). | — |
| `exp_metrics.py [n_c]` | **A** d = 4: Standard SMA / ESMA v2 / exact DP optimum under the Kapur, Otsu and normalized hybrid objectives; PSNR + SSIM under fixed-level and class-mean repaint. **B** d = 4, 6, 8, 10, 12 with the Kapur objective: gap / hit rate (3 seeds), PSNR / SSIM (class means). **C** SSIM-aware non-separable objective (w = 0.5 and 0.8) on the first `n_c` hold-out images (default 20). ≈ 27 min. | `exp_results.json` |
| `summarize_exp.py [all\|holdout]` | Tables A, A1–A5, B, B1, C with paired Wilcoxon tests from `exp_results.json`. | stdout |
| `exp_hybrid_d.py` | Hybrid objective **in Kapur units** (default `HybridObjectiveTable(scale="kapur")`) at d = 4, 8, 10: Standard SMA / ESMA v2 / exact optimum, gaps over 3 seeds, PSNR / SSIM (class means). ≈ 5 min. | `exp_hybrid_d.json` |
| `summarize_hybrid.py [file]` | Tables for `exp_hybrid_d.json` (or `exp_hybrid_d_norm.json`, the normalized-form run) joined with the Kapur-objective runs of `exp_results.json` on the same images. | stdout |
| `exp_results.json` | Raw per-image results of `exp_metrics.py` (100 images). | — |
| `exp_hybrid_d.json` | Raw per-image results of `exp_hybrid_d.py` (hybrid in Kapur units). | — |
| `exp_hybrid_d_norm.json` | Earlier run of the hybrid in its normalized [0, 1] form at d = 8, 10 (kept because it shows the scale sensitivity of Standard SMA's `tanh` gate). | — |

Reproduce:

```
cd backend/experiments
python exp_metrics.py 20 && python summarize_exp.py all && python summarize_exp.py holdout
python exp_hybrid_d.py && python summarize_hybrid.py
```

For thesis-grade runs on the DENTEX folder use the main benchmark instead, which now takes the same options:

```
cd backend
python compare_three_way.py --images_dir <xrays> --subset all --max_images 300 --seed 42 --skip_v1 --recon mean
python compare_three_way.py --images_dir <xrays> --subset all --max_images 300 --seed 42 --skip_v1 --recon mean --objective hybrid --d 8
```
