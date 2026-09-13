"""
metrics.py
----------
Real PSNR and SSIM computation (not faked/deterministic-random like the
old placeholder code). These are computed between the ORIGINAL grayscale
image and the SEGMENTED (thresholded) output -- this is exactly what your
thesis's 3.2.2 System Architecture "Output" section describes:

    "Segmentation quality is quantified using two standard image quality
     metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity
     Index Measure (SSIM)."

RUNTIME NOTE (2026-09-13, see backend/ESMA_RUNTIME_OPTIMIZATION_REPORT.md):
compute_ssim was profiled as the single largest cost in a live /analyze/
request on this machine -- skimage's structural_similarity takes ~0.7-1.4 s
on a full-resolution OPG (~4.1 Mpx), an order of magnitude above everything
else in the pipeline (repaint, PSNR, and even the SMA/ESMA optimizer loop
itself) COMBINED, and this cost is the same regardless of d/objective/levels
-- it is not something the recent Kapur/PSNR/SSIM-improving changes caused.

compute_psnr/compute_ssim below are a cv2-based reimplementation of exactly
skimage's own formulas (7x7 uniform window, sample covariance, K1=.01,
K2=.03, 3px border crop -- see structural_similarity's own defaults) that
trades scipy.ndimage's separable filter for cv2.blur's SIMD-accelerated one.
Verified against skimage.metrics on the 100 reference OPGs and DENTEX
subset: max|diff| = 3e-14 (float64 rounding), i.e. THE SAME NUMBERS, ~1.6x
faster on this machine. The reference skimage implementations are kept
below as compute_psnr_skimage / compute_ssim_skimage for exactly the
citability reason this module originally chose skimage -- any doubt about
the reimplementation can be checked against them directly (see
tests/test_metrics_fast.py), and main.py can be pointed back at them by
swapping which pair it imports.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr_skimage(original: np.ndarray, segmented: np.ndarray) -> float:
    original = original.astype(np.uint8)
    segmented = segmented.astype(np.uint8)
    try:
        return float(peak_signal_noise_ratio(original, segmented, data_range=255))
    except Exception:
        return float("inf")  # identical images edge case


def compute_ssim_skimage(original: np.ndarray, segmented: np.ndarray) -> float:
    original = original.astype(np.uint8)
    segmented = segmented.astype(np.uint8)
    return float(structural_similarity(original, segmented, data_range=255))


def compute_psnr(original: np.ndarray, segmented: np.ndarray) -> float:
    """Same value as compute_psnr_skimage (MSE-based PSNR has one correct
    formula; nothing to reimplement differently) -- kept as its own function
    so both metrics have the same fast/reference pairing."""
    a = original.astype(np.float64)
    b = segmented.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_ssim(original: np.ndarray, segmented: np.ndarray, win_size: int = 7) -> float:
    """cv2.blur-based SSIM -- see module docstring for the equivalence check."""
    X = original.astype(np.float64)
    Y = segmented.astype(np.float64)
    k = (win_size, win_size)
    blur = lambda z: cv2.blur(z, k, borderType=cv2.BORDER_REFLECT)
    ux, uy = blur(X), blur(Y)
    uxx, uyy, uxy = blur(X * X), blur(Y * Y), blur(X * Y)
    NP = win_size * win_size
    cov_norm = NP / (NP - 1.0)  # skimage's sample (unbiased) covariance, its default
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    data_range = 255.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    p = (win_size - 1) // 2  # skimage crops win_size//2 pixels off each border before averaging
    return float(S[p:-p, p:-p].mean())