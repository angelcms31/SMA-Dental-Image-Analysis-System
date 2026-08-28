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

Both use scikit-image's implementations so the numbers are citable /
reproducible (not something you have to defend as home-grown math).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(original: np.ndarray, segmented: np.ndarray) -> float:
    original = original.astype(np.uint8)
    segmented = segmented.astype(np.uint8)
    try:
        return float(peak_signal_noise_ratio(original, segmented, data_range=255))
    except Exception:
        return float("inf")  # identical images edge case


def compute_ssim(original: np.ndarray, segmented: np.ndarray) -> float:
    original = original.astype(np.uint8)
    segmented = segmented.astype(np.uint8)
    return float(structural_similarity(original, segmented, data_range=255))