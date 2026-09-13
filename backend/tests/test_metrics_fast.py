"""
Tests for the cv2-based compute_psnr/compute_ssim in backend/metrics.py
(added 2026-09-13 as a runtime optimization -- see
ESMA_RUNTIME_OPTIMIZATION_REPORT.md): they must agree with the
skimage-based compute_psnr_skimage/compute_ssim_skimage they replace to
within floating-point rounding, on both synthetic arrays and realistic
threshold-segmented images (large flat bands + sharp edges, exactly the
shape apply_thresholds produces). Synthetic-only so this suite does not
depend on the gitignored Datasets/ folder.
"""
import numpy as np
import pytest

from metrics import compute_psnr, compute_psnr_skimage, compute_ssim, compute_ssim_skimage
from sma_algorithms import apply_thresholds

RNG = np.random.default_rng(0)


def _random_gray(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _banded_pair(h, w, seed):
    """A grayscale image and its apply_thresholds output -- the ACTUAL pair
    passed to compute_psnr/compute_ssim in production (large uniform bands
    separated by sharp edges), not just random noise."""
    img = _random_gray(h, w, seed)
    # smooth it a bit so it looks like real anatomy, not salt-and-pepper noise
    img = ((img.astype(np.float64) + np.roll(img, 1, axis=0) + np.roll(img, 1, axis=1))
           / 3).astype(np.uint8)
    seg = apply_thresholds(img, [60, 110, 160, 200], levels="mean")
    return img, seg


@pytest.mark.parametrize("h,w,seed", [(64, 64, 1), (101, 137, 2), (50, 300, 3)])
def test_psnr_matches_skimage_random(h, w, seed):
    a = _random_gray(h, w, seed)
    b = _random_gray(h, w, seed + 100)
    fast = compute_psnr(a, b)
    ref = compute_psnr_skimage(a, b)
    assert fast == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("h,w,seed", [(64, 64, 1), (101, 137, 2), (50, 300, 3)])
def test_ssim_matches_skimage_random(h, w, seed):
    a = _random_gray(h, w, seed)
    b = _random_gray(h, w, seed + 100)
    fast = compute_ssim(a, b)
    ref = compute_ssim_skimage(a, b)
    assert fast == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("h,w,seed", [(128, 256, 10), (301, 199, 11), (400, 400, 12)])
def test_psnr_matches_skimage_thresholded(h, w, seed):
    """The realistic case: original image vs. its own thresholded/repainted
    segmentation -- large flat regions, exactly what main.py scores."""
    img, seg = _banded_pair(h, w, seed)
    fast = compute_psnr(img, seg)
    ref = compute_psnr_skimage(img, seg)
    assert fast == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("h,w,seed", [(128, 256, 10), (301, 199, 11), (400, 400, 12)])
def test_ssim_matches_skimage_thresholded(h, w, seed):
    img, seg = _banded_pair(h, w, seed)
    fast = compute_ssim(img, seg)
    ref = compute_ssim_skimage(img, seg)
    assert fast == pytest.approx(ref, abs=1e-9)


def test_psnr_identical_images_is_inf():
    img = _random_gray(50, 50, 42)
    assert compute_psnr(img, img) == float("inf")
    assert compute_psnr_skimage(img, img) == float("inf")


def test_ssim_identical_images_is_one():
    img = _random_gray(50, 50, 42)
    assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-9)
