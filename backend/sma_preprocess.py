"""
sma_preprocess.py
-------------------
Generalized version of the ESMA-YOLO preprocessing integration:
supports EITHER Standard SMA or Enhanced SMA (ESMA) as the
preprocessing step before YOLO, so you can train and compare BOTH --
"YOLO + Standard SMA preprocessing" vs. "YOLO + ESMA preprocessing" --
mirroring the Standard-vs-Enhanced comparison structure used
everywhere else in this thesis (Chapter 4's core comparison, the
DENTEX classifier). This is what actually answers "does ESMA's
segmentation help the detector more than Standard SMA's does?" rather
than just showing ESMA integrated with no baseline to compare against.

This supersedes esma_preprocess.py (kept as a thin wrapper for
backward compatibility, in case anything else still imports it).
"""

import cv2

from sma_algorithms import (
    apply_thresholds,
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)

ALGO_FUNCS = {
    "standard": standard_sma,
    "enhanced": enhanced_sma,
    "enhanced_v2": enhanced_sma,
}

# The YOLO detectors were trained on images preprocessed with the ESMA as it was
# BEFORE the panel revision, so detection must use those same settings.
# "enhanced" = revised ESMA; "enhanced_v2" = ESMA before revision (used by the detector).
# These kwargs reproduce the old enhanced_sma_v2 thresholds exactly (checked 25/25
# DENTEX images at T=150, seed=42).
ALGO_KWARGS = {
    "standard": {},
    "enhanced": {},
    "enhanced_v2": dict(weight_mode="rank", adapt="cr_pd", accept_equal=False, kicks=0),
}


def sma_preprocess(image, algorithm="enhanced", d=4, N=30, T=150, seed=42,
                    adaptive_k=True, crop_borders=True):
    """
    Takes a raw image (grayscale or BGR) and returns a 3-channel BGR
    image where the pixel values have been replaced by the chosen
    algorithm's multilevel-thresholded (segmented) output.

    algorithm: "standard", "enhanced" (the revised ESMA), or
               "enhanced_v2" (the ESMA before revision -- what the YOLO
               detector's training data was preprocessed with).

    crop_borders=False keeps the output the SAME SIZE/framing as the
    input (only pixel intensities change, nothing is cropped) -- this
    matters when preparing YOLO training data, since the DENTEX
    bounding boxes are defined in the ORIGINAL image's pixel
    coordinates; cropping here would shift that coordinate system and
    silently misalign every box. Use crop_borders=True for standalone
    visualization use where no external bbox coordinates need to stay
    aligned with this output.
    """
    if algorithm not in ALGO_FUNCS:
        raise ValueError(f"algorithm must be one of {sorted(ALGO_FUNCS)}, got '{algorithm}'")
    algo_fn = ALGO_FUNCS[algorithm]

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist_source = autocrop_black_borders(gray) if crop_borders else gray
    prob = compute_histogram_prob(hist_source)

    result = algo_fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **ALGO_KWARGS[algorithm])

    target = hist_source if crop_borders else gray
    segmented = apply_thresholds(target, result["thresholds"])

    return cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
