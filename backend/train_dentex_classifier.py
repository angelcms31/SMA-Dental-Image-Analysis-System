"""
train_dentex_classifier.py
---------------------------
Full pipeline: DENTEX JSON + images -> per-tooth crop -> SMA/ESMA
segmentation (feature extraction) -> Random Forest classifier ->
accuracy report, comparing Standard SMA-derived features against
Enhanced SMA (ESMA)-derived features.

This directly tests the thesis's core hypothesis on a DOWNSTREAM task:
if ESMA produces better segmentation than Standard SMA, a classifier
built on ESMA's features should classify diagnoses more accurately
than one built on Standard SMA's features, using DENTEX as an
independent, externally-validated ground truth.

WHAT THIS DOES NOT DO: this is not literally "SMA/ESMA diagnoses
teeth." SMA/ESMA only produce a segmentation (a threshold vector +
resulting regions) for each crop -- exactly like the rest of your
thesis. The classifier is a separate, standard supervised-learning
step trained on those segmentation-derived numbers against DENTEX's
dentist-verified labels.

Usage:
    python train_dentex_classifier.py \
        --json train_quadrant_enumeration_disease.json \
        --images_dir ./train_images \
        --sma_iterations 40 --sma_population 20

Requires: opencv-python-headless, numpy, scikit-learn, scikit-image
(same stack as the rest of the thesis system -- see requirements.txt)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from sma_algorithms import (
    apply_thresholds,
    compute_histogram_prob,
    enhanced_sma,
    standard_sma,
)


def load_dentex(json_path, images_dir):
    """Yields (gray_crop, label_name) for every annotated tooth."""
    with open(json_path) as f:
        data = json.load(f)

    id_to_label = {c["id"]: c["name"] for c in data["categories_3"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    for ann in data["annotations"]:
        filename = id_to_file.get(ann["image_id"])
        if filename is None:
            continue
        img_path = os.path.join(images_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape
        x, y, bw, bh = ann["bbox"]
        pad = 8
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(w, int(x + bw + pad)), min(h, int(y + bh + pad))
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue

        crop = image[y0:y1, x0:x1]
        label = id_to_label[ann["category_id_3"]]
        yield crop, label


def extract_features(crop, algo_fn, N, T, seed=0):
    """
    Runs one SMA variant on this crop and turns its segmentation result
    into a fixed-length numeric feature vector for the classifier.
    """
    prob = compute_histogram_prob(crop)
    d = 2  # 2 thresholds -> 3 bands (dark / mid / bright) -- small crop,
           # keep the search space modest so this runs fast per-tooth
    result = algo_fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
    thresholds = result["thresholds"]
    entropy = result["fitness"]

    h, w = crop.shape
    crop_area = h * w
    darkest_hi = thresholds[0] if thresholds else 128

    dark_mask = (crop < darkest_hi).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        bx, by, bw_, bh_ = cv2.boundingRect(largest)
        cx, cy = bx + bw_ / 2.0, by + bh_ / 2.0
        rel_x = cx / w          # 0 = left edge, 1 = right edge
        rel_y = cy / h          # 0 = top (crown side), 1 = bottom (root side)
        area_ratio = area / crop_area
        blob_aspect = bw_ / float(bh_) if bh_ > 0 else 0
        n_blobs = len(contours)
    else:
        rel_x = rel_y = area_ratio = blob_aspect = 0.0
        n_blobs = 0

    crop_aspect = w / float(h) if h > 0 else 0

    return np.array([
        entropy,
        *[float(t) / 255.0 for t in (thresholds + [0, 0])[:2]],  # pad to 2
        rel_x, rel_y, area_ratio, blob_aspect, n_blobs, crop_aspect,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "kapur_entropy", "threshold_1_norm", "threshold_2_norm",
    "dark_region_x", "dark_region_y", "dark_area_ratio",
    "dark_blob_aspect", "n_dark_blobs", "crop_aspect",
]


def run_pipeline(json_path, images_dir, N, T, test_size, max_samples, seed):
    print("Loading DENTEX crops...")
    crops, labels = [], []
    for crop, label in load_dentex(json_path, images_dir):
        crops.append(crop)
        labels.append(label)
        if max_samples and len(crops) >= max_samples:
            break
    print(f"Loaded {len(crops)} annotated tooth crops.")
    if len(crops) < 20:
        print("WARNING: very few samples -- results will not be meaningful. "
              "Point --json/--images_dir at the full training set (705 images), "
              "not the 50-image validation set.")

    for algo_name, algo_fn in [("standard", standard_sma), ("enhanced", enhanced_sma)]:
        print(f"\nExtracting features using {algo_name} SMA "
              f"({len(crops)} crops, N={N}, T={T})...")
        t0 = time.perf_counter()
        X = np.array([extract_features(c, algo_fn, N, T, seed=seed) for c in crops])
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=seed, stratify=labels
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print(f"\n=== {algo_name.upper()} SMA -- classifier results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
        print("Top features:")
        for name, imp in importances[:5]:
            print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--sma_population", type=int, default=20, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=30, dest="T")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap total crops for a quick test run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.json, args.images_dir, args.N, args.T, args.test_size, args.max_samples, args.seed)