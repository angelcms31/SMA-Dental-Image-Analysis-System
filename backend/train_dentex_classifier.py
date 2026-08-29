"""
train_dentex_classifier.py
---------------------------
Full pipeline: DENTEX JSON + images -> per-tooth crop -> SMA/ESMA
segmentation (feature extraction) -> Random Forest classifier ->
cross-validated accuracy report, comparing Standard SMA-derived
features against Enhanced SMA (ESMA)-derived features. Saves the
final trained model for each algorithm to disk.

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

WHY CROSS-VALIDATION: a single train/test split on a small dataset
(e.g. 100 crops -> only ~25 in the test set, split across 4 unequal
classes) is noisy -- one lucky/unlucky prediction swings the F1-score
a lot. 5-fold stratified cross-validation evaluates on every sample
exactly once (each fold takes a turn as the held-out set) and reports
mean +/- std, which is what you want to defend in Chapter 4.

Usage:
    python train_dentex_classifier.py \
        --json train_quadrant_enumeration_disease.json \
        --images_dir ./xrays \
        --sma_iterations 40 --sma_population 20

Requires: opencv-python-headless, numpy, scikit-learn, scikit-image,
joblib (same stack as the rest of the thesis system -- see requirements.txt)
"""

import argparse
import json
import os
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sma_algorithms import compute_histogram_prob, enhanced_sma, standard_sma


def load_dentex(json_path, images_dir):
    """Yields (gray_crop, label_name) for every annotated tooth."""
    with open(json_path) as f:
        data = json.load(f)

    id_to_label = {c["id"]: c["name"] for c in data["categories_3"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    n_skipped = 0
    for ann in data["annotations"]:
        filename = id_to_file.get(ann["image_id"])
        if filename is None:
            n_skipped += 1
            continue
        img_path = os.path.join(images_dir, filename)

        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            # a single corrupted/truncated file shouldn't kill the whole
            # run -- skip it and keep going, but tell the user which file
            print(f"  [skip] could not read '{filename}' ({e.__class__.__name__}: {e})")
            n_skipped += 1
            continue

        if image is None:
            print(f"  [skip] '{filename}' decoded to None (missing, unreadable, or not an image)")
            n_skipped += 1
            continue

        h, w = image.shape
        x, y, bw, bh = ann["bbox"]
        pad = 8
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(w, int(x + bw + pad)), min(h, int(y + bh + pad))
        if x1 - x0 < 10 or y1 - y0 < 10:
            n_skipped += 1
            continue

        crop = image[y0:y1, x0:x1]
        label = id_to_label[ann["category_id_3"]]
        yield crop, label

    if n_skipped:
        print(f"  ({n_skipped} annotations skipped due to missing/corrupted/too-small images)")


def extract_features(crop, algo_fn, N, T, seed=0, algo_kwargs=None):
    """
    Runs one SMA variant on this crop and turns its segmentation result
    into a fixed-length numeric feature vector for the classifier.
    """
    prob = compute_histogram_prob(crop)
    d = 2  # 2 thresholds -> 3 bands (dark / mid / bright) -- small crop,
           # keep the search space modest so this runs fast per-tooth
    result = algo_fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **(algo_kwargs or {}))
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


def run_pipeline(json_path, images_dir, N, T, n_folds, max_samples, seed, out_dir, use_adaptive_k=False):
    print("Loading DENTEX crops...")
    crops, labels = [], []
    for crop, label in load_dentex(json_path, images_dir):
        crops.append(crop)
        labels.append(label)
        if max_samples and len(crops) >= max_samples:
            break
    labels = np.array(labels)
    print(f"Loaded {len(crops)} annotated tooth crops.")
    if len(crops) < 50:
        print("WARNING: very few samples -- results will not be reliable. "
              "Point --json/--images_dir at the full training set (705 images), "
              "not the 50-image validation set.")

    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    algo_specs = [
        ("standard", standard_sma, {}),
        ("enhanced", enhanced_sma, {"adaptive_k": True} if use_adaptive_k else {}),
    ]

    for algo_name, algo_fn, algo_kwargs in algo_specs:
        print(f"\nExtracting features using {algo_name} SMA "
              f"({len(crops)} crops, N={N}, T={T}"
              f"{', adaptive_k=True' if algo_kwargs.get('adaptive_k') else ''})...")
        t0 = time.perf_counter()
        feature_rows = []
        good_labels = []
        n_failed = 0
        for i, (crop, lbl) in enumerate(zip(crops, labels)):
            if i > 0 and i % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  ...{i}/{len(crops)} crops done ({elapsed:.1f}s elapsed)", flush=True)
            try:
                feature_rows.append(extract_features(crop, algo_fn, N, T, seed=seed, algo_kwargs=algo_kwargs))
                good_labels.append(lbl)
            except Exception as e:
                n_failed += 1
                print(f"  [skip] feature extraction failed on crop {i} "
                      f"({e.__class__.__name__}: {e})", flush=True)
        X = np.array(feature_rows)
        labels_for_algo = np.array(good_labels)
        print(f"  done in {time.perf_counter() - t0:.1f}s "
              f"({len(X)} usable, {n_failed} failed)", flush=True)
        if len(X) < 20:
            print("  Too few usable samples for this algorithm -- skipping.")
            continue

        # class counts must support the requested number of folds
        min_class_count = min(np.unique(labels_for_algo, return_counts=True)[1])
        folds = min(n_folds, min_class_count)
        if folds < n_folds:
            print(f"  NOTE: smallest class has only {min_class_count} samples -- "
                  f"reducing folds from {n_folds} to {folds} so every fold has "
                  f"at least one sample of every class.")
        if folds < 2:
            print("  Not enough samples of the rarest class to cross-validate "
                  "(need at least 2). Skipping CV for this algorithm -- get "
                  "more samples of the rare class(es) before reporting results.")
            continue

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")

        y_pred = cross_val_predict(clf, X, labels_for_algo, cv=skf)
        acc = accuracy_score(labels_for_algo, y_pred)
        f1_macro = f1_score(labels_for_algo, y_pred, average="macro", zero_division=0)

        # per-fold accuracy, for the mean +/- std you'll want to report
        fold_accs = []
        for train_idx, test_idx in skf.split(X, labels_for_algo):
            fold_clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
            fold_clf.fit(X[train_idx], labels_for_algo[train_idx])
            fold_accs.append(fold_clf.score(X[test_idx], labels_for_algo[test_idx]))
        fold_accs = np.array(fold_accs)

        print(f"\n=== {algo_name.upper()} SMA -- {folds}-fold cross-validated results ===")
        print(f"Accuracy (pooled across folds): {acc:.4f}")
        print(f"Accuracy per fold: {fold_accs.mean():.4f} +/- {fold_accs.std():.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(classification_report(labels_for_algo, y_pred, zero_division=0))

        # fit the FINAL model on all available data and save it
        final_clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
        final_clf.fit(X, labels_for_algo)

        importances = sorted(zip(FEATURE_NAMES, final_clf.feature_importances_), key=lambda x: -x[1])
        print("Top features:")
        for name, imp in importances[:5]:
            print(f"  {name}: {imp:.3f}")

        model_path = os.path.join(out_dir, f"rf_{algo_name}_sma.joblib")
        joblib.dump({
            "model": final_clf,
            "feature_names": FEATURE_NAMES,
            "classes": list(final_clf.classes_),
            "sma_params": {"N": N, "T": T, "d": 2},
            "algo": algo_name,
        }, model_path)
        print(f"Saved trained model -> {model_path}")

        summary[algo_name] = {
            "accuracy_mean": float(fold_accs.mean()),
            "accuracy_std": float(fold_accs.std()),
            "macro_f1": float(f1_macro),
            "n_folds": folds,
            "n_samples": len(crops),
        }

    summary_path = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison summary -> {summary_path}")
    if "standard" in summary and "enhanced" in summary:
        print("\n=== STANDARD vs ENHANCED (for Chapter 4) ===")
        print(f"Standard SMA -- accuracy: {summary['standard']['accuracy_mean']:.4f} "
              f"+/- {summary['standard']['accuracy_std']:.4f}, macro F1: {summary['standard']['macro_f1']:.4f}")
        print(f"Enhanced SMA -- accuracy: {summary['enhanced']['accuracy_mean']:.4f} "
              f"+/- {summary['enhanced']['accuracy_std']:.4f}, macro F1: {summary['enhanced']['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--sma_population", type=int, default=20, dest="N")
    parser.add_argument("--sma_iterations", type=int, default=30, dest="T")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap total crops for a quick test run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="./models", help="Where to save trained models + summary")
    parser.add_argument("--adaptive_k", action="store_true",
                         help="Use diversity-driven adaptive leader count for ESMA (extension beyond literal Algorithm 3.1)")
    args = parser.parse_args()
    run_pipeline(args.json, args.images_dir, args.N, args.T, args.folds, args.max_samples, args.seed,
                 args.out_dir, use_adaptive_k=args.adaptive_k)