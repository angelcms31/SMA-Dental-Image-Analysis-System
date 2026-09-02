"""
merge_mendeley_into_dentex_yolo.py
-------------------------------------
Adds Mendeley Dental OPG X-ray images (only the Caries and Impacted
Teeth findings -- the two classes visually confirmed to match the
guessed class-ID mapping) into an EXISTING DENTEX-based YOLO dataset
folder (the ones produced by dentex_to_yolo.py), so you can retrain
with the combined data.

Confirmed Mendeley -> DENTEX class ID mapping (from visual inspection):
    Mendeley class 0 (Caries)         -> DENTEX class 1 (Caries)
    Mendeley class 2 (Impacted Teeth) -> DENTEX class 0 (Impacted)
Any other Mendeley class (1, 3, 4, 5) is DROPPED from merged labels --
those don't have a confirmed, clean DENTEX equivalent.

Images with NO class-0 or class-2 annotations are skipped entirely
(nothing useful to add).

If --sma_algorithm is given, each qualifying image is preprocessed the
SAME way dentex_to_yolo.py --use_esma does (crop_borders=False, so
the bounding boxes -- computed from the ORIGINAL image -- stay
aligned) before being copied into the target dataset. This MUST match
whichever algorithm preprocessed the target dataset you're merging
into, or the model will be trained on an inconsistent mix.

Usage:
    python merge_mendeley_into_dentex_yolo.py \
        --mendeley_dir "C:\\Users\\Analyn\\Downloads\\Dental OPG XRAY Dataset" \
        --target_dataset_dir ./yolo_dataset_standard \
        --sma_algorithm standard \
        --val_fraction 0.15
"""

import argparse
import glob
import os
import random
import shutil

import cv2

from sma_preprocess import sma_preprocess

# confirmed via visual inspection (see conversation) -- do not change
# without re-verifying against sample images
MENDELEY_TO_DENTEX = {
    "0": "1",  # Mendeley Caries -> DENTEX Caries
    "2": "0",  # Mendeley Impacted -> DENTEX Impacted
}


def find_pairs(mendeley_dir):
    txt_files = glob.glob(os.path.join(mendeley_dir, "**", "*.txt"), recursive=True)
    pairs = []
    for txt_path in txt_files:
        base = os.path.splitext(txt_path)[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            img_path = base + ext
            if os.path.exists(img_path):
                pairs.append((img_path, txt_path))
                break
    return pairs


def remap_labels(label_path):
    """Returns remapped YOLO lines (only Caries/Impacted, others dropped),
    or None if the image has no qualifying annotations."""
    kept_lines = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = parts[0]
            if class_id in MENDELEY_TO_DENTEX:
                new_class_id = MENDELEY_TO_DENTEX[class_id]
                kept_lines.append(" ".join([new_class_id] + parts[1:]))
    return kept_lines if kept_lines else None


def main(mendeley_dir, target_dataset_dir, sma_algorithm, val_fraction, seed,
         esma_d, esma_N, esma_T):
    pairs = find_pairs(mendeley_dir)
    print(f"Found {len(pairs)} Mendeley image+label pairs.")

    qualifying = []
    for image_path, label_path in pairs:
        remapped = remap_labels(label_path)
        if remapped:
            qualifying.append((image_path, remapped))

    print(f"{len(qualifying)} images have at least one Caries/Impacted annotation "
          f"and will be added.")
    if not qualifying:
        print("Nothing to merge -- check the mendeley_dir path.")
        return

    rng = random.Random(seed)
    rng.shuffle(qualifying)
    n_val = max(1, int(len(qualifying) * val_fraction))

    for split in ("train", "val"):
        os.makedirs(os.path.join(target_dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dataset_dir, "labels", split), exist_ok=True)

    n_written = 0
    for i, (image_path, lines) in enumerate(qualifying):
        split = "val" if i < n_val else "train"
        out_name = f"mendeley_{i}.png"
        dst_img = os.path.join(target_dataset_dir, "images", split, out_name)
        dst_label = os.path.join(target_dataset_dir, "labels", split,
                                  os.path.splitext(out_name)[0] + ".txt")

        if sma_algorithm:
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            processed = sma_preprocess(
                img_bgr, algorithm=sma_algorithm, d=esma_d, N=esma_N, T=esma_T,
                seed=seed, crop_borders=False
            )
            cv2.imwrite(dst_img, processed)
        else:
            shutil.copy2(image_path, dst_img)

        with open(dst_label, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

        if sma_algorithm and n_written % 25 == 0:
            print(f"  ...{n_written}/{len(qualifying)} images processed", flush=True)

    print(f"\nDone. Added {n_written} images to '{target_dataset_dir}' "
          f"({len(qualifying) - n_val} train / {n_val} val).")
    print("IMPORTANT: dentex_to_yolo.py's data.yaml is still valid (same "
          "4 classes, same paths) -- no need to regenerate it. You can retrain "
          "directly with train_yolo_teeth.py pointed at this dataset's data.yaml.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mendeley_dir", required=True)
    parser.add_argument("--target_dataset_dir", required=True,
                         help="e.g. ./yolo_dataset_standard or ./yolo_dataset_enhanced -- "
                              "must already exist (created by dentex_to_yolo.py)")
    parser.add_argument("--sma_algorithm", choices=["standard", "enhanced", "none"], default="none",
                         help="Must match how the target dataset was preprocessed. Use 'none' "
                              "if the target dataset was created WITHOUT --use_esma.")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esma_d", type=int, default=4)
    parser.add_argument("--esma_N", type=int, default=30)
    parser.add_argument("--esma_T", type=int, default=150)
    args = parser.parse_args()
    algo = None if args.sma_algorithm == "none" else args.sma_algorithm
    main(args.mendeley_dir, args.target_dataset_dir, algo, args.val_fraction, args.seed,
         args.esma_d, args.esma_N, args.esma_T)