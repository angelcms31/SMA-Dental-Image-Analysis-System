"""
prepare_dentex_crops.py
------------------------
Reads a DENTEX quadrant-enumeration-diagnosis JSON file (matches the
structure of validation_triple.json: images / annotations /
categories_1 / categories_2 / categories_3) and crops out every
annotated abnormal tooth into a folder named after its diagnosis --
ready to zip and upload to Teachable Machine (one class per folder).

Usage:
    python prepare_dentex_crops.py \
        --json train_quadrant_enumeration_disease.json \
        --images_dir ./train_images \
        --out_dir ./crops

Folder layout produced:
    crops/
      Impacted/
        val_15_ann1.png
        ...
      Caries/
      Periapical_Lesion/
      Deep_Caries/

If you want SMA/ESMA-segmented crops instead of raw crops (recommended,
since that keeps the pipeline tied to your actual algorithm), run your
standard_sma()/enhanced_sma() on each crop after this script produces
the raw ones, then feed THOSE into Teachable Machine instead.
"""

import argparse
import json
import os

import cv2
import numpy as np


def main(json_path, images_dir, out_dir, use_segmentation_mask, padding):
    with open(json_path) as f:
        data = json.load(f)

    id_to_label = {c["id"]: c["name"] for c in data["categories_3"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    for label in id_to_label.values():
        safe_label = label.replace(" ", "_")
        os.makedirs(os.path.join(out_dir, safe_label), exist_ok=True)

    counts = {label: 0 for label in id_to_label.values()}
    skipped = 0

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        filename = id_to_file.get(image_id)
        if filename is None:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        h, w = image.shape[:2]
        x, y, bw, bh = ann["bbox"]
        x0 = max(0, int(x - padding))
        y0 = max(0, int(y - padding))
        x1 = min(w, int(x + bw + padding))
        y1 = min(h, int(y + bh + padding))
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue

        crop = image[y0:y1, x0:x1].copy()

        if use_segmentation_mask and ann.get("segmentation"):
            # zero out everything outside the polygon so the classifier
            # only sees the annotated tooth/lesion, not neighbors
            poly = np.array(ann["segmentation"][0], dtype=np.int32).reshape(-1, 2)
            poly_shifted = poly - [x0, y0]
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly_shifted], 255)
            crop = cv2.bitwise_and(crop, crop, mask=mask)

        label = id_to_label[ann["category_id_3"]]
        safe_label = label.replace(" ", "_")
        counts[label] += 1
        out_name = f"{os.path.splitext(filename)[0]}_ann{ann['id']}.png"
        cv2.imwrite(os.path.join(out_dir, safe_label, out_name), crop)

    print("Done. Crops per diagnosis:")
    for label, n in counts.items():
        print(f"  {label}: {n}")
    if skipped:
        print(f"Skipped {skipped} annotations (missing image or bad bbox).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to the DENTEX diagnosis JSON file")
    parser.add_argument("--images_dir", required=True, help="Folder containing the DENTEX images")
    parser.add_argument("--out_dir", default="./crops", help="Where to write the per-class crop folders")
    parser.add_argument("--use_segmentation_mask", action="store_true",
                         help="Mask out everything outside the polygon (tighter crop, no neighboring tooth)")
    parser.add_argument("--padding", type=int, default=10, help="Extra pixels around each bbox")
    args = parser.parse_args()
    main(args.json, args.images_dir, args.out_dir, args.use_segmentation_mask, args.padding)