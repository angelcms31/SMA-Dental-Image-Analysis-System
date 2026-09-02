"""
dentex_to_yolo.py
-------------------
Converts DENTEX's COCO-style quadrant-enumeration-diagnosis annotations
into YOLO training format, so you can train a REAL object detector
(YOLOv8, via the ultralytics library) on the actual ground-truth tooth
boxes + diagnosis labels -- this is what actually solves "boxes aren't
on teeth" and "gaps get flagged as findings", because the detector is
trained with real supervision on exactly this task, instead of the
heuristic (brightness/contrast) detection the overlay currently uses.

IMPORTANT SCOPE NOTE: This YOLO detector is a SEPARATE, SUPPLEMENTARY
component from SMA/ESMA (a completely different algorithm family --
a convolutional neural network trained via gradient descent, not a
metaheuristic threshold optimizer). Document it as such in your
methodology if you use it: it replaces the overlay's heuristic tooth
LOCALIZATION step, it does not change or replace SMA/ESMA's actual
role (Kapur's entropy-based threshold optimization), which remains
your core contribution.

Output layout:
    yolo_dataset/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt   (one line per box: class_id cx cy w h, normalized)
      labels/val/*.txt
      data.yaml            (tells YOLO where everything is + class names)

Usage:
    python dentex_to_yolo.py \
        --json train_quadrant_enumeration_disease.json \
        --images_dir ./xrays \
        --out_dir ./yolo_dataset \
        --val_fraction 0.15
"""

import argparse
import json
import os
import random
import shutil
import time

import cv2

from sma_preprocess import sma_preprocess


def convert(json_path, images_dir, out_dir, val_fraction, seed, use_esma=False,
            sma_algorithm="enhanced", esma_d=4, esma_N=30, esma_T=150):
    with open(json_path) as f:
        data = json.load(f)

    # class order MUST match categories_3's id ordering exactly, since
    # YOLO just uses integer class ids -- we preserve DENTEX's own ids
    id_to_name = {c["id"]: c["name"] for c in data["categories_3"]}
    class_names = [id_to_name[i] for i in sorted(id_to_name)]
    print("Class order (id -> name):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # group annotations by image so each image gets ONE label file with
    # all its boxes (an X-ray usually has multiple annotated teeth)
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(anns_by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * val_fraction))
    val_ids = set(image_ids[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    n_written, n_skipped = 0, 0
    t_start = time.perf_counter()
    if use_esma:
        print(f"SMA preprocessing enabled (algorithm={sma_algorithm}, d={esma_d}, N={esma_N}, T={esma_T}) -- "
              f"this will take a while (roughly 1-2s per image).")
    for image_id in image_ids:
        filename = id_to_file.get(image_id)
        if filename is None:
            n_skipped += 1
            continue
        src_path = os.path.join(images_dir, filename)
        if not os.path.exists(src_path):
            n_skipped += 1
            continue

        img_w, img_h = id_to_size.get(image_id, (None, None))
        if img_w is None:
            # fall back to reading the image if size wasn't in the JSON
            img = cv2.imread(src_path)
            if img is None:
                n_skipped += 1
                continue
            img_h, img_w = img.shape[:2]

        split = "val" if image_id in val_ids else "train"

        lines = []
        for ann in anns_by_image[image_id]:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            # clip to [0,1] -- guards against any off-by-a-pixel boxes
            cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
            class_id = ann["category_id_3"]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_img = os.path.join(out_dir, "images", split, filename)
        dst_label = os.path.join(out_dir, "labels", split, os.path.splitext(filename)[0] + ".txt")

        if use_esma:
            img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                n_skipped += 1
                continue
            # crop_borders=False: preserves the original framing/size so
            # the bboxes above (computed from the ORIGINAL image
            # dimensions) still line up correctly with this output
            processed = sma_preprocess(
                img_bgr, algorithm=sma_algorithm, d=esma_d, N=esma_N, T=esma_T,
                seed=seed, crop_borders=False
            )
            cv2.imwrite(dst_img, processed)
        else:
            shutil.copy2(src_path, dst_img)

        with open(dst_label, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

        if use_esma and n_written % 25 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  ...{n_written} images {sma_algorithm}-SMA-preprocessed ({elapsed:.1f}s elapsed)", flush=True)

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"\nDone. {n_written} images written ({len(image_ids) - n_val} train / {n_val} val), "
          f"{n_skipped} skipped (missing file).")
    print(f"YOLO config written -> {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", default="./yolo_dataset")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_esma", action="store_true",
                         help="Preprocess every image with SMA/ESMA segmentation before saving "
                              "(per adviser's integration requirement) -- much slower, since "
                              "the algorithm runs once per image during conversion.")
    parser.add_argument("--sma_algorithm", choices=["standard", "enhanced"], default="enhanced",
                         help="Which algorithm does the preprocessing when --use_esma is set. "
                              "Run this script TWICE (once with each) to get comparable "
                              "'YOLO+Standard' vs 'YOLO+ESMA' datasets -- use different --out_dir "
                              "for each so you can train and compare both.")
    parser.add_argument("--esma_d", type=int, default=4)
    parser.add_argument("--esma_N", type=int, default=30)
    parser.add_argument("--esma_T", type=int, default=150)
    args = parser.parse_args()
    convert(args.json, args.images_dir, args.out_dir, args.val_fraction, args.seed,
            use_esma=args.use_esma, sma_algorithm=args.sma_algorithm,
            esma_d=args.esma_d, esma_N=args.esma_N, esma_T=args.esma_T)