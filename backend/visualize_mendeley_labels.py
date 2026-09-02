"""
visualize_mendeley_labels.py
------------------------------
Draws YOLO-format bounding boxes (with their raw class ID number) on
a handful of sample images from the Mendeley Dental OPG dataset, so
you can visually confirm the guessed class-ID mapping from
count_mendeley_classes.py before trusting it for anything real.

What to look for once you view the output images:
  - Boxes labeled "0" (guessed = Caries) should look like small dark
    spots ON individual teeth crowns
  - Boxes labeled "2" (guessed = Impacted Teeth) should look like a
    WHOLE tooth, often at an odd angle, often near the back/jaw edge
    (wisdom-tooth area), not fully visible/erupted

If what you see doesn't match the guessed label, the count-based
mapping was wrong for that class -- don't use it without further
checking.

Usage:
    python visualize_mendeley_labels.py \
        --dataset_dir "C:\\Users\\Analyn\\Downloads\\Dental OPG XRAY Dataset" \
        --n_samples 6
"""

import argparse
import glob
import os
import random

import cv2

COLORS = [(0, 200, 0), (0, 0, 220), (200, 130, 0), (0, 210, 210), (180, 0, 180), (100, 100, 255)]


def find_pairs(dataset_dir):
    """Finds (image_path, label_path) pairs by matching filenames."""
    txt_files = glob.glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True)
    pairs = []
    for txt_path in txt_files:
        base = os.path.splitext(txt_path)[0]
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            img_path = base + ext
            if os.path.exists(img_path):
                pairs.append((img_path, txt_path))
                break
    return pairs


def draw_boxes(image_path, label_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(class_id), (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, str(class_id), (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return True


def main(dataset_dir, n_samples, out_dir, seed):
    pairs = find_pairs(dataset_dir)
    print(f"Found {len(pairs)} image+label pairs.")
    if not pairs:
        print("No matching image/label pairs found -- check the dataset_dir path.")
        return

    rng = random.Random(seed)
    rng.shuffle(pairs)
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for image_path, label_path in pairs:
        if n_written >= n_samples:
            break
        out_path = os.path.join(out_dir, f"sample_{n_written}.png")
        if draw_boxes(image_path, label_path, out_path):
            print(f"  wrote {out_path}  (from {os.path.basename(image_path)})")
            n_written += 1

    print(f"\nDone. {n_written} sample images with boxes+class IDs saved to '{out_dir}'.")
    print("Open them and check: do class-0 boxes look like caries spots, "
          "and class-2 boxes look like whole impacted teeth?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--out_dir", default="./mendeley_label_check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset_dir, args.n_samples, args.out_dir, args.seed)