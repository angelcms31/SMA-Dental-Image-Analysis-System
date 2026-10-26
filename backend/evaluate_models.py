"""
evaluate_models.py
--------------------
Per-class metrics for one set of YOLO weights on one test yaml produced by
build_improved_dataset.py (images on disk are ALREADY SMA/ESMA-preprocessed,
so nothing is preprocessed again here).

Two views, both measured, nothing copied from docstrings:
  * ultralytics val  (conf=0.001): P, R, mAP50, mAP50-95 per class
  * app operating point (conf=0.25, same NMS as predict_yolo.run_yolo_detection,
    IoU>=0.5 greedy match): TP/FP/FN -> precision, recall, F1 per class.
    This is the view that answers "does the app still miss findings?"

Usage:
    python evaluate_models.py --weights runs_yolo/train_standard_v2/weights/best.pt \
        --data ds_standard/test_dentex50.yaml --out eval/standard_old_dentex50.json
"""

import argparse
import glob
import json
import os

import numpy as np
import yaml

from predict_yolo import _iou, _non_max_suppress


def _read_gt(label_path, w, h):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if p:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                boxes.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
    return boxes


def operating_point(model, image_paths, names, imgsz, conf=0.25, nms_iou=0.35, match_iou=0.5):
    tp, fp, fn = ({n: 0 for n in names.values()} for _ in range(3))
    confs = {n: [] for n in names.values()}  # confidences the app would DISPLAY (kept after NMS)
    for res in model.predict(image_paths, conf=conf, imgsz=imgsz, verbose=False, stream=True):
        h, w = res.orig_shape
        preds = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append({"label": names[int(b.cls[0])], "bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": float(b.conf[0])})
        preds = _non_max_suppress(preds, iou_thresh=nms_iou)
        for p in preds:
            confs[p["label"]].append(p["confidence"])
        d, f = os.path.split(res.path)
        lbl = os.path.join(os.path.dirname(d), "labels", os.path.splitext(f)[0] + ".txt")
        gt = [(names[c], b) for c, b in _read_gt(lbl, w, h)]
        used = set()
        for p in sorted(preds, key=lambda r: -r["confidence"]):
            best, best_i = match_iou, None
            for i, (gl, gb) in enumerate(gt):
                if i not in used and gl == p["label"] and _iou(p["bbox"], gb) >= best:
                    best, best_i = _iou(p["bbox"], gb), i
            if best_i is None:
                fp[p["label"]] += 1
            else:
                used.add(best_i)
                tp[p["label"]] += 1
        for i, (gl, _) in enumerate(gt):
            if i not in used:
                fn[gl] += 1
    out = {}
    for n in names.values():
        p = tp[n] / (tp[n] + fp[n]) if tp[n] + fp[n] else 0.0
        r = tp[n] / (tp[n] + fn[n]) if tp[n] + fn[n] else 0.0
        q = np.percentile(confs[n], [10, 50, 90]).round(3).tolist() if confs[n] else None
        out[n] = dict(tp=tp[n], fp=fp[n], fn=fn[n], gt_boxes=tp[n] + fn[n], precision=p, recall=r,
                      f1=2 * p * r / (p + r) if p + r else 0.0, shown_conf_p10_p50_p90=q)
    return out


def main(weights, data, out_path, imgsz, device):
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names
    imgsz = imgsz or model.overrides.get("imgsz", 640)  # app predicts at the checkpoint's own size
    cfg = yaml.safe_load(open(data))
    img_dir = os.path.join(cfg["path"], cfg["val"])
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert images, f"no images in {img_dir}"

    m = model.val(data=data, imgsz=imgsz, conf=0.001, plots=False, verbose=False, device=device)
    per_class = {}
    for k, ci in enumerate(m.box.ap_class_index):
        per_class[names[int(ci)]] = dict(P=float(m.box.p[k]), R=float(m.box.r[k]),
                                         mAP50=float(m.box.ap50[k]), mAP50_95=float(m.box.all_ap[k].mean()))
    result = dict(weights=weights, data=data, n_images=len(images), imgsz=imgsz,
                  val=dict(per_class=per_class, mAP50=float(m.box.map50), mAP50_95=float(m.box.map)),
                  app_operating_point_conf0_25=operating_point(model, images, names, imgsz))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="default: the checkpoint's own training size")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    main(a.weights, a.data, a.out, a.imgsz, a.device)
