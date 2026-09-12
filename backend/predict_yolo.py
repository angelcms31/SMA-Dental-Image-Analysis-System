"""
predict_yolo.py
-----------------
Runs the trained YOLO tooth detector on a full panoramic OPG image and
draws Q1-Q4 quadrant-colored boxes + diagnosis labels -- this is the
real-detector replacement for overlay.py's heuristic
generate_annotated_overlay(), producing the accurate, per-tooth
result your adviser is asking for.

Usage (standalone test):
    python predict_yolo.py --weights best.pt --image some_xray.png --out result.png

To wire this into main.py as a new endpoint, see integrate_into_main()
at the bottom of this file for the exact code to add.
"""

import argparse

import cv2
import numpy as np

from sma_preprocess import sma_preprocess

# Colored by DIAGNOSIS TYPE (not quadrant) so a finding's severity/category
# is identifiable at a glance -- kept consistent with the frontend legend
# (OpgAnalyzer.tsx DIAGNOSIS_ACCENT). BGR order for OpenCV.
DIAGNOSIS_COLORS = {
    "Caries": (61, 163, 232),             # amber
    "Deep Caries": (76, 96, 232),         # coral/red -- more severe than Caries
    "Impacted": (191, 95, 139),           # purple -- structural, not decay-related
    "Periapical Lesion": (232, 143, 76),  # blue
}
DEFAULT_COLOR = (180, 180, 180)  # gray fallback for any unrecognized label


def _quadrant_for(cx, cy, x_mid, y_mid):
    if cx < x_mid and cy < y_mid:
        return "Q1"
    if cx >= x_mid and cy < y_mid:
        return "Q2"
    if cx >= x_mid and cy >= y_mid:
        return "Q3"
    return "Q4"


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def _non_max_suppress(regions, iou_thresh=0.35):
    """Keeps the higher-confidence box when two detections overlap
    heavily -- prevents the label-clutter seen when several boxes land
    on the same tooth (e.g. from adjacent/overlapping ground-truth
    boxes in training)."""
    regions = sorted(regions, key=lambda r: r["confidence"], reverse=True)
    kept = []
    for r in regions:
        if any(_iou(r["bbox"], k["bbox"]) > iou_thresh for k in kept):
            continue
        kept.append(r)
    return kept


def run_yolo_detection(image_bgr, weights_path, conf_thresh=0.25, iou_thresh=0.35,
                        use_esma=False, sma_algorithm="enhanced", esma_kwargs=None):
    """
    Returns (annotated_image_bgr, detected_regions_list). Each region:
    {quadrant, label, bbox, confidence}. Unlike overlay.py's heuristic
    version, EVERY box here comes from a real trained detector, so
    there's no separate "WHERE" vs "WHAT" step -- the model does both
    at once. NMS is applied on top of YOLO's own internal NMS as an
    extra safeguard against label clutter on real images.

    use_esma=True runs the SAME SMA/ESMA preprocessing used at training
    time (see sma_preprocess.py) before detection -- sma_algorithm MUST
    match whichever algorithm preprocessed the dataset this model was
    trained on ("standard" or "enhanced"), or accuracy will be badly
    off (this is the single most common way to break a model that used
    custom preprocessing: forgetting to apply the SAME preprocessing at
    inference too).
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)

    detect_input = image_bgr
    if use_esma:
        detect_input = sma_preprocess(
            image_bgr, algorithm=sma_algorithm, crop_borders=False, **(esma_kwargs or {})
        )

    h, w = image_bgr.shape[:2]
    x_mid, y_mid = w / 2.0, h / 2.0

    results = model.predict(detect_input, conf=conf_thresh, verbose=False)[0]
    class_names = results.names

    candidates = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = class_names[cls_id]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        quadrant = _quadrant_for(cx, cy, x_mid, y_mid)
        x, y, bw, bh = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        candidates.append({
            "quadrant": quadrant,
            "label": label,
            "bbox": [x, y, bw, bh],
            "confidence": round(confidence, 4),
        })

    detected = _non_max_suppress(candidates, iou_thresh=iou_thresh)

    output = image_bgr.copy()
    overlay = output.copy()
    for region in detected:
        x, y, bw, bh = region["bbox"]
        color = DIAGNOSIS_COLORS.get(region["label"], DEFAULT_COLOR)
        overlay[y:y + bh, x:x + bw] = color
        cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

    blended = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)
    for region in detected:
        x, y, bw, bh = region["bbox"]
        color = DIAGNOSIS_COLORS.get(region["label"], DEFAULT_COLOR)
        text = f'Q={region["quadrant"][1]} D={region["label"]} ({region["confidence"]*100:.0f}%)'
        ty = max(y - 8, 15)
        cv2.putText(blended, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return blended, detected


def integrate_into_main():
    """
    Not a real function -- just documents the FastAPI endpoint to add
    to main.py once you have trained weights (best.pt). Copy this
    into main.py:

    from predict_yolo import run_yolo_detection

    YOLO_WEIGHTS_PATH = "./models/best.pt"  # wherever you put best.pt

    @app.post("/analyze/detect-teeth/")
    async def analyze_detect_teeth(file: UploadFile = File(...), conf: float = Form(0.25)):
        '''
        Real per-tooth detection + diagnosis via a trained YOLO model
        (supplementary component, separate from SMA/ESMA -- see
        dentex_to_yolo.py / train_yolo_teeth.py docstrings). Replaces
        the heuristic /analyze/classify-full/ endpoint's box placement
        with actual trained detection.
        '''
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode image.")

            annotated, detected_regions = run_yolo_detection(image_bgr, YOLO_WEIGHTS_PATH)
            _, buffer = cv2.imencode(".png", annotated)
            annotated_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

            return {
                "status": "success",
                "annotated_image": annotated_b64,
                "detected_regions": detected_regions,
                "disclaimer": DISCLAIMER + " Detections come from a YOLO model "
                              "trained on DENTEX -- validate its accuracy on your "
                              "own held-out images before treating this as reliable.",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="./yolo_result.png")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--use_esma", action="store_true",
                         help="Apply SMA/ESMA preprocessing before detection -- REQUIRED if the "
                              "model was trained with --use_esma in dentex_to_yolo.py")
    parser.add_argument("--sma_algorithm", choices=["standard", "enhanced"], default="enhanced",
                         help="Must match whichever algorithm preprocessed the training data")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    annotated, regions = run_yolo_detection(
        img, args.weights, args.conf, args.iou,
        use_esma=args.use_esma, sma_algorithm=args.sma_algorithm
    )
    cv2.imwrite(args.out, annotated)
    print(f"Detected {len(regions)} regions. Saved -> {args.out}")
    for r in regions:
        print(r)