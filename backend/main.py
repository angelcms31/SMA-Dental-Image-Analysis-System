"""
main.py
-------
FastAPI backend for the thesis system. Replaces the previous placeholder
(hardcoded threshold range + fake "confidence %" derived from x+y+area).

Endpoints:
  POST /analyze/standard   -> runs the ORIGINAL SMA (Li et al., 2020)
  POST /analyze/enhanced   -> runs the proposed ESMA
  POST /analyze/compare    -> runs BOTH on the same image with the same
                               params/seed and returns a side-by-side
                               result -- this is what feeds your Chapter 4
                               comparative-analysis table/figures directly.
  POST /analyze/classify   -> SUPPLEMENTARY: predicts a diagnosis label
                               for a single cropped tooth image, using the
                               trained DENTEX Random Forest models. See the
                               warning in that endpoint's docstring before
                               wiring this into any patient-facing screen.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import base64
import os
import time

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sma_algorithms import (
    apply_thresholds,
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    kapurs_entropy_fitness,
    standard_sma,
)
from metrics import compute_psnr, compute_ssim
from overlay import DISCLAIMER, generate_annotated_overlay
from train_dentex_classifier import extract_features
from predict_yolo import run_yolo_detection

YOLO_WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", "./runs_yolo/train/weights/best.pt")

MODELS_DIR = os.environ.get("DENTEX_MODELS_DIR", "./models")

app = FastAPI(title="ESMA Dental OPG Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode image -- check the file is a valid image.")
    image = autocrop_black_borders(image)
    return image


def _encode_image(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


def _run_and_package(algo_name, algo_fn, image, prob, d, N, T, seed, **extra_kwargs):
    result = algo_fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **extra_kwargs)
    segmented = apply_thresholds(image, result["thresholds"])

    psnr = compute_psnr(image, segmented)
    ssim = compute_ssim(image, segmented)

    annotated, detected_regions, quadrant_summary = generate_annotated_overlay(
        image, result["thresholds"]
    )

    return {
        "algorithm": algo_name,
        "thresholds": result["thresholds"],
        "kapur_entropy_fitness": round(result["fitness"], 6),
        "psnr": round(psnr, 4) if psnr != float("inf") else None,
        "ssim": round(ssim, 6),
        "runtime_sec": round(result["runtime_sec"], 4),
        "convergence_curve": [round(v, 6) for v in result["convergence"]],
        "segmented_image": _encode_image(segmented),
        "annotated_image": _encode_image(annotated),
        "detected_regions": detected_regions,
        "quadrant_summary": quadrant_summary,
        "disclaimer": DISCLAIMER,
    }


@app.post("/analyze/standard/")
async def analyze_standard(
    file: UploadFile = File(...),
    d: int = Form(4),          # number of threshold levels
    N: int = Form(30),         # population size
    T: int = Form(100),        # max iterations
    seed: int = Form(None),    # set a fixed seed for reproducible comparisons
):
    try:
        contents = await file.read()
        image = _decode_image(contents)
        prob = compute_histogram_prob(image)
        result = _run_and_package("Standard SMA", standard_sma, image, prob, d, N, T, seed)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze/enhanced/")
async def analyze_enhanced(
    file: UploadFile = File(...),
    d: int = Form(4),
    N: int = Form(30),
    T: int = Form(100),
    seed: int = Form(None),
    k: int = Form(3),
    alpha: float = Form(0.10),
    beta: float = Form(0.10),
    gamma: float = Form(0.10),
    delta: float = Form(0.10),
    h: int = Form(5),
):
    try:
        contents = await file.read()
        image = _decode_image(contents)
        prob = compute_histogram_prob(image)
        result = _run_and_package(
            "Enhanced SMA (ESMA)", enhanced_sma, image, prob, d, N, T, seed,
            k=k, alpha=alpha, beta=beta, gamma=gamma, delta=delta, h=h,
        )
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze/compare/")
async def analyze_compare(
    file: UploadFile = File(...),
    d: int = Form(4),
    N: int = Form(30),
    T: int = Form(100),
    seed: int = Form(42),   # same seed for both -> fair side-by-side comparison
):
    """
    Runs Standard SMA and ESMA on the SAME image with the SAME
    population/iteration/seed settings. This is the endpoint your
    Chapter 4 comparative-analysis table should pull from, since it
    guarantees both algorithms saw identical conditions.
    """
    try:
        contents = await file.read()
        image = _decode_image(contents)
        prob = compute_histogram_prob(image)

        standard_result = _run_and_package(
            "Standard SMA", standard_sma, image, prob, d, N, T, seed
        )
        enhanced_result = _run_and_package(
            "Enhanced SMA (ESMA)", enhanced_sma, image, prob, d, N, T, seed
        )

        improvement = {
            "entropy_gain": round(
                enhanced_result["kapur_entropy_fitness"] - standard_result["kapur_entropy_fitness"], 6
            ),
            "psnr_gain": (
                round(enhanced_result["psnr"] - standard_result["psnr"], 4)
                if enhanced_result["psnr"] is not None and standard_result["psnr"] is not None
                else None
            ),
            "ssim_gain": round(enhanced_result["ssim"] - standard_result["ssim"], 6),
            "runtime_diff_sec": round(
                enhanced_result["runtime_sec"] - standard_result["runtime_sec"], 4
            ),
        }

        return {
            "status": "success",
            "standard": standard_result,
            "enhanced": enhanced_result,
            "improvement": improvement,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Cache loaded models in memory so we don't hit disk on every request
_model_cache = {}


def _load_classifier(algorithm: str):
    if algorithm not in ("standard", "enhanced"):
        raise ValueError("algorithm must be 'standard' or 'enhanced'")
    if algorithm in _model_cache:
        return _model_cache[algorithm]

    model_path = os.path.join(MODELS_DIR, f"rf_{algorithm}_sma.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. Run "
            f"train_dentex_classifier.py first to produce it."
        )
    bundle = joblib.load(model_path)
    _model_cache[algorithm] = bundle
    return bundle


@app.post("/analyze/classify/")
async def analyze_classify(
    file: UploadFile = File(...),
    algorithm: str = Form("enhanced"),   # "standard" or "enhanced"
    N: int = Form(20),
    T: int = Form(40),
    seed: int = Form(42),
):
    """
    SUPPLEMENTARY / EXPLORATORY ENDPOINT -- this is the DENTEX-trained
    Random Forest diagnosis classifier discussed as a supplementary
    analysis in Chapter 4, NOT the core SMA/ESMA segmentation
    comparison. Do not present this endpoint's output as a validated
    diagnostic tool: cross-validated accuracy was ~65% (both variants),
    with especially weak performance on minority classes (Deep Caries,
    Periapical Lesion) due to class imbalance in the training data.

    Expects a single already-cropped tooth image (not a full panoramic
    OPG) -- crop it the same way train_dentex_classifier.py did (a
    tight bounding box around one tooth) before uploading here.
    """
    try:
        bundle = _load_classifier(algorithm)
    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "message": str(e)}

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if crop is None:
            raise ValueError("Could not decode image -- check the file is a valid image.")

        algo_fn = standard_sma if algorithm == "standard" else enhanced_sma
        algo_kwargs = {"adaptive_k": True} if algorithm == "enhanced" else {}
        features = extract_features(crop, algo_fn, N, T, seed=seed, algo_kwargs=algo_kwargs)

        clf = bundle["model"]
        pred = clf.predict([features])[0]
        proba = clf.predict_proba([features])[0]
        proba_by_class = {
            str(cls): round(float(p), 4) for cls, p in zip(clf.classes_, proba)
        }

        return {
            "status": "success",
            "algorithm": algorithm,
            "predicted_label": str(pred),
            "probabilities": proba_by_class,
            "sma_params": bundle.get("sma_params"),
            "disclaimer": (
                DISCLAIMER + " This specific classifier is a supplementary "
                "research finding (~65% cross-validated accuracy, weak on "
                "minority classes) -- not a validated diagnostic tool."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze/classify-full/")
async def analyze_classify_full(
    file: UploadFile = File(...),
    algorithm: str = Form("enhanced"),   # "standard" or "enhanced"
    d: int = Form(4),          # threshold levels for the FULL-image SMA/ESMA run
    N: int = Form(30),
    T: int = Form(150),
    seed: int = Form(42),
    clf_N: int = Form(20),     # SMA/ESMA settings used for the per-region classifier features
    clf_T: int = Form(40),
):
    """
    SUPPLEMENTARY / EXPLORATORY ENDPOINT -- full-panoramic version of
    /analyze/classify/. Runs SMA or ESMA on the WHOLE uploaded OPG image
    to find candidate abnormal regions (same detection pipeline as
    /analyze/standard/ and /analyze/enhanced/), then classifies EACH
    detected region with the trained DENTEX Random Forest model instead
    of the darkness-rank heuristic -- this is what lets you upload a
    full panoramic image (like DENTEX's own ground-truth figures) and
    get quadrant + diagnosis boxes back, rather than having to crop
    each tooth yourself.

    WHAT THIS DOES NOT DO: it does not number individual teeth (no
    "N: 8" style FDI position) -- that requires detecting and counting
    EVERY tooth (including normal ones) to know each one's position
    within its quadrant, which this system does not do. Only quadrant
    (Q1-Q4) and predicted diagnosis are returned per region.

    Same accuracy caveats as /analyze/classify/ apply: ~65%
    cross-validated accuracy, weak on minority classes. This is a
    research prototype, not a diagnostic tool -- see 'disclaimer' in
    the response.
    """
    try:
        clf_bundle = _load_classifier(algorithm)
    except (ValueError, FileNotFoundError) as e:
        return {"status": "error", "message": str(e)}

    try:
        contents = await file.read()
        image = _decode_image(contents)
        prob = compute_histogram_prob(image)

        sma_fn = standard_sma if algorithm == "standard" else enhanced_sma
        sma_kwargs = {"adaptive_k": True} if algorithm == "enhanced" else {}
        sma_result = sma_fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, **sma_kwargs)

        clf_algo_fn = standard_sma if algorithm == "standard" else enhanced_sma
        clf_algo_kwargs = {"adaptive_k": True} if algorithm == "enhanced" else {}
        clf = clf_bundle["model"]

        def label_fn(crop_patch):
            feats = extract_features(
                crop_patch, clf_algo_fn, clf_N, clf_T, seed=seed, algo_kwargs=clf_algo_kwargs
            )
            pred = clf.predict([feats])[0]
            proba = clf.predict_proba([feats])[0]
            confidence = float(proba[list(clf.classes_).index(pred)])
            return str(pred), confidence

        annotated, detected_regions, quadrant_summary = generate_annotated_overlay(
            image, sma_result["thresholds"], label_fn=label_fn
        )

        return {
            "status": "success",
            "algorithm": algorithm,
            "annotated_image": _encode_image(annotated),
            "detected_regions": detected_regions,
            "quadrant_summary": quadrant_summary,
            "sma_params": {"d": d, "N": N, "T": T},
            "classifier_sma_params": clf_bundle.get("sma_params"),
            "disclaimer": (
                DISCLAIMER + " Diagnosis labels come from a supplementary "
                "research classifier (~65% cross-validated accuracy, weak "
                "on minority classes) -- not a validated diagnostic tool. "
                "Tooth position numbering (FDI) is not provided -- only "
                "quadrant and predicted diagnosis."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


_yolo_model_cache = {}


@app.post("/analyze/detect-teeth/")
async def analyze_detect_teeth(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.35),
    use_esma: bool = Form(False),
    sma_algorithm: str = Form("enhanced"),
    weights_path: str = Form(None),
):
    """
    Real per-tooth detection + diagnosis via a YOLO model trained on
    DENTEX ground-truth boxes (see dentex_to_yolo.py / train_yolo_teeth.py).
    This is a SUPPLEMENTARY component, separate from SMA/ESMA -- it
    replaces /analyze/classify-full/'s heuristic box placement with
    actual trained object detection, which is what makes boxes land on
    real teeth (and not gaps/sinus/bone) reliably.

    use_esma=True runs ESMA preprocessing (esma_preprocess.py) before
    detection, per the adviser's integration requirement -- this MUST
    match how the currently-loaded weights were trained (if you trained
    with dentex_to_yolo.py --use_esma, pass use_esma=True here too, or
    accuracy will be badly degraded; if you trained without it, leave
    this False).

    Validation performance from training (40 epochs, CPU, YOLOv8n):
    mAP50 = 0.529 overall (Impacted 0.946, Deep Caries 0.541,
    Caries 0.394, Periapical Lesion 0.224) -- this was WITHOUT ESMA
    preprocessing; if you retrain with --use_esma, update this
    docstring with the new numbers. Weakest on the classes with the
    least training data -- same class-imbalance limitation as the
    Random Forest classifier. Not a validated diagnostic tool.

    weights_path lets you point at a specific .pt file per request --
    useful since you'll likely have TWO trained YOLO models to compare
    (one trained on Standard-SMA-preprocessed data, one on
    ESMA-preprocessed data). Falls back to YOLO_WEIGHTS_PATH if omitted.
    """
    resolved_weights = weights_path or YOLO_WEIGHTS_PATH
    if not os.path.exists(resolved_weights):
        return {
            "status": "error",
            "message": f"YOLO weights not found at '{resolved_weights}'. "
                        f"Train it first with train_yolo_teeth.py, or pass "
                        f"the correct weights_path / set YOLO_WEIGHTS_PATH.",
        }

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image -- check the file is a valid image.")

        annotated, detected_regions = run_yolo_detection(
            image_bgr, resolved_weights, conf_thresh=conf, iou_thresh=iou,
            use_esma=use_esma, sma_algorithm=sma_algorithm
        )
        _, buffer = cv2.imencode(".png", annotated)
        annotated_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

        quadrant_summary = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        for r in detected_regions:
            quadrant_summary[r["quadrant"]] += 1

        return {
            "status": "success",
            "annotated_image": annotated_b64,
            "detected_regions": detected_regions,
            "quadrant_summary": quadrant_summary,
            "disclaimer": (
                DISCLAIMER + " Detections come from a YOLOv8 model trained on "
                "DENTEX (mAP50 ~0.53, weaker on underrepresented classes) -- "
                "a supplementary research component, not a validated "
                "diagnostic tool."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}