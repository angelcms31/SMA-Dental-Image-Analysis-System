"""
main.py
-------
FastAPI backend for the thesis system. Replaces the previous placeholder
(hardcoded threshold range + fake "confidence %" derived from x+y+area).

Three endpoints:
  POST /analyze/standard   -> runs the ORIGINAL SMA (Li et al., 2020)
  POST /analyze/enhanced   -> runs the proposed ESMA
  POST /analyze/compare    -> runs BOTH on the same image with the same
                               params/seed and returns a side-by-side
                               result -- this is what feeds your Chapter 4
                               comparative-analysis table/figures directly.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sma_algorithms import (
    apply_thresholds,
    compute_histogram_prob,
    enhanced_sma,
    kapurs_entropy_fitness,
    standard_sma,
)
from metrics import compute_psnr, compute_ssim
from overlay import DISCLAIMER, generate_annotated_overlay

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