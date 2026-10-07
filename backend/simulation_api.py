"""
simulation_api.py
------------------
Backend for the "Algorithm Simulation" view.

POST /simulate/agents/ runs the REAL standard_sma() and enhanced_sma()
from sma_algorithms.py on the uploaded OPG image and returns, for every
iteration, the actual position of every agent (its threshold vector),
which agents were the leaders, the best Kapur entropy so far and the
oscillation bound a(t). The frontend only draws what this endpoint
returns -- it does no searching of its own.

The exact Kapur optimum (dynamic programming) is also returned so the
simulation can mark the true best answer. It is computed after the two
runs and is never given to the optimizers.

It also returns real 2D slices of the Kapur-entropy landscape for the
image (see _landscapes), which the frontend draws behind the agents as
the heat map with its global and local optima.

Wire it into main.py with two lines:

    from simulation_api import router as simulation_router
    app.include_router(simulation_router)
"""

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from sma_algorithms import (
    KapurEntropyTable,
    apply_thresholds,
    autocrop_black_borders,
    compute_histogram_prob,
    enhanced_sma,
    kapur_optimal_thresholds,
    standard_sma,
)

router = APIRouter()


def _encode_png(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def _round_positions(frames):
    # one decimal is plenty for drawing and keeps the response small
    return [np.round(f, 1).tolist() for f in frames]


def _landscapes(table, optimum, d, step=4, window=15, max_local=2):
    """
    Real 2D slices of the Kapur-entropy landscape for this image, one per
    pair of thresholds (i < j). For a pair, the two plotted thresholds vary
    over 0..255 while the other thresholds are held at their exact-optimum
    values, so the global optimum always lies on the slice. Local optima
    are the other peaks of that slice (local maxima inside a
    `window` x `window` neighbourhood, in the region t_i < t_j).
    """
    g = np.arange(256, dtype=np.float64)
    X, Y = np.meshgrid(g, g, indexing="ij")
    half = window // 2
    out = {}
    for i in range(d):
        for j in range(i + 1, d):
            P = np.tile(np.asarray(optimum, dtype=np.float64), (256 * 256, 1))
            P[:, i] = X.ravel()
            P[:, j] = Y.ravel()
            Z = table.evaluate(P).reshape(256, 256)          # Z[x, y]

            padded = np.pad(Z, half, mode="constant", constant_values=-np.inf)
            neigh_max = np.lib.stride_tricks.sliding_window_view(padded, (window, window)).max(axis=(2, 3))
            is_peak = (Z >= neigh_max) & (X < Y)
            peaks = sorted(((float(Z[a, b]), int(a), int(b)) for a, b in zip(*np.where(is_peak))), reverse=True)
            gx, gy = int(optimum[i]), int(optimum[j])
            local = [
                {"x": a, "y": b, "fitness": round(v, 4)}
                for v, a, b in peaks
                if abs(a - gx) > half or abs(b - gy) > half
            ][:max_local]

            valid = Z[X < Y]
            out[f"{i}-{j}"] = {
                "step": step,
                "grid": np.round(Z[::step, ::step], 3).tolist(),   # grid[x_index][y_index]
                "max": round(float(Z.max()), 4),
                "floor": round(float(np.percentile(valid, 55)), 4),
                "local_optima": local,
            }
    return out


def _package(name, result, image, k_leaders):
    return {
        "algorithm": name,
        "thresholds": result["thresholds"],
        "fitness": round(float(result["fitness"]), 6),
        "runtime_sec": round(float(result["runtime_sec"]), 4),
        "iterations_used": int(result.get("iterations_used", len(result["convergence"]) - 1)),
        "k_leaders": k_leaders,
        "convergence": [round(float(v), 6) for v in result["convergence"]],
        "a_trace": [round(float(v), 4) for v in result["a_trace"]],
        "positions": _round_positions(result["positions"]),
        "leaders": result["leaders"],
        "segmented_image": _encode_png(apply_thresholds(image, result["thresholds"])),
        "segmented_color": _encode_png(
            cv2.applyColorMap(apply_thresholds(image, result["thresholds"]), cv2.COLORMAP_JET)),
    }


@router.post("/simulate/agents/")
async def simulate_agents(
    file: UploadFile = File(...),
    d: int = Form(4),
    N: int = Form(30),
    T: int = Form(100),
    seed: int = Form(42),
):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not decode image -- check the file is a valid image.")
        image = autocrop_black_borders(image)
        prob = compute_histogram_prob(image)

        std = standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, record_positions=True)
        esma = enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, record_positions=True)

        # answer key: exact optimum, computed AFTER the runs, for display only
        table = KapurEntropyTable(prob)
        best_score, best_thresholds = kapur_optimal_thresholds(prob, d, table=table)

        k_esma = len(esma["leaders"][0])
        enhanced = _package("Enhanced SMA (ESMA)", esma, image, k_esma)
        enhanced["init_unsorted"] = np.round(esma["init_unsorted"], 1).tolist()

        return {
            "status": "success",
            "params": {"d": d, "N": N, "T": T, "seed": seed},
            "optimum": {"thresholds": best_thresholds, "fitness": round(float(best_score), 6)},
            "landscapes": _landscapes(table, best_thresholds, d),
            "input_image": _encode_png(image),
            "histogram": np.bincount(image.ravel(), minlength=256)[:256].astype(int).tolist(),
            "standard": _package("Standard SMA", std, image, 1),
            "enhanced": enhanced,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}