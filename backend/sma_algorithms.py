"""
sma_algorithms.py
------------------
Real implementations of:
  1. Standard Slime Mould Algorithm (Li et al., 2020) applied to
     multilevel thresholding (Kapur's entropy as fitness).
  2. Enhanced Slime Mould Algorithm (ESMA) per Chapter 3 of the thesis:
       - Objective 1: Fitness-Weighted Multi-Leader Guidance
       - Objective 2: Quasi-Uniform Initialization
       - Objective 3: Performance-Feedback Adaptive Control (CR, PD)

Both return: best threshold vector, best Kapur entropy fitness,
and the convergence curve (best fitness per iteration) -- this is
what you need for the Chapter 4 comparative analysis (convergence
plots, mean/std over multiple runs, etc.)

NOTE ON THINGS THE THESIS TEXT DOES NOT SPECIFY:
The extracted Chapter 3 text references "Algorithm 3.1" for the full
ESMA pseudocode, but that figure/box was never actually inserted into
the PDF (the text jumps straight from "...summarized in Algorithm 3.1
below." to section 3.2.2 with nothing in between). This means the
following are NOT given anywhere in your draft and I had to pick
defaults for them -- you should decide on final values and state them
explicitly in your Chapter 3 (this is normal, every metaheuristic
paper has to state its control-parameter values somewhere):
  - alpha, beta, gamma, delta  (adaptation step sizes for z(t), a(t))
  - h                          (sliding window size for CR)
  - z_min, z_max               (clamp bounds for switching parameter)
  - k                          (number of leaders in multi-leader guidance)
  - stagnation thresholds for "CR near zero" / "PD low"
All are exposed as function parameters below so you can tune + report them.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Shared: Kapur's entropy fitness for multilevel thresholding
# ---------------------------------------------------------------------------

def compute_histogram_prob(gray_image: np.ndarray) -> np.ndarray:
    """Normalized 256-bin grayscale histogram (probability distribution)."""
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return hist
    return hist / total


def autocrop_black_borders(gray_image: np.ndarray, black_thresh: int = 8) -> np.ndarray:
    """
    Crops away solid/near-black letterboxing borders (common in exported
    OPG images that pad the film to a fixed canvas size) before the image
    is used for anything -- histogram, SMA/ESMA optimization, or the
    overlay. Left uncropped, a large black border dominates the pixel
    count near intensity 0 and skews Kapur's entropy toward separating
    "black border vs. content" rather than actual anatomical structures.
    Falls back to the original image untouched if no clear border is found.
    """
    mask = gray_image > black_thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray_image
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    # sanity check -- don't crop away almost everything on a weird image
    if (y1 - y0) < gray_image.shape[0] * 0.2 or (x1 - x0) < gray_image.shape[1] * 0.2:
        return gray_image
    return gray_image[y0:y1, x0:x1]


def kapurs_entropy_fitness(thresholds, prob: np.ndarray) -> float:
    """
    Kapur's entropy for a candidate threshold vector.
    Higher = better (this is a MAXIMIZATION problem).
    """
    th = sorted(int(np.clip(round(t), 0, 255)) for t in thresholds)
    bounds = [0] + th + [256]
    total_entropy = 0.0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        region = prob[lo:hi]
        Pi = region.sum()
        if Pi <= 1e-12:
            continue
        p_norm = region[region > 0] / Pi
        total_entropy += -np.sum(p_norm * np.log(p_norm))
    return float(total_entropy)


def apply_thresholds(gray_image: np.ndarray, thresholds) -> np.ndarray:
    """Segment the image into len(thresholds)+1 intensity bands."""
    th = sorted(int(np.clip(round(t), 0, 255)) for t in thresholds)
    bounds = [0] + th + [256]
    out = np.zeros_like(gray_image, dtype=np.uint8)
    n_bands = len(bounds) - 1
    for i in range(n_bands):
        lo, hi = bounds[i], bounds[i + 1]
        # evenly spread output gray levels across the band count so bands
        # are visually distinguishable (0..255)
        level = int(round(255 * i / max(1, n_bands - 1))) if n_bands > 1 else 255
        mask = (gray_image >= lo) & (gray_image < hi)
        out[mask] = level
    return out


# ---------------------------------------------------------------------------
# 1. STANDARD SMA (Li et al., 2020)
# ---------------------------------------------------------------------------

def standard_sma(prob, d, N=30, T=100, lb=0, ub=255, seed=None):
    """
    Baseline / original SMA -- single-leader guidance, random uniform
    initialization, fixed z and iteration-based oscillation schedule.
    Matches the "Existing Algorithm" formulas quoted in Chapter 3.1
    of the thesis (Statement of the Problem section).
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # --- random uniform initialization ---
    X = rng.uniform(lb, ub, size=(N, d))
    fitness = np.array([kapurs_entropy_fitness(X[i], prob) for i in range(N)])

    best_idx = int(np.argmax(fitness))
    Xb = X[best_idx].copy()
    bF = float(fitness[best_idx])
    convergence = [bF]

    for t in range(1, T + 1):
        order = np.argsort(-fitness)          # descending: index 0 = best
        curr_bF = float(fitness[order[0]])
        curr_wF = float(fitness[order[-1]])

        # weight W_i (top half vs bottom half of ranked population)
        W = np.zeros(N)
        half = N // 2
        for rank, idx in enumerate(order):
            r = rng.random()
            ratio = (curr_bF - fitness[idx]) / (curr_bF - curr_wF + 1e-12) + 1
            if rank < half:
                W[idx] = 1 + r * np.log(ratio)
            else:
                W[idx] = 1 - r * np.log(ratio)

        # fixed switching parameter, iteration-based oscillation bound
        z = 0.03
        a = np.arctanh(np.clip(-t / T + 1, -0.999999, 0.999999))
        vc = 1 - t / T  # linear decay 1 -> 0

        newX = X.copy()
        for i in range(N):
            if rng.random() < z:
                # random re-exploration
                newX[i] = rng.uniform(lb, ub, size=d)
            else:
                p = np.tanh(abs(fitness[i] - curr_bF))
                vb = rng.uniform(-a, a, size=d)
                ia, ib = rng.choice(N, 2, replace=False)
                XA, XB = X[ia], X[ib]
                if rng.random() < p:
                    newX[i] = Xb + vb * (W[i] * XA - XB)
                else:
                    newX[i] = vc * X[i]

        newX = np.clip(newX, lb, ub)
        newFitness = np.array([kapurs_entropy_fitness(newX[i], prob) for i in range(N)])

        # greedy selection (keep the better of old/new per agent)
        improve = newFitness > fitness
        X[improve] = newX[improve]
        fitness[improve] = newFitness[improve]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > bF:
            Xb = X[gen_best].copy()
            bF = float(fitness[gen_best])

        convergence.append(bF)

    elapsed = time.perf_counter() - t_start
    return {
        "thresholds": sorted(int(round(v)) for v in Xb),
        "fitness": bF,
        "convergence": convergence,
        "runtime_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# 2. ENHANCED SMA (ESMA) -- per thesis Chapter 3.2.1, Objectives 1-3
# ---------------------------------------------------------------------------

def enhanced_sma(
    prob, d, N=30, T=100, lb=0, ub=255, seed=None,
    k=3,                 # number of leaders (Objective 1) -- NOT specified in draft, tune/report
    alpha=0.10, beta=0.10, gamma=0.10, delta=0.10,   # adaptation step sizes -- NOT specified in draft
    h=5,                  # CR sliding window -- NOT specified in draft
    z_min=0.01, z_max=0.5,  # clamp bounds -- NOT specified in draft
    cr_stall_eps=1e-6,    # "CR near zero" threshold
    pd_low_thresh=0.10,   # "PD low" threshold
):
    """
    ESMA with all three proposed modifications:
      Obj 1: fitness-weighted multi-leader guidance (replaces single Xb)
      Obj 2: quasi-uniform (stratified) initialization
      Obj 3: performance-feedback adaptive control of z(t) and a(t)
             driven by convergence rate CR(t) and population diversity PD(t)
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # --- Objective 2: quasi-uniform initialization ---
    X = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            X[i, j] = lb + ((i / N) + rng.random() / N) * (ub - lb)

    fitness = np.array([kapurs_entropy_fitness(X[i], prob) for i in range(N)])
    best_idx = int(np.argmax(fitness))
    Xb = X[best_idx].copy()
    bF = float(fitness[best_idx])

    bF_history = [bF]
    convergence = [bF]

    # search space diameter (for PD normalization)
    D = float(np.sqrt(d) * (ub - lb))

    z = 0.03      # starting switching parameter (same start as standard SMA)
    a = 1.0       # starting oscillation bound

    for t in range(1, T + 1):
        order = np.argsort(-fitness)
        curr_bF = float(fitness[order[0]])
        curr_wF = float(fitness[order[-1]])

        # same W_i formula as standard SMA
        W = np.zeros(N)
        half = N // 2
        for rank, idx in enumerate(order):
            r = rng.random()
            ratio = (curr_bF - fitness[idx]) / (curr_bF - curr_wF + 1e-12) + 1
            if rank < half:
                W[idx] = 1 + r * np.log(ratio)
            else:
                W[idx] = 1 - r * np.log(ratio)

        # --- Objective 1: top-k leaders + fitness-proportional weights ---
        kk = min(k, N)
        leader_idx = order[:kk]
        leader_fits = fitness[leader_idx]
        # shift to strictly positive before normalizing (Kapur's entropy is
        # always >= 0 in practice, but this keeps it robust either way)
        shifted = leader_fits - min(0.0, float(leader_fits.min())) + 1e-9
        w_leaders = shifted / shifted.sum()

        vc = 1 - t / T
        newX = X.copy()
        for i in range(N):
            if rng.random() < z:
                newX[i] = rng.uniform(lb, ub, size=d)
            else:
                p = np.tanh(abs(fitness[i] - curr_bF))
                vb = rng.uniform(-a, a, size=d)
                ia, ib = rng.choice(N, 2, replace=False)
                XA, XB = X[ia], X[ib]
                if rng.random() < p:
                    pos = np.zeros(d)
                    for jl in range(kk):
                        Lj = X[leader_idx[jl]]
                        pos += w_leaders[jl] * (Lj + vb * (W[i] * XA - XB))
                    newX[i] = pos
                else:
                    newX[i] = vc * X[i]

        newX = np.clip(newX, lb, ub)
        newFitness = np.array([kapurs_entropy_fitness(newX[i], prob) for i in range(N)])

        improve = newFitness > fitness
        X[improve] = newX[improve]
        fitness[improve] = newFitness[improve]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > bF:
            Xb = X[gen_best].copy()
            bF = float(fitness[gen_best])

        bF_history.append(bF)
        convergence.append(bF)

        # --- Objective 3: performance-feedback adaptive control ---
        if t >= h:
            bF_prev = bF_history[t - h]
            # NOTE: sign flipped vs. the thesis formula because Kapur's
            # entropy is MAXIMIZED here (bF increases as the run improves),
            # whereas the thesis's CR formula assumes a minimization
            # convention (bF decreases when improving). CR > 0 == "improving"
            # in both cases with this adjustment -- flag this in your
            # methodology writeup.
            CR = (bF - bF_prev) / (abs(bF) + 1e-10)
        else:
            CR = 0.0

        # mean pairwise distance / diameter
        if N > 1:
            diffs = X[:, None, :] - X[None, :, :]
            dist_sum = np.sqrt((diffs ** 2).sum(axis=-1)).sum()
            PD = dist_sum / (N * (N - 1) * D)
        else:
            PD = 0.0

        stagnating = (abs(CR) < cr_stall_eps) and (PD < pd_low_thresh)
        if stagnating:
            z = float(np.clip(z + alpha * (1 - PD) * (1 - CR), z_min, z_max))
            a = a + gamma * (1 - CR)
        elif CR > 0:
            z = float(np.clip(z - beta * CR, z_min, z_max))
            a = max(a - delta * CR, 1e-6)
        # else (CR < 0, i.e. got worse overall -- shouldn't happen with
        # greedy selection, but kept for completeness): leave z, a unchanged

    elapsed = time.perf_counter() - t_start
    return {
        "thresholds": sorted(int(round(v)) for v in Xb),
        "fitness": bF,
        "convergence": convergence,
        "runtime_sec": elapsed,
    }