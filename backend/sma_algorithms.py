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


def autocrop_black_borders(gray_image: np.ndarray, black_thresh: int = 8,
                            white_thresh: int = 247, row_std_thresh: float = 3.0) -> np.ndarray:
    """
    Crops away solid, near-uniform letterboxing borders -- both BLACK
    padding and WHITE padding (some exported OPG images have a solid
    white frame at the top/bottom instead of black; a full-white row is
    just as much "not anatomical content" as a full-black one). This
    runs before the image is used for anything -- histogram, SMA/ESMA
    optimization, or the overlay. Left uncropped, a uniform border
    (black OR white) dominates the pixel count at one intensity extreme
    and skews Kapur's entropy toward separating "border vs. content"
    rather than actual anatomical structures, and can fool
    brightness-based heuristics (like the overlay's arch-band detector)
    into thinking the border is the brightest, most relevant region.

    A row/column counts as padding only if it is BOTH near-uniform (low
    std -- real tissue always has texture) AND near an intensity
    extreme (very dark or very bright) -- this avoids accidentally
    stripping a genuinely bright but textured anatomical row.
    Falls back to the original image untouched if no clear border is found.
    """
    h, w = gray_image.shape

    def _is_padding_row(row):
        return row.std() < row_std_thresh and (row.mean() < black_thresh or row.mean() > white_thresh)

    top = 0
    while top < h and _is_padding_row(gray_image[top, :]):
        top += 1
    bottom = h
    while bottom > top and _is_padding_row(gray_image[bottom - 1, :]):
        bottom -= 1
    left = 0
    while left < w and _is_padding_row(gray_image[:, left]):
        left += 1
    right = w
    while right > left and _is_padding_row(gray_image[:, right - 1]):
        right -= 1

    # sanity check -- don't crop away almost everything on a weird image
    if (bottom - top) < h * 0.2 or (right - left) < w * 0.2:
        return gray_image
    return gray_image[top:bottom, left:right]


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

        # weight W_i (top half vs bottom half of ranked population) --
        # vectorized: same formula as before, no Python-level agent loop
        half = N // 2
        r_w = rng.random(N)
        ratio = (curr_bF - fitness[order]) / (curr_bF - curr_wF + 1e-12) + 1
        sign = np.where(np.arange(N) < half, 1.0, -1.0)
        W_ordered = 1 + sign * r_w * np.log(ratio)
        W = np.empty(N)
        W[order] = W_ordered

        # fixed switching parameter, iteration-based oscillation bound
        z = 0.03
        a = np.arctanh(np.clip(-t / T + 1, -0.999999, 0.999999))
        vc = 1 - t / T  # linear decay 1 -> 0

        # vectorized branching (same three cases as before: random
        # re-exploration / multi-agent exploitation / decay toward Xb)
        rand_explore = rng.random(N) < z
        p_vals = np.tanh(np.abs(fitness - curr_bF))
        choose_exploit = rng.random(N) < p_vals

        vb = rng.uniform(-a, a, size=(N, d))
        ia = rng.integers(0, N, size=N)
        ib = rng.integers(0, N, size=N)
        clash = ia == ib
        while np.any(clash):
            ib[clash] = rng.integers(0, N, size=int(clash.sum()))
            clash = ia == ib
        XA = X[ia]
        XB = X[ib]

        exploit_pos = Xb[None, :] + vb * (W[:, None] * XA - XB)
        decay_pos = vc * X
        random_pos = rng.uniform(lb, ub, size=(N, d))

        newX = np.where(
            rand_explore[:, None], random_pos,
            np.where(choose_exploit[:, None], exploit_pos, decay_pos)
        )

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
    k=3,                 # number of leaders (Algorithm 3.1) / k_max if adaptive_k=True
    alpha=0.10, beta=0.10, gamma=0.10, delta=0.10,   # adaptation step sizes
    h=5,                  # CR sliding window
    z_min=0.01, z_max=0.5,  # clamp bounds
    cr_stall_eps=1e-6,    # tau_CR in the thesis
    pd_low_thresh=0.10,   # tau_PD in the thesis
    adaptive_k=False,     # extension beyond Algorithm 3.1 -- see docstring
):
    """
    ESMA implementing all three proposed modifications, matching
    Algorithm 3.1 in the thesis LITERALLY (as found in the full PDF,
    which includes the pseudocode box that was missing from an earlier
    draft):
      Obj 1: fitness-weighted multi-leader guidance
      Obj 2: quasi-uniform (stratified) initialization
      Obj 3: performance-feedback adaptive control of z(t) and a(t)

    IMPORTANT: Algorithm 3.1 as written has EVERY agent, EVERY
    iteration, move via the multi-leader weighted formula (step 5.4)
    UNCONDITIONALLY -- there is no z-triggered random-reinitialization
    branch and no probability-gated choice between "exploit" and
    "decay" (both of which standard SMA has, and which an earlier
    version of this function incorrectly carried over into ESMA). This
    version follows the pseudocode literally: no branching in the
    position update. z(t) is still computed and updated per steps
    5.5/5.6 (as the thesis specifies), but note it does not appear
    inside the step 5.4 formula itself -- only a(t) does, via vb's
    range. That asymmetry (z computed but not consumed by name) is in
    the thesis's own pseudocode, not something introduced here; it may
    be worth flagging to your adviser as a documentation point, but
    this function implements exactly what Algorithm 3.1 specifies.

    OPTIONAL EXTENSION -- Diversity-Driven Adaptive Leader Count:
    when adaptive_k=True, k is no longer fixed; it decays from
    k_max toward 1 as population diversity PD(t) collapses, using
        k(t) = max(1, round(k_max * PD(t-1) / PD_max))
    where PD_max is the diversity of the initial (quasi-uniform)
    population and PD(t-1) is the diversity measured at the END of
    the previous iteration (using the previous iteration's value,
    not the current one, for the same causal reason z(t) and a(t)
    are updated at the end of an iteration and consumed at the start
    of the next). This is documented as an extension beyond the
    literal Algorithm 3.1 -- it must be described in Chapter 3 if
    used, since the pseudocode itself specifies a fixed k.
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # --- Step 2: quasi-uniform initialization ---
    X = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            X[i, j] = lb + ((i / N) + rng.random() / N) * (ub - lb)

    # --- Steps 3-4 ---
    fitness = np.array([kapurs_entropy_fitness(X[i], prob) for i in range(N)])
    best_idx = int(np.argmax(fitness))
    Xb = X[best_idx].copy()
    bF = float(fitness[best_idx])

    bF_history = [bF]
    convergence = [bF]
    D = float(np.sqrt(d) * (ub - lb))
    z = 0.03
    a = 1.0

    k_max = k  # the k passed in is treated as the ceiling when adaptive_k is on
    if adaptive_k and N > 1:
        diffs0 = X[:, None, :] - X[None, :, :]
        PD_max = float(np.sqrt((diffs0 ** 2).sum(axis=-1)).sum() / (N * (N - 1) * D))
        PD_max = max(PD_max, 1e-9)
    else:
        PD_max = 1.0
    PD_prev = PD_max  # iteration 1 sees the initial (maximally diverse) population

    for t in range(1, T + 1):
        # --- adaptive leader count (extension) or fixed k (Algorithm 3.1) ---
        if adaptive_k:
            kk = max(1, min(N, int(round(k_max * PD_prev / PD_max))))
        else:
            kk = min(k, N)

        # --- 5.1: rank population, select top-k leaders ---
        order = np.argsort(-fitness)
        leader_idx = order[:kk]
        leader_fits = fitness[leader_idx]

        # --- 5.2: fitness weights w[j] = S(Lj) / sum(S(Lm)) ---
        denom = leader_fits.sum()
        w_leaders = leader_fits / denom if denom > 1e-12 else np.full(kk, 1.0 / kk)

        # --- 5.3: adaptive weight W[i] (same W_i formula as standard SMA)
        # -- vectorized, no Python-level agent loop ---
        curr_bF = float(fitness[order[0]])
        curr_wF = float(fitness[order[-1]])
        half = N // 2
        r_w = rng.random(N)
        ratio = (curr_bF - fitness[order]) / (curr_bF - curr_wF + 1e-12) + 1
        sign = np.where(np.arange(N) < half, 1.0, -1.0)
        W_ordered = 1 + sign * r_w * np.log(ratio)
        W = np.empty(N)
        W[order] = W_ordered

        # --- 5.4: EVERY agent moves via the multi-leader weighted
        # formula, unconditionally (no branching -- see docstring).
        # Vectorized across agents AND leaders via broadcasting/tensordot
        # -- mathematically identical to summing per-leader per-agent in
        # a Python loop, just without the loop overhead. ---
        vb = rng.uniform(-a, a, size=(N, d))
        ia = rng.integers(0, N, size=N)
        ib = rng.integers(0, N, size=N)
        clash = ia == ib
        while np.any(clash):
            ib[clash] = rng.integers(0, N, size=int(clash.sum()))
            clash = ia == ib
        XA = X[ia]                       # (N, d)
        XB = X[ib]                       # (N, d)
        inner = W[:, None] * XA - XB     # (N, d)
        L = X[leader_idx]                # (kk, d)
        # term[j, i, :] = L[j] + vb[i] * inner[i]  -> shape (kk, N, d)
        term = L[:, None, :] + vb[None, :, :] * inner[None, :, :]
        newX = np.tensordot(w_leaders, term, axes=(0, 0))  # (N, d)

        newX = np.clip(newX, lb, ub)
        newFitness = np.array([kapurs_entropy_fitness(newX[i], prob) for i in range(N)])

        # --- 5.7/5.8: greedy update (keep the better of old/new per
        # agent -- standard convention, prevents fitness regressing) ---
        improve = newFitness > fitness
        X[improve] = newX[improve]
        fitness[improve] = newFitness[improve]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > bF:
            Xb = X[gen_best].copy()
            bF = float(fitness[gen_best])

        bF_history.append(bF)
        convergence.append(bF)

        # --- 5.5: compute CR(t) and PD(t) ---
        if t >= h:
            bF_prev = bF_history[t - h]
            # NOTE: sign flipped vs. the thesis formula because Kapur's
            # entropy is MAXIMIZED here (bF increases as the run
            # improves), whereas the thesis's CR formula as written
            # assumes a minimization convention (bF decreases when
            # improving -- confirmed by the thesis text itself: "when
            # CR(t) is positive, indicating productive convergence").
            # This adjustment preserves that intended meaning (CR > 0
            # == productive convergence) under maximization.
            CR = (bF - bF_prev) / (abs(bF) + 1e-10)
        else:
            CR = 0.0

        if N > 1:
            diffs = X[:, None, :] - X[None, :, :]
            dist_sum = np.sqrt((diffs ** 2).sum(axis=-1)).sum()
            PD = dist_sum / (N * (N - 1) * D)
        else:
            PD = 0.0

        # --- 5.6: update z(t) and a(t) ---
        stagnating = (abs(CR) < cr_stall_eps) and (PD < pd_low_thresh)
        if stagnating:
            z = float(np.clip(z + alpha * (1 - PD) * (1 - CR), z_min, z_max))
            a = a + gamma * (1 - CR)
        elif CR > 0:
            z = float(np.clip(z - beta * CR, z_min, z_max))
            a = max(a - delta * CR, 1e-6)

        # store this iteration's diversity for next iteration's leader-count
        # decision (adaptive_k) -- causally correct: k(t+1) is chosen using
        # information available at the end of iteration t, same as z/a
        PD_prev = PD

    elapsed = time.perf_counter() - t_start
    return {
        "thresholds": sorted(int(round(v)) for v in Xb),
        "fitness": bF,
        "convergence": convergence,
        "runtime_sec": elapsed,
    }