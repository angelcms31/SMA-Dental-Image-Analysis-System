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

  3. Enhanced Slime Mould Algorithm v2 (enhanced_sma_v2): the same three
     objectives with the defects of the v1 implementation repaired, plus
     adaptive termination and an opt-in local polish (see its docstring
     for the full change list). Shared v2 infrastructure:
       - KapurEntropyTable: every class entropy tabulated once per image,
         whole population evaluated in one vectorized call. Same objective
         values as kapurs_entropy_fitness (agreement ~1e-14), ~9x faster
         runs. standard_sma / enhanced_sma use it too (fast_fitness=True;
         pass False for the original per-agent loop -- identical results,
         verified on the 100 reference images in tests/).
       - kapur_optimal_thresholds: the EXACT global Kapur optimum by
         dynamic programming (Kapur's objective is a sum over classes).
         Evaluation reference only ("optimality gap", "% of images where a
         run reached the true optimum") -- never called by the optimizers.

     Defects of the v1 ESMA that motivated v2 (all verified on this code):
       D1 z(t) was updated but never consumed (no re-initialization branch)
       D2 a(t) -= delta*CR with CR ~ 1e-5..1e-3 -> a(t) never left 1.0
       D3 a(t) grew without bound during stagnation
       D4 quasi-uniform init used the same stratum in every dimension ->
          the whole population started on the diagonal t1 ~ t2 ~ ... ~ td
       D5 agents were not kept sorted, so leader centroids and W*XA - XB
          mixed unrelated threshold coordinates (d! redundant orderings)
       D6 w_j = S(L_j)/sum S with all S ~ 19.2 -> uniform leader weights

All three return: best threshold vector, best Kapur entropy fitness,
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
    if gray_image.dtype == np.uint8:
        # exact same counts as np.histogram(..., bins=256, range=(0, 256))
        # for 8-bit input, ~10x faster on multi-megapixel OPG images
        hist = np.bincount(gray_image.ravel(), minlength=256)
    else:
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


# ---------------------------------------------------------------------------
# Shared v2 infrastructure (implementation-level, objective unchanged):
#   * KapurEntropyTable      -- every class entropy tabulated once per image,
#                               whole population evaluated in one numpy call
#   * kapur_optimal_thresholds -- EXACT global optimum by dynamic programming
#                               (evaluation reference only, never used inside
#                               the optimizers)
# ---------------------------------------------------------------------------

_TRIU_CACHE = {}


def _triu_mask(L):
    m = _TRIU_CACHE.get(L)
    if m is None:
        m = np.triu(np.ones((L, L), dtype=bool))
        _TRIU_CACHE[L] = m
    return m


class KapurEntropyTable:
    """
    Precomputed Kapur class entropies for EVERY possible intensity class.

    Kapur's objective is a sum of independent per-class terms. For the class
    covering bins [lo, hi) with P = sum(prob[lo:hi]) and S = sum(p ln p) over
    the class,

        H[lo, hi) = -sum (p/P) ln (p/P) = ln P - S / P ,

    so all (L+1)^2 class entropies can be tabulated once per image (~1 ms,
    0.5 MB for L=256) and any threshold vector is then evaluated with d+1
    table lookups. The value is the SAME as kapurs_entropy_fitness() -- same
    formula, same rounding / clipping / sorting of the thresholds, empty or
    near-empty classes (P <= 1e-12) contribute 0 -- to ~1e-12 (checked in
    tests/test_sma_algorithms.py). Only the Python-level per-agent loop is
    removed, so this is an implementation optimization, not a change of the
    objective.

    Partial sums start fresh at `lo` (per-row cumsum over a triangular
    matrix) rather than being differences of two global cumulative sums, so
    small classes in the histogram tails do not lose precision to
    cancellation.
    """
    __slots__ = ("H", "L")

    def __init__(self, prob):
        prob = np.asarray(prob, dtype=np.float64).ravel()
        L = prob.shape[0]
        self.L = L
        plogp = np.zeros(L)
        nz = prob > 0
        plogp[nz] = prob[nz] * np.log(prob[nz])
        tri = _triu_mask(L)
        # row lo holds prob[i] for i >= lo -> cumsum along the row gives
        # P(lo, hi) = sum(prob[lo:hi]) in column hi-1 (no cancellation)
        P = np.cumsum(np.where(tri, prob[None, :], 0.0), axis=1)
        S = np.cumsum(np.where(tri, plogp[None, :], 0.0), axis=1)
        valid = P > 1e-12                       # same skip rule as the scalar function
        Psafe = np.where(valid, P, 1.0)
        Hv = np.where(valid, np.log(Psafe) - S / Psafe, 0.0)
        H = np.zeros((L + 1, L + 1))
        # H[lo, hi] = Hv[lo, hi-1] for hi > lo; hi <= lo (empty class) stays 0
        H[:L, 1:] = np.where(tri, Hv, 0.0)
        self.H = H

    def evaluate(self, X, assume_sorted=False):
        """
        Kapur fitness of every row of X (shape (N, d)) -> shape (N,).
        Identical semantics to kapurs_entropy_fitness applied row by row:
        round half-to-even, clip to [0, L-1], sort, sum class entropies.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Ti = np.clip(np.rint(X), 0, self.L - 1).astype(np.intp)
        if not assume_sorted:
            Ti.sort(axis=1)
        n = Ti.shape[0]
        lo = np.concatenate([np.zeros((n, 1), dtype=np.intp), Ti], axis=1)
        hi = np.concatenate([Ti, np.full((n, 1), self.L, dtype=np.intp)], axis=1)
        return self.H[lo, hi].sum(axis=1)

    def evaluate_one(self, thresholds) -> float:
        return float(self.evaluate(np.asarray(thresholds, dtype=np.float64))[0])


def _population_fitness(X, prob, table):
    """Fitness of all rows of X: table lookup when available, otherwise the
    original per-agent Python loop (byte-identical legacy path)."""
    if table is not None:
        return table.evaluate(X)
    return np.array([kapurs_entropy_fitness(X[i], prob) for i in range(X.shape[0])])


def kapur_optimal_thresholds(prob, d, table=None):
    """
    EXACT global maximum of Kapur's entropy for d integer thresholds in
    [0, L-1] (duplicates allowed -- exactly the space the SMA variants search),
    by dynamic programming over the class-entropy table:

        g_1[t] = H[0, t]
        g_j[t] = max_{s <= t} ( g_{j-1}[s] + H[s, t] )      j = 2..d
        H*     = max_t ( g_d[t] + H[t, L] )

    Cost O(d * L^2): a few milliseconds for L = 256. This is an EVALUATION
    REFERENCE (optimality gap of a metaheuristic run, "% of images where the
    optimizer found the true optimum") -- the same role known optima of
    benchmark functions play in the metaheuristics literature. It is never
    called inside standard_sma / enhanced_sma / enhanced_sma_v2.

    Returns (H_star, thresholds) with thresholds sorted ascending.
    """
    table = table if table is not None else KapurEntropyTable(prob)
    H, L = table.H, table.L
    Hc = H[:L, :L]                                      # H[s, t], s,t in 0..L-1
    invalid = ~_triu_mask(L)                            # s > t is not allowed
    g = H[0, :L].copy()                                 # g_1[t] = H[0, t]
    back = []
    for _ in range(1, int(d)):
        cand = g[:, None] + Hc                          # cand[s, t]
        cand[invalid] = -np.inf
        arg = np.argmax(cand, axis=0)
        back.append(arg)
        g = cand[arg, np.arange(L)]
    final = g + H[:L, L]                                # + H[t, L): last class
    t = int(np.argmax(final))
    best = float(final[t])
    thr = [t]
    for arg in reversed(back):
        t = int(arg[t])
        thr.append(t)
    return best, sorted(thr)


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

def standard_sma(prob, d, N=30, T=100, lb=0, ub=255, seed=None, fast_fitness=True):
    """
    Baseline / original SMA -- single-leader guidance, random uniform
    initialization, fixed z and iteration-based oscillation schedule.
    Matches the "Existing Algorithm" formulas quoted in Chapter 3.1
    of the thesis (Statement of the Problem section).

    fast_fitness: evaluate the population through KapurEntropyTable (same
    objective values, one vectorized call per iteration) instead of the
    original per-agent Python loop. The random-number stream and every
    update rule are unchanged, so trajectories are identical except where
    two candidate threshold vectors have EXACTLY equal entropy and the
    two implementations differ in the last floating-point digit
    (see tests/test_sma_algorithms.py for the measured agreement).
    Pass fast_fitness=False to run the legacy path byte-for-byte.
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    table = KapurEntropyTable(prob) if fast_fitness else None

    # --- random uniform initialization ---
    X = rng.uniform(lb, ub, size=(N, d))
    fitness = _population_fitness(X, prob, table)

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
        newFitness = _population_fitness(newX, prob, table)

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
    fast_fitness=True,    # table-based population evaluation (see standard_sma docstring)
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
    table = KapurEntropyTable(prob) if fast_fitness else None

    # --- Step 2: quasi-uniform initialization ---
    X = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            X[i, j] = lb + ((i / N) + rng.random() / N) * (ub - lb)

    # --- Steps 3-4 ---
    fitness = _population_fitness(X, prob, table)
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
        newFitness = _population_fitness(newX, prob, table)

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


# ---------------------------------------------------------------------------
# 3. ENHANCED SMA v2 -- the three thesis objectives, repaired and completed
# ---------------------------------------------------------------------------

def _mean_pairwise_distance(X):
    """
    Mean Euclidean distance over all agent pairs (numerator of the thesis
    PD(t)). Plain numpy broadcasting: for populations this small (N <= ~50)
    it is ~5x faster than scipy's pdist wrapper and keeps the core module
    free of a scipy dependency.
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs)).sum() / (n * (n - 1)))


def _integer_polish(x, f, evaluate, lb, ub, max_rounds=64):
    """
    OPT-IN v2 operator (local_refine=True): coordinate-wise +/-1 hill climb on
    the integer threshold grid around the final best solution, using the
    fitness table. 2*d candidates per round, stops at the first round with no
    improvement. This is an operator BEYOND the SMA's own components, so it is
    off by default and reported separately in the benchmarks.
    """
    x = np.rint(np.asarray(x, dtype=np.float64))
    d = x.shape[0]
    idx = np.arange(d)
    n_eval = 0
    for _ in range(max_rounds):
        cand = np.repeat(x[None, :], 2 * d, axis=0)
        cand[idx, idx] += 1.0
        cand[d + idx, idx] -= 1.0
        np.clip(cand, lb, ub, out=cand)
        vals = evaluate(cand)
        n_eval += 2 * d
        j = int(np.argmax(vals))
        if vals[j] > f + 1e-12:
            x = np.sort(cand[j])
            f = float(vals[j])
        else:
            break
    return x, f, n_eval


def _initial_population(rng, N, d, lb, ub, init="lhs"):
    """
    Initial agent positions (Objective 2).
      'lhs'    : Latin-hypercube quasi-uniform strata -- every dimension is
                 split into N equal intervals and each interval holds exactly
                 one agent, with the interval assignment permuted
                 independently per dimension (agents cover the hypercube).
      'strata' : thesis-literal iSMA formula X[i,j] = lb + ((i-1)/N + rand/N)(ub-lb):
                 the SAME interval index for every dimension, so the whole
                 population lies on the diagonal t1 ~ t2 ~ ... ~ td (v1 defect).
      'uniform': Standard-SMA random uniform initialization.
    """
    span = float(ub - lb)
    if init == "lhs":
        strata = np.column_stack([rng.permutation(N) for _ in range(d)]).astype(np.float64)
        return lb + (strata + rng.random((N, d))) / N * span
    if init == "strata":
        return lb + (np.arange(N, dtype=np.float64)[:, None] + rng.random((N, d))) / N * span
    if init == "uniform":
        return rng.uniform(lb, ub, size=(N, d))
    raise ValueError(f"init must be 'lhs', 'strata' or 'uniform', got {init!r}")


def enhanced_sma_v2(
    prob, d, N=30, T=100, lb=0, ub=255, seed=None,
    # ---- Objective 1: fitness-weighted multi-leader guidance ----
    k=5,                       # number of leaders (thesis 3.4.1 uses k = 5)
    leader_mode="sampled",     # "sampled" : each agent follows ONE leader drawn with prob w_j
                               #             (E[anchor] = sum_j w_j L_j, i.e. the thesis formula in expectation)
                               # "centroid": X_guide = sum_j w_j L_j (thesis formula, deterministic)
    weight_mode="rank",        # "rank": w_j ~ k-j+1 | "fitness": w_j = S(L_j)/sum S (thesis literal)
                               # "advantage": w_j ~ S(L_j) - S(L_k) + 0.1*spread
    # ---- Objective 2: quasi-uniform initialization ----
    init="lhs",                # "lhs": Latin-hypercube strata | "strata": thesis-literal (diagonal)
                               # "uniform": Standard-SMA random init
    canonical=True,            # keep every agent's threshold vector sorted ascending
    # ---- Objective 3: performance-feedback adaptive control ----
    h=5,                       # CR(t) look-back window
    a0=1.0, a_min=0.05, a_max=1.0,      # oscillation amplitude a(t): start, floor, cap
    z0=0.03, z_min=0.01, z_max=0.10,    # switching probability z(t): start, clamp
    alpha=0.10, beta=0.10, gamma=0.10, delta=0.10,   # adaptation step sizes (same roles as enhanced_sma)
    cr_stall_eps=1e-6, pd_low_thresh=0.10,           # tau_CR, tau_PD
    contract_mode="prop",      # "prop": steps scaled by normalized CR (thesis form) | "geom": fixed steps
    pd_scope="core",           # "core": PD over the fitter half of the population | "all": every agent
    explore_channel=False,     # True: z(t) gates an exploration move for non-leader agents (see docstring:
                               # measured neutral-to-harmful for Kapur thresholding at T=50, hence off)
    explore_mode="wide",       # "wide": SMA move with the full amplitude a_max (greedy-selected)
                               # "reinit": Standard-SMA random re-initialization (accepted unconditionally)
    # ---- v2 additions ----
    early_stop=True, patience=20,   # adaptive termination after `patience` non-improving iterations
    local_refine=False,        # opt-in integer-neighbourhood polish of the final best (see _integer_polish)
    fast_fitness=True,         # table-based population evaluation (implementation-level)
    record_history=False,      # return per-iteration a, z, PD, CR traces (for Chapter 4 figures)
):
    """
    Enhanced SMA, version 2. Same three objectives as enhanced_sma() (thesis
    Chapter 3), with the defects found in the v1 implementation repaired and
    two additions. Returns the SAME dict schema as standard_sma/enhanced_sma
    (thresholds, fitness, convergence, runtime_sec) plus `iterations_used`,
    `n_reinitialized`, `n_polish_evaluations` and (optionally) `history`.

    WHAT CHANGED vs enhanced_sma (v1) AND WHY
    -----------------------------------------
    Objective 1 -- multi-leader guidance
      * Position update is the thesis formula with the sum pulled through:
        X_i(t+1) = sum_j w_j [L_j + vb (W XA - XB)] = X_guide + vb (W XA - XB),
        X_guide = sum_j w_j L_j (exact since sum_j w_j = 1; v1's tensordot
        computed the same thing at (k,N,d) cost).
      * weight_mode: raw Kapur values are all ~19.2, so v1's w_j = S(L_j)/sum S
        was ~1/k for every leader ("fitness-weighted" in name only). Rank or
        advantage weights are scale-free and actually favour the best leader.
      * leader_mode="sampled" is an optional stochastic form: an agent follows
        one leader drawn with probability w_j, so E[anchor] = X_guide exactly
        while different agents are pulled toward different leaders.
    Objective 2 -- quasi-uniform initialization
      * v1 used the same stratum index i in EVERY dimension, so agent i had all
        d thresholds inside one (ub-lb)/N-wide band (the population lay on the
        diagonal t1~t2~...~td). init="lhs" keeps the guarantee "every dimension
        is split into N strata and each stratum holds exactly one agent" but
        assigns strata by an independent random permutation per dimension
        (Latin hypercube), so agents cover threshold CONFIGURATIONS.
      * canonical=True: fitness depends only on sorted(thresholds), so every
        agent is kept sorted. Without this, coordinate j means different
        things in different agents and both the leader centroid and the
        difference vector W XA - XB mix unrelated thresholds (v1 defect).
    Objective 3 -- performance-feedback control
      * z(t) was computed but never used in v1. Here it gates the Standard
        SMA's own random re-initialization branch (explore_channel): each
        non-leader agent is re-initialized with probability z(t); those agents
        are accepted unconditionally (leaders are protected, so bF cannot be
        lost) -- otherwise, under greedy selection, a random restart could
        never survive and diversity could never be re-injected.
      * CR(t) is a RELATIVE entropy change (~1e-5..1e-3), so v1's
        a -= delta*CR and z -= beta*CR were ~1e-5 per iteration and a(t) never
        left 1.0 (no fine exploitation phase). Here CR is normalized by its
        running maximum (thesis 3.4.1: CR/CR_max), a(t) contracts
        multiplicatively toward a_min under progress and grows additively
        toward a_max under stagnation (bounded -- v1 grew a without limit).
      * PD(t) normalized by the initial diversity (thesis: PD/PD_max);
        pd_scope="core" measures it over the fitter half so a handful of
        freshly re-initialized agents cannot mask a collapsed core.
    Additions
      * early_stop/patience: adaptive termination once the best has not
        improved for `patience` iterations (the feedback loop has already
        widened a and z by then). Reported against the equal-budget run.
      * local_refine (default False): see _integer_polish.
      * fast_fitness: KapurEntropyTable population evaluation (same values).

    HOW THE DEFAULTS WERE CHOSEN (development split only: 40 Mendeley OPGs
    disjoint from the reported hold-out set; 3-8 seeds x 40 images per
    configuration; metric = gap to the exact DP optimum, N=30, T=50, d=4)
      * canonical ordering and sampled leaders are the two large effects
        (mean gap 0.0010 vs 0.0075 without canonical; 0.0011 vs 0.0071 for
        centroid guidance, averaged over 200 random configurations).
      * mild amplitude contraction (delta = 0.1) beats aggressive contraction
        (delta = 0.5 was the worst setting); a0 = 1 beats a0 = 2.
      * a_min, gamma, pd_low_thresh, weight_mode and init (lhs vs uniform)
        change results by less than the seed-to-seed noise.
      * k = 5 with h = 5 gave the best mean gap AND hit rate over 8 seeds
        (0.00015 / 88.1%, vs 0.00364 / 62.8% for enhanced_sma and
        0.00812 / 0.3% for standard_sma).
      * the z(t) exploration channel (either mode, any z0/z_max tried) never
        improved on explore_channel=False at T=50 (hit rate 81-88% vs 88%);
        random re-initialization spends evaluations that Kapur thresholding
        rewards more when spent on polishing. z(t) is still computed and
        returned in `history`; set explore_channel=True to use it.
      * patience = 20 keeps the mean gap of the full-T run (0.00015) at ~20%
        fewer iterations (39.8 vs 50; hit rate 85.9% vs 88.1%).
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    N, d, T = int(N), int(d), int(T)
    k = max(1, min(int(k), N))
    half = max(1, N // 2)
    span = float(ub - lb)
    table = KapurEntropyTable(prob) if fast_fitness else None

    def _fit(P, sorted_rows):
        if table is not None:
            return table.evaluate(P, assume_sorted=sorted_rows)
        return np.array([kapurs_entropy_fitness(P[i], prob) for i in range(P.shape[0])])

    # ---- Objective 2: quasi-uniform initialization ----
    X = _initial_population(rng, N, d, lb, ub, init)
    if canonical:
        X.sort(axis=1)

    fitness = _fit(X, canonical)
    best_idx = int(np.argmax(fitness))
    Xb = X[best_idx].copy()
    bF = float(fitness[best_idx])
    convergence = [bF]

    D = float(np.sqrt(d) * span)

    def _diversity(Xp, f):
        if pd_scope == "core" and N > 2:
            idx = np.argpartition(-f, half - 1)[:half]
            return _mean_pairwise_distance(Xp[idx]) / D
        return _mean_pairwise_distance(Xp) / D

    PD0 = max(_diversity(X, fitness), 1e-12)
    a = float(a0)
    z = float(z0)
    CR_max = 0.0
    last_improve = 0
    iterations_used = 0
    n_reinit_total = 0
    hist = {"a": [], "z": [], "PD": [], "CR": [], "n_reinit": []} if record_history else None
    sign = np.where(np.arange(N) < half, 1.0, -1.0)
    rank_w = np.arange(k, 0, -1, dtype=np.float64)

    for t in range(1, T + 1):
        order = np.argsort(-fitness)
        leader_idx = order[:k]
        curr_bF = float(fitness[order[0]])
        curr_wF = float(fitness[order[-1]])

        # ---- Objective 1: leader weights w_j and guidance anchor ----
        lf = fitness[leader_idx]
        if weight_mode == "rank":
            w = rank_w
        elif weight_mode == "fitness":
            w = lf if lf.sum() > 1e-12 else np.ones(k)
        elif weight_mode == "advantage":
            w = (lf - lf[-1]) + 0.1 * float(lf[0] - lf[-1]) + 1e-12
        else:
            raise ValueError(f"unknown weight_mode {weight_mode!r}")
        w = w / w.sum()
        Lk = X[leader_idx]
        if leader_mode == "centroid":
            anchor = (w @ Lk)[None, :]
        elif leader_mode == "sampled":
            # inverse-CDF sampling of one leader per agent with P(L_j) = w_j
            # (same distribution as rng.choice(k, size=N, p=w), ~4x cheaper)
            pick = np.searchsorted(np.cumsum(w), rng.random(N))
            anchor = Lk[np.minimum(pick, k - 1)]
        else:
            raise ValueError(f"unknown leader_mode {leader_mode!r}")

        # ---- SMA adaptive weight W (unchanged formula) ----
        r_w = rng.random(N)
        ratio = (curr_bF - fitness[order]) / (curr_bF - curr_wF + 1e-12) + 1
        W = np.empty(N)
        W[order] = 1 + sign * r_w * np.log(ratio)

        # ---- position update: X_guide + vb (W XA - XB) ----
        vb = rng.uniform(-a, a, size=(N, d))
        ia = rng.integers(0, N, size=N)
        ib = rng.integers(0, N, size=N)
        clash = ia == ib
        while np.any(clash):
            ib[clash] = rng.integers(0, N, size=int(clash.sum()))
            clash = ia == ib
        inner = W[:, None] * X[ia] - X[ib]
        newX = anchor + vb * inner

        # ---- Objective 3: exploration channel gated by z(t) ----
        n_re = 0
        reinit = None
        if explore_channel:
            explore = rng.random(N) < z
            explore[leader_idx] = False
            n_re = int(explore.sum())
            if n_re:
                if explore_mode == "reinit":
                    newX[explore] = rng.uniform(lb, ub, size=(n_re, d))
                    reinit = explore
                else:   # "wide": same SMA move, full oscillation amplitude
                    anc = anchor if anchor.shape[0] == 1 else anchor[explore]
                    newX[explore] = anc + rng.uniform(-a_max, a_max, size=(n_re, d)) * inner[explore]
                n_reinit_total += n_re

        np.clip(newX, lb, ub, out=newX)
        if canonical:
            newX.sort(axis=1)
        newF = _fit(newX, canonical)

        # greedy selection; randomly re-initialized agents are accepted unconditionally
        improve = newF > fitness
        if reinit is not None:
            improve |= reinit
        X[improve] = newX[improve]
        fitness[improve] = newF[improve]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > bF:
            Xb = X[gen_best].copy()
            bF = float(fitness[gen_best])
            last_improve = t
        convergence.append(bF)
        iterations_used = t

        # ---- Objective 3: feedback signals and parameter update ----
        CR = (bF - convergence[t - h]) / (abs(bF) + 1e-10) if t >= h else 0.0
        if CR > CR_max:
            CR_max = CR
        CRn = CR / CR_max if CR_max > 0 else 0.0
        PD = _diversity(X, fitness)
        PDn = PD / PD0
        stagnating = (CR <= cr_stall_eps) and (PDn < pd_low_thresh)
        if stagnating:
            z = min(z + alpha * (1.0 - PDn) * (1.0 - CRn), z_max)
            a = min(a + gamma * (1.0 - CRn), a_max)
        elif CR > 0:
            if contract_mode == "prop":
                z = max(z - beta * CRn, z_min)
                a = max(a * (1.0 - delta * CRn), a_min)
            else:
                z = max(z - beta, z_min)
                a = max(a * (1.0 - delta), a_min)
        if hist is not None:
            hist["a"].append(a); hist["z"].append(z); hist["PD"].append(PD)
            hist["CR"].append(CR); hist["n_reinit"].append(n_re)

        if early_stop and (t - last_improve) >= patience:
            break

    n_polish_eval = 0
    if local_refine:
        Xb, bF, n_polish_eval = _integer_polish(
            Xb, bF, lambda P: _fit(P, False), lb, ub)

    elapsed = time.perf_counter() - t_start
    return {
        "thresholds": sorted(int(round(v)) for v in Xb),
        "fitness": bF,
        "convergence": convergence,
        "runtime_sec": elapsed,
        "iterations_used": iterations_used,
        "n_reinitialized": n_reinit_total,
        "n_polish_evaluations": n_polish_eval,
        "history": hist,
    }