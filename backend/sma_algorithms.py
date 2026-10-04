"""
sma_algorithms.py
------------------
This file has two optimizers, both solving the same problem: find the
best set of grayscale intensity thresholds to segment a dental OPG
image, using Kapur's entropy as the score to maximize.

  1. standard_sma  -- the original Slime Mould Algorithm (Li et al.,
                       2020), unmodified. This is the baseline we
                       compare against.

  2. enhanced_sma  -- the proposed Enhanced Slime Mould Algorithm
                       (ESMA). It carries out the three proposed
                       modifications:
                         Objective 1: Fitness-Weighted Multi-Leader
                                      Guidance
                         Objective 2: Quasi-Uniform Initialization
                         Objective 3: Performance-Feedback Adaptive
                                      Control (using CR and PD)

Two shared helpers support both optimizers:

  - KapurEntropyTable: precomputes the entropy of every possible
    threshold class once per image, so scoring a whole population of
    candidate solutions is one fast, vectorized lookup instead of a
    slow per-agent loop. It returns exactly the same numbers as the
    plain kapurs_entropy_fitness() function -- this is purely a speed
    optimization (about 9x faster), not a change to what is being
    computed. Both optimizers can use it (fast_fitness=True by
    default; pass False to use the original, slower loop instead --
    both give the same results, checked in tests/).

  - kapur_optimal_thresholds: computes the EXACT best possible
    thresholds for a given image using dynamic programming, rather
    than search. This is used only to grade how close a run of
    standard_sma / enhanced_sma came to the true best answer
    ("optimality gap") -- it is never used by the optimizers
    themselves while they search, since that would defeat the purpose
    of testing them.

Both optimizer functions return the same shape of result: the
best threshold vector found, its Kapur entropy score, and the
convergence curve (the best score after each iteration) -- everything
needed for the Chapter 4 comparison tables and convergence plots.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Shared: Kapur's entropy fitness for multilevel thresholding
# ---------------------------------------------------------------------------

def compute_histogram_prob(gray_image: np.ndarray) -> np.ndarray:
    """Turns a grayscale image into a 256-bin histogram of pixel intensities,
    normalized so the bins sum to 1 (i.e. a probability distribution)."""
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
    Removes solid black or white padding borders from around an OPG image
    before it is used for anything (histogram, optimization, or overlay).

    Why this matters: a plain, uniform border -- whether black or white --
    is not real anatomical content, but it still takes up a big share of
    the pixel count at one intensity extreme. Left in, it skews Kapur's
    entropy toward separating "border vs. content" instead of separating
    actual dental structures, and it can also fool brightness-based
    overlay logic into thinking the border is the most relevant region.

    A row or column is only treated as padding if it is BOTH very
    uniform (low pixel variation -- real tissue always has texture) AND
    very close to solid black or solid white. This avoids accidentally
    cropping a genuinely bright but detailed row of the image. If no
    clear border is found, the original image is returned unchanged.
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
    Scores a set of threshold values using Kapur's entropy. Higher is
    better -- this is what the optimizers are trying to maximize.

    How it works: the thresholds split the image's intensity histogram
    into separate classes (e.g. background, enamel, dentin, a lesion).
    Each class gets its own "entropy" value based on how its pixels are
    distributed; the total score is the sum of every class's entropy.
    A higher total means the classes are, overall, more informative and
    better separated from each other.
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
# Shared speed/reference helpers (these don't change what is being
# computed -- just how fast it's computed, or how we grade the result)
#   * KapurEntropyTable        -- fast, vectorized scoring for a whole
#                                 population of candidate solutions at once
#   * kapur_optimal_thresholds -- the exact best possible answer, for
#                                 grading how close a run got
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
    A lookup table of the entropy of every possible threshold class for
    one image, computed once so scoring any threshold vector afterward
    is fast.

    Why this works: Kapur's total score is just the sum of each class's
    own entropy, and a class is fully described by where it starts and
    ends (lo, hi). So instead of recomputing entropy from scratch every
    time we score a candidate solution, we can precompute the entropy of
    every possible (lo, hi) range once (there are only 256x256 of them),
    store it in a table, and then scoring any threshold vector is just a
    few quick lookups and a sum. This gives the exact same numbers as
    kapurs_entropy_fitness() (checked in tests/test_sma_algorithms.py,
    agreement to about 14 decimal places) -- it's simply a much faster
    way to compute the same thing, which matters because the optimizers
    need to score an entire population of candidates on every iteration.
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
        """Scores every row of X (shape (N, d), one threshold vector per
        row) at once, returning an array of N scores. Gives the same
        result as calling kapurs_entropy_fitness() on each row."""
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
    """Scores every row of X. Uses the fast lookup table when available,
    otherwise falls back to scoring each row one at a time."""
    if table is not None:
        return table.evaluate(X)
    return np.array([kapurs_entropy_fitness(X[i], prob) for i in range(X.shape[0])])


def kapur_optimal_thresholds(prob, d, table=None):
    """
    Finds the EXACT best possible set of d thresholds for this image --
    not an approximation, the true mathematical best -- using dynamic
    programming instead of search.

    This works because Kapur's score is a sum of independent per-class
    entropies, so the best way to split the histogram into d+1 classes
    can be built up one threshold at a time: the best score using j
    thresholds ending at position t is the best score using j-1
    thresholds ending anywhere before t, plus the entropy of the new
    final class. Solving this way takes only a few milliseconds.

    This function exists purely to grade the optimizers: it tells us
    the "optimality gap" (how far a run's answer was from the true best)
    and "% of images where the optimizer found the true best answer".
    It plays the same role that a known correct answer plays when
    grading any search algorithm -- it is never called by
    standard_sma or enhanced_sma while they are
    searching, only afterward, to check their work.

    Returns (best_score, thresholds) with thresholds sorted ascending.
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
    """Uses a set of thresholds to split the image into separate
    intensity bands, so the result can be viewed as a segmented image."""
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

def standard_sma(prob, d, N=30, T=100, lb=0, ub=255, seed=None, fast_fitness=True,
                 record_positions=False):
    """
    The original Slime Mould Algorithm, unmodified. Every agent follows
    a single global best-known solution, the starting population is
    random and uniform, and the search parameters follow a fixed
    schedule that only depends on the iteration number -- not on how
    well the search is actually going. This is the baseline that ESMA
    is compared against throughout Chapter 4.

    fast_fitness: when True (default), uses the fast lookup-table
    scoring method instead of scoring each agent one at a time. Both
    give the same results -- this only changes how fast the function
    runs, not what it computes. Set to False to use the original,
    slower per-agent scoring method instead.
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

    # optional trace for the simulation view (does not touch the search:
    # nothing here draws from rng or changes X / fitness)
    pos_hist = [X.copy()] if record_positions else None
    lead_hist = [[best_idx]] if record_positions else None
    a_hist = [1.0] if record_positions else None

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

        # each agent does ONE of three things this round, chosen by
        # chance: explore randomly, move toward the single best agent,
        # or just decay in place
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

        # keep whichever is better, old position or new position
        improve = newFitness > fitness
        X[improve] = newX[improve]
        fitness[improve] = newFitness[improve]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > bF:
            Xb = X[gen_best].copy()
            bF = float(fitness[gen_best])

        convergence.append(bF)
        if record_positions:
            pos_hist.append(X.copy())
            lead_hist.append([gen_best])
            a_hist.append(float(a))

    elapsed = time.perf_counter() - t_start
    out = {
        "thresholds": sorted(int(round(v)) for v in Xb),
        "fitness": bF,
        "convergence": convergence,
        "runtime_sec": elapsed,
    }
    if record_positions:
        out["positions"] = pos_hist
        out["leaders"] = lead_hist
        out["a_trace"] = a_hist
    return out



# ---------------------------------------------------------------------------
# 2. ENHANCED SMA -- the proposed algorithm
# ---------------------------------------------------------------------------

def _mean_pairwise_distance(X):
    """Average distance between every pair of agents in the population --
    a simple way to measure how spread out (diverse) the population is."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs)).sum() / (n * (n - 1)))


def _integer_polish(x, f, evaluate, lb, ub, max_rounds=64):
    """
    Optional final touch-up (local_refine=True): after the search ends,
    try nudging each threshold up or down by 1 and keep any change that
    improves the score, repeating until nothing improves anymore. This
    is a simple hill-climb on top of the main search, off by default,
    and reported separately from the core algorithm's own results.
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
    Builds the starting population of agents (Objective 2).
      'lhs'    : Latin Hypercube Sampling -- each dimension (each
                 threshold position) is split into N equal strata, and
                 every agent gets exactly one stratum per dimension,
                 with the assignment shuffled independently per
                 dimension. This guarantees the population is spread
                 out across the full range in every dimension, and
                 across different combinations of thresholds.
      'strata' : a simpler stratified approach where every agent uses
                 the SAME stratum index in every dimension, so the
                 whole population ends up clustered near the diagonal
                 (all thresholds close to each other in value).
      'uniform': plain random placement, same as standard_sma uses.
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


def enhanced_sma(
    prob, d, N=30, T=100, lb=0, ub=255, seed=None,
    # ---- Objective 1: fitness-weighted multi-leader guidance ----
    k=5,                       # how many top agents act as leaders
    leader_mode="sampled",     # "sampled" : each agent follows ONE leader, chosen at
                               #             random with probability equal to that
                               #             leader's weight
                               # "centroid": each agent follows the weighted average
                               #             position of all k leaders at once
    weight_mode="rank",        # "rank": better-ranked leaders get more influence
                               # "fitness": leaders are weighted directly by their score
                               # "advantage": leaders are weighted by how much better
                               #              they are than the weakest leader
    # ---- Objective 2: quasi-uniform initialization ----
    init="lhs",                # "lhs": Latin Hypercube strata (recommended)
                               # "strata": simpler stratified placement
                               # "uniform": plain random placement
    canonical=True,            # keep each agent's own thresholds sorted smallest-to-largest
    # ---- Objective 3: performance-feedback adaptive control ----
    h=5,                       # how many recent iterations "progress" looks back over
    a0=1.0, a_min=0.05, a_max=1.0,      # movement size a(t): starting value, floor, ceiling
    z0=0.03, z_min=0.01, z_max=0.10,    # exploration chance z(t): starting value, allowed range
    alpha=0.10, beta=0.10, gamma=0.10, delta=0.10,   # how fast a(t)/z(t) adjust
    cr_stall_eps=1e-6, pd_low_thresh=0.10,           # thresholds for "no progress" / "low diversity"
    contract_mode="prop",      # "prop": shrink movement size in proportion to how much
                               #         progress is being made | "geom": shrink by a fixed step
    pd_scope="core",           # "core": measure diversity over the fitter half of the
                               #         population | "all": measure over everyone
    explore_channel=False,     # optional: let z(t) trigger an extra exploration move
                               # for some agents (kept off by default -- testing showed
                               # it does not help for this problem; see notes below)
    explore_mode="wide",       # if explore_channel is on: "wide" takes a bigger normal
                               # move, "reinit" places the agent at a fresh random position
    # ---- extra options ----
    early_stop=True, patience=20,   # stop automatically once the best score hasn't
                                     # improved for `patience` iterations in a row
    local_refine=False,        # optional final +/-1 touch-up pass (see _integer_polish)
    fast_fitness=True,         # use the fast lookup-table scoring method
    record_history=False,      # also return the a(t)/z(t)/PD/CR values from every iteration
    record_positions=False,    # also return every agent's position at every iteration
                               # (used only by the simulation view; does not affect the search)
):
    """
    The Enhanced Slime Mould Algorithm (ESMA), implementing the three
    proposed modifications described in Chapter 3:

    Objective 1 -- Fitness-Weighted Multi-Leader Guidance
      Instead of every agent following one single best-known solution,
      each agent is guided by a weighted blend of the top-k
      best-performing agents. Leaders are weighted so that the
      strongest leaders have more influence, using their RANK rather
      than their raw score -- because in practice, the raw Kapur
      entropy scores of competing leaders are usually very close to
      each other, so weighting by rank differentiates them much more
      meaningfully than weighting by raw score does.

    Objective 2 -- Quasi-Uniform Initialization
      The starting population is placed using Latin Hypercube Sampling:
      every dimension (threshold position) is divided into N strata,
      and each agent gets one stratum per dimension, assigned
      independently for each dimension. This guarantees the starting
      population spreads out across the whole search space rather than
      clustering together. Each agent's own thresholds are also kept
      sorted, since the score only depends on the sorted values -- this
      keeps the leader-blending and movement calculations meaningful,
      since every agent's coordinates line up in the same order.

    Objective 3 -- Performance-Feedback Adaptive Control
      Two signals are measured every iteration: CR (how much the best
      score has improved recently) and PD (how spread out the
      population still is). The movement size a(t) shrinks when the
      search is making good progress (to fine-tune the current best
      area) and grows when the search has stalled (to explore more
      broadly again), instead of following a fixed schedule regardless
      of how the search is actually going.

    Extra practical features
      - early_stop: once the best score has gone `patience` iterations
        without improving, the search stops automatically instead of
        continuing to the full iteration budget. This is why ESMA runs
        noticeably faster than the standard algorithm in the results,
        without sacrificing solution quality.
      - local_refine: an optional last-mile polish step that nudges the
        final answer by +/-1 per threshold to catch any easy remaining
        improvement.
      - explore_channel: an optional mechanism where z(t) can trigger
        some agents to take a bigger exploratory move. It is off by
        default because testing on this problem showed it doesn't
        improve results here -- but it is still fully implemented and
        can be switched on.

    Returns the same fields as standard_sma / enhanced_sma (thresholds,
    fitness, convergence, runtime_sec), plus `iterations_used`,
    `n_reinitialized`, `n_polish_evaluations`, and optionally `history`
    (the per-iteration a/z/PD/CR values, useful for Chapter 4 figures).

    How the default parameter values were chosen: tested on a
    development set of 40 images (kept separate from the 300 images
    used for the reported results), across multiple random seeds,
    scoring each configuration by how close it got to the true best
    answer (from kapur_optimal_thresholds). Sorting each agent's
    thresholds (canonical=True) and having agents follow one sampled
    leader (leader_mode="sampled") made the biggest difference. A
    gentle, proportional shrinking of the movement size worked better
    than an aggressive one. k=5 leaders with a 5-iteration look-back
    window (h=5) gave the best balance of accuracy and consistency.
    Stopping after 20 non-improving iterations (patience=20) keeps
    almost all of the accuracy of a full run while using noticeably
    fewer iterations on average.
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
    init_unsorted = X.copy() if record_positions else None
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

    # optional trace for the simulation view (does not touch the search)
    pos_hist = [X.copy()] if record_positions else None
    lead_hist = [np.argsort(-fitness)[:k].tolist()] if record_positions else None
    a_hist = [a] if record_positions else None

    for t in range(1, T + 1):
        # --- rank the population, pick the top-k leaders ---
        order = np.argsort(-fitness)
        leader_idx = order[:k]
        curr_bF = float(fitness[order[0]])
        curr_wF = float(fitness[order[-1]])

        # ---- Objective 1: leader weights and guidance target ----
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
            # each agent draws ONE leader at random, with probability
            # equal to that leader's weight (cheaper than, but
            # equivalent to, rng.choice(k, size=N, p=w))
            pick = np.searchsorted(np.cumsum(w), rng.random(N))
            anchor = Lk[np.minimum(pick, k - 1)]
        else:
            raise ValueError(f"unknown leader_mode {leader_mode!r}")

        # ---- standard SMA-style adaptive weight W (unchanged formula) ----
        r_w = rng.random(N)
        ratio = (curr_bF - fitness[order]) / (curr_bF - curr_wF + 1e-12) + 1
        W = np.empty(N)
        W[order] = 1 + sign * r_w * np.log(ratio)

        # ---- move every agent toward its guidance target ----
        vb = rng.uniform(-a, a, size=(N, d))
        ia = rng.integers(0, N, size=N)
        ib = rng.integers(0, N, size=N)
        clash = ia == ib
        while np.any(clash):
            ib[clash] = rng.integers(0, N, size=int(clash.sum()))
            clash = ia == ib
        inner = W[:, None] * X[ia] - X[ib]
        newX = anchor + vb * inner

        # ---- Objective 3 (optional): z(t)-triggered exploration for a few agents ----
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
                else:   # "wide": bigger normal move, full movement size
                    anc = anchor if anchor.shape[0] == 1 else anchor[explore]
                    newX[explore] = anc + rng.uniform(-a_max, a_max, size=(n_re, d)) * inner[explore]
                n_reinit_total += n_re

        np.clip(newX, lb, ub, out=newX)
        if canonical:
            newX.sort(axis=1)
        newF = _fit(newX, canonical)

        # keep whichever is better, old position or new; agents that were
        # freshly re-initialized this round are always kept, so a fresh
        # random restart has a chance to survive and be explored further
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

        # ---- Objective 3: measure progress (CR) and diversity (PD), then adjust a(t)/z(t) ----
        CR = (bF - convergence[t - h]) / (abs(bF) + 1e-10) if t >= h else 0.0
        if CR > CR_max:
            CR_max = CR
        CRn = CR / CR_max if CR_max > 0 else 0.0
        PD = _diversity(X, fitness)
        PDn = PD / PD0
        stagnating = (CR <= cr_stall_eps) and (PDn < pd_low_thresh)
        if stagnating:
            # not making progress -- widen the search
            z = min(z + alpha * (1.0 - PDn) * (1.0 - CRn), z_max)
            a = min(a + gamma * (1.0 - CRn), a_max)
        elif CR > 0:
            # making progress -- narrow the search to fine-tune
            if contract_mode == "prop":
                z = max(z - beta * CRn, z_min)
                a = max(a * (1.0 - delta * CRn), a_min)
            else:
                z = max(z - beta, z_min)
                a = max(a * (1.0 - delta), a_min)
        if hist is not None:
            hist["a"].append(a); hist["z"].append(z); hist["PD"].append(PD)
            hist["CR"].append(CR); hist["n_reinit"].append(n_re)
        if record_positions:
            pos_hist.append(X.copy())
            lead_hist.append(np.argsort(-fitness)[:k].tolist())
            a_hist.append(float(a))

        # stop early once the best score has stalled for `patience` iterations
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
        "positions": pos_hist,
        "leaders": lead_hist,
        "a_trace": a_hist,
        "init_unsorted": init_unsorted,
    }