"""
Tests for backend/sma_algorithms.py (run from the repo or backend/ with
`python -m pytest backend/tests -q` or `python -m pytest tests -q`).

fixtures/mendeley_reference.npz holds, for the 100 local Mendeley OPG images,
the 256-bin histogram counts after autocrop_black_borders() plus the
thresholds / fitness the ORIGINAL (pre-v2) standard_sma / enhanced_sma
produced at N=30, T=50, d=4, seed=42. It pins the legacy behaviour.
"""

import os

import numpy as np
import pytest

import sma_algorithms as sa

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mendeley_reference.npz")


@pytest.fixture(scope="module")
def ref():
    fx = np.load(FIXTURE)
    counts = fx["counts"].astype(np.float64)
    return {
        "probs": counts / counts.sum(axis=1, keepdims=True),
        "sma_thr": fx["sma_thresholds"], "sma_fit": fx["sma_fitness"],
        "esma_thr": fx["esma_thresholds"], "esma_fit": fx["esma_fitness"],
        "params": [int(v) for v in fx["params"]],   # N, T, d, seed
    }


def synthetic_histograms():
    rng = np.random.default_rng(123)
    hs = []
    hs.append(np.full(256, 1 / 256))                                # uniform
    p = np.zeros(256); p[[10, 60, 200]] = [0.2, 0.5, 0.3]; hs.append(p)  # three spikes, many empty bins
    p = rng.random(256); p[100:180] = 0; hs.append(p / p.sum())      # interior gap
    p = np.zeros(256); p[0] = 1.0; hs.append(p)                      # degenerate single bin
    x = np.arange(256); p = np.exp(-0.5 * ((x - 90) / 20) ** 2) + 0.6 * np.exp(-0.5 * ((x - 180) / 15) ** 2)
    hs.append(p / p.sum())                                           # bimodal
    return hs


# --------------------------------------------------------------------------
# N1: table fitness == scalar fitness, DP optimum is the global optimum
# --------------------------------------------------------------------------

def test_table_matches_scalar_fitness(ref):
    rng = np.random.default_rng(0)
    worst = 0.0
    for prob in list(ref["probs"][:15]) + synthetic_histograms():
        table = sa.KapurEntropyTable(prob)
        for d in (1, 2, 4, 6):
            X = np.concatenate([
                rng.uniform(-5, 260, size=(60, d)),                       # out of range -> clipped
                rng.integers(0, 256, size=(30, d)).astype(float),
                np.repeat(rng.integers(0, 256, size=(10, 1)), d, axis=1).astype(float),  # duplicates
                np.full((1, d), 0.0), np.full((1, d), 255.0), np.full((1, d), 127.5),    # half -> even
            ])
            ref_vals = np.array([sa.kapurs_entropy_fitness(x, prob) for x in X])
            got = table.evaluate(X)
            worst = max(worst, float(np.max(np.abs(ref_vals - got))))
            assert got.shape == (X.shape[0],)
    assert worst < 1e-9, worst


def test_table_evaluate_one_and_sorted_flag(ref):
    prob = ref["probs"][0]
    table = sa.KapurEntropyTable(prob)
    thr = [200.0, 30.0, 120.0, 75.0]
    assert abs(table.evaluate_one(thr) - sa.kapurs_entropy_fitness(thr, prob)) < 1e-9
    srt = np.sort(np.array([thr]), axis=1)
    assert abs(table.evaluate(srt, assume_sorted=True)[0] - table.evaluate_one(thr)) < 1e-12


def test_dp_optimum_is_global(ref):
    rng = np.random.default_rng(1)
    N, T, d, seed = ref["params"]
    for prob in list(ref["probs"][:20]) + synthetic_histograms():
        table = sa.KapurEntropyTable(prob)
        for dd in (1, 2, 4):
            h_star, thr = sa.kapur_optimal_thresholds(prob, dd, table=table)
            assert len(thr) == dd and thr == sorted(thr)
            assert all(0 <= t <= 255 for t in thr)
            assert abs(table.evaluate_one(thr) - h_star) < 1e-9
            assert table.evaluate(rng.uniform(0, 255, size=(3000, dd))).max() <= h_star + 1e-9
            if dd == 2:   # exhaustive check is cheap for d = 2
                grid = np.array([(a, b) for a in range(256) for b in range(a, 256)], dtype=float)
                assert abs(table.evaluate(grid).max() - h_star) < 1e-9
    # every stored optimizer result is bounded by the optimum
    for i, prob in enumerate(ref["probs"]):
        h_star, _ = sa.kapur_optimal_thresholds(prob, d)
        assert h_star + 1e-9 >= ref["sma_fit"][i]
        assert h_star + 1e-9 >= ref["esma_fit"][i]


def test_histogram_prob_bincount_matches_histogram():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(300, 400)).astype(np.uint8)
    img[:50] = 0; img[-50:] = 255
    fast = sa.compute_histogram_prob(img)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    assert np.allclose(fast, hist / hist.sum(), rtol=0, atol=1e-15)
    assert abs(fast.sum() - 1.0) < 1e-12


# --------------------------------------------------------------------------
# Legacy algorithms: fast path == legacy path == frozen reference
# --------------------------------------------------------------------------

def test_legacy_algorithms_unchanged_against_frozen_reference(ref):
    N, T, d, seed = ref["params"]
    n_same_s = n_same_e = 0
    for i, prob in enumerate(ref["probs"]):
        r1 = sa.standard_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)     # fast path (default)
        r2 = sa.enhanced_sma(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        n_same_s += r1["thresholds"] == list(ref["sma_thr"][i])
        n_same_e += r2["thresholds"] == list(ref["esma_thr"][i])
        assert abs(r1["fitness"] - ref["sma_fit"][i]) < 1e-9
        assert abs(r2["fitness"] - ref["esma_fit"][i]) < 1e-9
    # identical trajectories on every stored image (ties at float precision would show up here)
    assert n_same_s == len(ref["probs"]) and n_same_e == len(ref["probs"])


def test_fast_and_legacy_fitness_paths_agree(ref):
    N, T, d, seed = ref["params"]
    for prob in ref["probs"][:4]:
        for fn in (sa.standard_sma, sa.enhanced_sma):
            slow = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, fast_fitness=False)
            fast = fn(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, fast_fitness=True)
            assert slow["thresholds"] == fast["thresholds"]
            assert abs(slow["fitness"] - fast["fitness"]) < 1e-9
            assert np.allclose(slow["convergence"], fast["convergence"], atol=1e-9)
        v_slow = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, fast_fitness=False)
        v_fast = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, fast_fitness=True)
        assert v_slow["thresholds"] == v_fast["thresholds"]
        assert abs(v_slow["fitness"] - v_fast["fitness"]) < 1e-9


# --------------------------------------------------------------------------
# ESMA v2
# --------------------------------------------------------------------------

def _check_result(r, prob, d, T):
    assert set(["thresholds", "fitness", "convergence", "runtime_sec", "iterations_used"]) <= set(r)
    thr = r["thresholds"]
    assert len(thr) == d and thr == sorted(thr)
    assert all(isinstance(t, int) and 0 <= t <= 255 for t in thr)
    assert abs(sa.kapurs_entropy_fitness(thr, prob) - r["fitness"]) < 1e-9   # reported fitness is real
    conv = np.array(r["convergence"])
    assert len(conv) == r["iterations_used"] + 1
    assert np.all(np.diff(conv) >= 0)                                        # best-so-far is monotone
    assert abs(conv[-1] - r["fitness"]) < 1e-9
    assert 1 <= r["iterations_used"] <= T
    assert r["runtime_sec"] > 0


def test_v2_output_schema_validity_and_determinism(ref):
    N, T, d, seed = ref["params"]
    for prob in ref["probs"][:8]:
        r = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        _check_result(r, prob, d, T)
        r_again = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed)
        assert r_again["thresholds"] == r["thresholds"] and r_again["fitness"] == r["fitness"]
        full = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, lb=0, ub=255, seed=seed, early_stop=False)
        _check_result(full, prob, d, T)
        assert full["iterations_used"] == T


def test_v2_all_component_switches_run(ref):
    N, T, d, seed = ref["params"]
    prob = ref["probs"][3]
    variants = [
        dict(canonical=False), dict(init="strata"), dict(init="uniform"), dict(leader_mode="centroid"),
        dict(weight_mode="fitness"), dict(weight_mode="advantage"), dict(explore_channel=False),
        dict(contract_mode="geom"), dict(pd_scope="all"), dict(k=1), dict(k=5), dict(k=100),
        dict(local_refine=True), dict(record_history=True), dict(delta=0.0, gamma=0.0),
    ]
    for kw in variants:
        r = sa.enhanced_sma_v2(prob, d=d, N=N, T=20, lb=0, ub=255, seed=seed, **kw)
        _check_result(r, prob, d, 20)
    hist = sa.enhanced_sma_v2(prob, d=d, N=N, T=20, seed=seed, early_stop=False, record_history=True)["history"]
    assert all(len(hist[key]) == 20 for key in ("a", "z", "PD", "CR", "n_reinit"))
    assert all(0.01 - 1e-12 <= z <= 0.30 + 1e-12 for z in hist["z"])
    assert all(0.02 - 1e-12 <= a <= 2.5 + 1e-12 for a in hist["a"])
    for d_small in (1, 2, 3):
        _check_result(sa.enhanced_sma_v2(prob, d=d_small, N=12, T=15, seed=seed), prob, d_small, 15)
    with pytest.raises(ValueError):
        sa.enhanced_sma_v2(prob, d=d, N=N, T=5, seed=seed, init="bogus")


def test_v2_local_polish_never_decreases_fitness(ref):
    N, T, d, seed = ref["params"]
    for prob in ref["probs"][:10]:
        base = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, seed=seed, early_stop=False)
        polished = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, seed=seed, early_stop=False, local_refine=True)
        assert polished["fitness"] >= base["fitness"] - 1e-12
        h_star, _ = sa.kapur_optimal_thresholds(prob, d)
        assert polished["fitness"] <= h_star + 1e-9


def test_initialization_strata_properties():
    rng = np.random.default_rng(5)
    N, d, lb, ub = 30, 4, 0, 255
    X = sa._initial_population(rng, N, d, lb, ub, "lhs")
    assert X.shape == (N, d) and X.min() >= lb and X.max() <= ub
    strata = np.floor((X - lb) / (ub - lb) * N).astype(int)
    for j in range(d):        # Latin hypercube: every stratum holds exactly one agent, per dimension
        assert sorted(strata[:, j].tolist()) == list(range(N))
    Xs = sa._initial_population(rng, N, d, lb, ub, "strata")
    strata_s = np.floor((Xs - lb) / (ub - lb) * N).astype(int)
    assert np.all(strata_s == np.arange(N)[:, None])      # thesis-literal: same stratum in every dimension
    Xu = sa._initial_population(rng, N, d, lb, ub, "uniform")
    assert Xu.shape == (N, d)


def test_v2_beats_or_matches_v1_on_reference_set(ref):
    """Sanity guard on the improvement claim (mean optimality gap over the stored images)."""
    N, T, d, seed = ref["params"]
    gaps_v2, gaps_v1 = [], []
    for i, prob in enumerate(ref["probs"][:40]):
        h_star, _ = sa.kapur_optimal_thresholds(prob, d)
        r = sa.enhanced_sma_v2(prob, d=d, N=N, T=T, seed=seed, early_stop=False)
        gaps_v2.append(h_star - r["fitness"])
        gaps_v1.append(h_star - ref["esma_fit"][i])
    assert np.mean(gaps_v2) <= np.mean(gaps_v1) + 1e-12