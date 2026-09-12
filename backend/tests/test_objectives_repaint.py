"""
Tests for the metric-oriented additions to backend/sma_algorithms.py:
  * apply_thresholds(..., levels="even") is byte-identical to the original repaint,
    levels="mean" paints every band with its own mean intensity;
  * BetweenClassVarianceTable (Otsu) equals a brute-force sigma_B^2 and the DP
    optimum over it is the exhaustive maximum (d = 2);
  * HybridObjectiveTable is the normalized weighted sum of the two tables;
  * the `objective` keyword defaults leave every optimizer's output unchanged and
    the legacy per-agent path refuses non-Kapur objectives.
"""

import os

import numpy as np
import pytest

import sma_algorithms as sa

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mendeley_reference.npz")


@pytest.fixture(scope="module")
def probs():
    fx = np.load(FIXTURE)
    counts = fx["counts"].astype(np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def _legacy_apply_thresholds(gray_image, thresholds):
    """The original implementation, verbatim (fixed evenly spaced band levels)."""
    th = sorted(int(np.clip(round(t), 0, 255)) for t in thresholds)
    bounds = [0] + th + [256]
    out = np.zeros_like(gray_image, dtype=np.uint8)
    n_bands = len(bounds) - 1
    for i in range(n_bands):
        lo, hi = bounds[i], bounds[i + 1]
        level = int(round(255 * i / max(1, n_bands - 1))) if n_bands > 1 else 255
        mask = (gray_image >= lo) & (gray_image < hi)
        out[mask] = level
    return out


def _synthetic_image(seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, 320)[None, :] + rng.normal(0, 12, size=(240, 320))
    return np.clip(x, 0, 255).astype(np.uint8)


def test_even_repaint_is_unchanged():
    img = _synthetic_image()
    for thr in ([56, 103, 149, 194], [10, 10, 200, 250], [0, 255, 128, 64], [127.5], []):
        assert np.array_equal(sa.apply_thresholds(img, thr), _legacy_apply_thresholds(img, thr))
        assert np.array_equal(sa.apply_thresholds(img, thr, "even"), _legacy_apply_thresholds(img, thr))


def test_mean_repaint_paints_band_means():
    img = _synthetic_image(1)
    thr = [40, 90, 150, 210]
    seg = sa.apply_thresholds(img, thr, levels="mean")
    vals = sa.band_levels(img, thr, "mean")
    for lo, hi, v in zip([0] + thr, thr + [256], vals):
        m = (img >= lo) & (img < hi)
        assert m.any()
        assert abs(img[m].mean() - v) <= 0.5 + 1e-9
        assert np.all(seg[m] == v)
    # minimum-MSE property: the mean repaint never has a larger MSE than the even repaint
    even = sa.apply_thresholds(img, thr, "even").astype(float)
    assert np.mean((img - seg.astype(float)) ** 2) <= np.mean((img - even) ** 2)
    with pytest.raises(ValueError):
        sa.apply_thresholds(img, thr, levels="median")


def _sigma_b(p, th):
    lv = np.arange(p.shape[0], dtype=float)
    mu = float((p * lv).sum())
    b = [0] + sorted(int(v) for v in th) + [p.shape[0]]
    s = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        P = p[lo:hi].sum()
        if P > 1e-12:
            s += P * ((p[lo:hi] * lv[lo:hi]).sum() / P - mu) ** 2
    return s


def test_otsu_table_matches_brute_force_and_dp_is_exact(probs):
    rng = np.random.default_rng(2)
    cases = list(probs[:3]) + [rng.dirichlet(np.ones(256) * 0.3)]
    for p in cases:
        tab = sa.BetweenClassVarianceTable(p)
        X = np.array([[a, b] for a in range(0, 256, 7) for b in range(a, 256, 7)], dtype=float)
        ref = np.array([_sigma_b(p, x) for x in X])
        assert np.max(np.abs(tab.evaluate(X) - ref)) < 1e-9
        h_star, thr_star = sa.kapur_optimal_thresholds(p, 2, table=tab)
        brute = max(_sigma_b(p, [a, b]) for a in range(256) for b in range(a, 256))
        assert abs(h_star - brute) < 1e-9
        assert abs(tab.evaluate_one(thr_star) - h_star) < 1e-9
        # sigma_B^2 can never exceed the total variance
        lv = np.arange(256.0)
        sigma_t2 = float((p * lv ** 2).sum() - ((p * lv).sum()) ** 2)
        assert h_star <= sigma_t2 + 1e-9


def test_hybrid_table_is_normalized_weighted_sum(probs):
    p = probs[0]
    kt, ot = sa.KapurEntropyTable(p), sa.BetweenClassVarianceTable(p)
    hk = sa.kapur_optimal_thresholds(p, 4, table=kt)[0]
    ho = sa.kapur_optimal_thresholds(p, 4, table=ot)[0]
    hyb = sa.HybridObjectiveTable(p, 4, weight=0.3, kapur_table=kt, otsu_table=ot, scale=1.0)
    assert hyb.norms == (hk, ho)
    X = np.random.default_rng(3).uniform(0, 255, size=(40, 4))
    expect = 0.3 * kt.evaluate(X) / hk + 0.7 * ot.evaluate(X) / ho
    assert np.max(np.abs(hyb.evaluate(X) - expect)) < 1e-12
    f_star = sa.optimal_thresholds(p, 4, "hybrid", hybrid_weight=0.3)[0]     # default scale: Kapur units
    assert f_star <= hk + 1e-9
    hyb_k = sa.HybridObjectiveTable(p, 4, weight=0.3, kapur_table=kt, otsu_table=ot)   # scale="kapur"
    assert np.allclose(hyb_k.evaluate(X), hk * hyb.evaluate(X))
    assert hyb_k.scale == hk
    # the pure cases collapse to the single criteria (Kapur units / normalized)
    assert np.allclose(sa.HybridObjectiveTable(p, 4, 1.0, kt, ot).evaluate(X), kt.evaluate(X))
    assert np.allclose(sa.HybridObjectiveTable(p, 4, 0.0, kt, ot, scale=1.0).evaluate(X), ot.evaluate(X) / ho)
    assert sa.make_objective_table(p, hyb) is hyb
    with pytest.raises(ValueError):
        sa.make_objective_table(p, "hybrid")          # needs d
    with pytest.raises(ValueError):
        sa.make_objective_table(p, "renyi")


def test_objective_kwarg_defaults_leave_outputs_unchanged(probs):
    p = probs[1]
    kw = dict(d=4, N=12, T=15, lb=0, ub=255, seed=7)
    for fn in (sa.standard_sma, sa.enhanced_sma, sa.enhanced_sma_v2):
        a = fn(p, **kw)
        b = fn(p, objective="kapur", **kw)
        c = fn(p, objective=sa.KapurEntropyTable(p), **kw)    # table injection
        assert a["thresholds"] == b["thresholds"] == c["thresholds"]
        assert a["fitness"] == b["fitness"] == c["fitness"]
        r = fn(p, objective="otsu", **kw)
        assert abs(r["fitness"] - sa.BetweenClassVarianceTable(p).evaluate_one(r["thresholds"])) < 1e-9
        r = fn(p, objective="hybrid", **kw)
        assert 0.0 <= r["fitness"] <= sa.kapur_optimal_thresholds(p, 4)[0] + 1e-9
    with pytest.raises(ValueError):
        sa.standard_sma(p, fast_fitness=False, objective="otsu", **kw)
