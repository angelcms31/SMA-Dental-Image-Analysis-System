"""
Tests for the /analyze/compare/ runtime fix (2026-09-13, see
ESMA_RUNTIME_OPTIMIZATION_REPORT.md): _shared_tables() lets several algorithm
runs on the same image+d+objective reuse ONE fitness table (and, for hybrid,
its Kapur table) instead of each one rebuilding its own -- this must be a
pure implementation change, i.e. _run_and_package(table=..., kapur_table=...)
must return EXACTLY the same thresholds/fitness/objective_value as calling it
with no table at all (which rebuilds internally, the pre-fix behaviour).
"""
import numpy as np
import pytest

import main
import sma_algorithms as sa

RNG = np.random.default_rng(7)


def _fake_image_and_prob(h=180, w=260, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    prob = sa.compute_histogram_prob(img)
    return img, prob


@pytest.mark.parametrize("objective", ["otsu", "hybrid"])
def test_run_and_package_matches_with_and_without_shared_table(objective):
    img, prob = _fake_image_and_prob()
    d, N, T, seed = 4, 20, 30, 42

    baseline = main._run_and_package(
        "Standard SMA", sa.standard_sma, img, prob, d, N, T, seed,
        levels="even", objective=objective,
    )

    table, kapur_table = main._shared_tables(prob, d, objective)
    shared = main._run_and_package(
        "Standard SMA", sa.standard_sma, img, prob, d, N, T, seed,
        levels="even", objective=objective, table=table, kapur_table=kapur_table,
    )

    assert shared["thresholds"] == baseline["thresholds"]
    assert shared["objective_value"] == pytest.approx(baseline["objective_value"], abs=1e-12)
    assert shared["kapur_entropy_fitness"] == pytest.approx(baseline["kapur_entropy_fitness"], abs=1e-12)
    assert shared["psnr"] == baseline["psnr"]
    assert shared["ssim"] == baseline["ssim"]


def test_shared_tables_kapur_is_a_noop():
    """objective='kapur' must not build/share anything -- _run_and_package
    should fall back to its untouched original path (see _shared_tables
    docstring)."""
    _, prob = _fake_image_and_prob()
    table, kapur_table = main._shared_tables(prob, d=4, objective="kapur")
    assert table is None
    assert kapur_table is None


def test_shared_table_is_actually_reused_across_two_algorithms():
    """The whole point: build once, use for both standard_sma and
    enhanced_sma_v2, and both must still agree with their own from-scratch
    (no shared table) results."""
    img, prob = _fake_image_and_prob(seed=2)
    d, N, T, seed = 4, 20, 30, 7
    table, kapur_table = main._shared_tables(prob, d, "hybrid")

    for algo_name, algo_fn in (("Standard SMA", sa.standard_sma), ("ESMA v2", sa.enhanced_sma_v2)):
        baseline = main._run_and_package(algo_name, algo_fn, img, prob, d, N, T, seed,
                                         levels="even", objective="hybrid")
        shared = main._run_and_package(algo_name, algo_fn, img, prob, d, N, T, seed,
                                       levels="even", objective="hybrid",
                                       table=table, kapur_table=kapur_table)
        assert shared["thresholds"] == baseline["thresholds"]
        assert shared["kapur_entropy_fitness"] == pytest.approx(baseline["kapur_entropy_fitness"], abs=1e-12)
