import numpy as np
import pytest

from sir_graph.analysis import (
    fit_degree_baseline,
    per_node_spreading_metrics,
)
from sir_graph.dynamics import SimulationResult


def make_result(
    infection_time,
    infector,
    seeds,
):
    infection_time = np.array(
        infection_time,
        dtype=np.int64,
    )

    infector = np.array(
        infector,
        dtype=np.int64,
    )

    recovery_time = np.where(
        infection_time >= 0,
        infection_time + 1,
        -1,
    )

    return SimulationResult(
        counts=np.zeros((1, 3), dtype=np.int64),
        infection_time=infection_time,
        recovery_time=recovery_time,
        infector=infector,
        seeds=np.array(seeds, dtype=np.int64),
    )


def test_per_node_spreading_metrics():
    # Run 1:
    # 0 is seed
    # 0 -> 1 -> 2
    run1 = make_result(
        infection_time=[0, 1, 2, -1],
        infector=[-1, 0, 1, -1],
        seeds=[0],
    )

    # Run 2:
    # 2 is seed
    # 2 -> 0
    run2 = make_result(
        infection_time=[1, -1, 0, -1],
        infector=[2, -1, -1, -1],
        seeds=[2],
    )

    counts, mean_secondary = (
        per_node_spreading_metrics(
            [run1, run2]
        )
    )

    np.testing.assert_array_equal(
        counts,
        [1, 1, 1, 0],
    )

    np.testing.assert_allclose(
        mean_secondary[:3],
        [0.0, 1.0, 0.0],
    )

    assert np.isnan(mean_secondary[3])


def test_seed_infections_are_excluded():
    # Node 0 causes an infection,
    # but node 0 is a seed and must not count.
    run = make_result(
        infection_time=[0, 1],
        infector=[-1, 0],
        seeds=[0],
    )

    counts, spread = (
        per_node_spreading_metrics([run])
    )

    assert counts[0] == 0
    assert np.isnan(spread[0])

    assert counts[1] == 1
    assert spread[1] == 0.0


def test_empty_runs_rejected():
    with pytest.raises(ValueError):
        per_node_spreading_metrics([])


def test_fit_degree_baseline():
    degrees = np.array(
        [1, 2, 3, 4],
        dtype=float,
    )

    spread = np.array(
        [1, 2, 3, 4],
        dtype=float,
    )

    result = fit_degree_baseline(
        degrees,
        spread,
    )

    assert result.slope == pytest.approx(1.0)
    assert result.intercept == pytest.approx(0.0)

    np.testing.assert_allclose(
        result.expected,
        spread,
    )

    np.testing.assert_allclose(
        result.residuals,
        np.zeros(4),
        atol=1e-12,
    )


def test_fit_degree_baseline_ignores_nan():
    degrees = np.array(
        [1, 2, 3, 4],
        dtype=float,
    )

    spread = np.array(
        [1, np.nan, 3, 4],
        dtype=float,
    )

    result = fit_degree_baseline(
        degrees,
        spread,
    )

    assert result.slope == pytest.approx(1.0)
    assert result.intercept == pytest.approx(0.0)
    assert np.isnan(result.residuals[1])