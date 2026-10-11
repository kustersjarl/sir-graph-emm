from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sir_graph.dynamics import SimulationResult


__all__ = [
    "DegreeBaseline",
    "per_node_spreading_metrics",
    "fit_degree_baseline",
]


@dataclass(frozen=True)
class DegreeBaseline:
    """
    Linear baseline for expected spreading behaviour from node degree.
    """

    slope: float
    intercept: float
    expected: NDArray[np.float64]
    residuals: NDArray[np.float64]


def per_node_spreading_metrics(
    runs: Sequence[SimulationResult],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """
    Aggregate spreading behaviour across epidemic simulations.

    Seeds are excluded because they were artificially infected.

    Returns
    -------
    n_natural_infections:
        Number of runs in which each node became naturally infected.

    mean_secondary_when_infected:
        Mean number of direct secondary infections caused by each node,
        conditional on that node being naturally infected.
    """

    if len(runs) == 0:
        raise ValueError("runs must contain at least one simulation")

    n_nodes = runs[0].n_nodes

    if any(run.n_nodes != n_nodes for run in runs):
        raise ValueError(
            "all simulation results must have the same number of nodes"
        )

    infection_times = np.stack(
        [run.infection_time for run in runs]
    )

    offspring = np.stack(
        [run.offspring for run in runs]
    )

    # > 0 means naturally infected.
    # 0 means the node was artificially selected as a seed.
    naturally_infected = infection_times > 0

    n_natural_infections = naturally_infected.sum(
        axis=0,
        dtype=np.int64,
    )

    total_secondary = np.where(
        naturally_infected,
        offspring,
        0,
    ).sum(
        axis=0,
        dtype=np.int64,
    )

    mean_secondary_when_infected = np.divide(
        total_secondary,
        n_natural_infections,
        out=np.full(
            n_nodes,
            np.nan,
            dtype=np.float64,
        ),
        where=n_natural_infections > 0,
    )

    return (
        n_natural_infections,
        mean_secondary_when_infected,
    )


def fit_degree_baseline(
    degrees: ArrayLike,
    spread: ArrayLike,
) -> DegreeBaseline:
    """
    Fit the expected spreading behaviour from node degree.

    Fits:

        expected_spread = intercept + slope * degree

    Residual:

        actual_spread - expected_spread

    Positive residual:
        node spreads more than expected for its degree.

    Negative residual:
        node spreads less than expected for its degree.
    """

    x = np.asarray(
        degrees,
        dtype=np.float64,
    )

    y = np.asarray(
        spread,
        dtype=np.float64,
    )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "degrees and spread must be 1-D arrays"
        )

    if x.shape != y.shape:
        raise ValueError(
            f"degrees and spread must have the same shape, "
            f"got {x.shape} and {y.shape}"
        )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 2:
        raise ValueError(
            "at least two valid observations are required"
        )

    if np.unique(x[valid]).size < 2:
        raise ValueError(
            "degree must contain at least two distinct values"
        )

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    expected = (
        intercept
        + slope * x
    )

    residuals = (
        y
        - expected
    )

    return DegreeBaseline(
        slope=float(slope),
        intercept=float(intercept),
        expected=expected,
        residuals=residuals,
    )