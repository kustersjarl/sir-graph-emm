import numpy as np
import pytest

from sir_graph.analysis import (
    fit_model_a_hazard,
    model_a_observations,
)
from sir_graph.dynamics import (
    SimulationResult,
)
from sir_graph.graph import Graph


def test_model_a_weighted_exposure() -> None:
    graph = Graph.from_edges(
        2,
        np.array(
            [
                [0, 1],
            ],
            dtype=np.int64,
        ),
    )

    # Node 0 is a seed and is infectious
    # during t = 0, 1, 2.
    #
    # Node 1 becomes infected at time 2.
    #
    # Therefore node 1:
    #
    # t = 0: exposed and escapes
    # t = 1: exposed and becomes infected
    #
    # Raw:
    # escape exposure    = 1
    # infection exposure = 1
    #
    # Weighted source strength:
    # infectiousness 1.5 * edge multiplier 2
    # = 3 per contact-step.

    run = SimulationResult(
        counts=np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
            dtype=np.int64,
        ),
        infection_time=np.array(
            [0, 2],
            dtype=np.int64,
        ),
        recovery_time=np.array(
            [3, 3],
            dtype=np.int64,
        ),
        infector=np.array(
            [-1, 0],
            dtype=np.int64,
        ),
        seeds=np.array(
            [0],
            dtype=np.int64,
        ),
    )

    infectiousness = np.array(
        [1.5, 1.0],
        dtype=np.float64,
    )

    edge_multipliers = np.array(
        [2.0, 2.0],
        dtype=np.float64,
    )

    observations = model_a_observations(
        graph,
        [run],
        infectiousness=
            infectiousness,
        edge_multipliers=
            edge_multipliers,
    )

    # Seed node 0 is excluded.
    assert observations.node_index.tolist() == [
        1
    ]

    assert observations.infected.tolist() == [
        True
    ]

    assert (
        observations
        .raw_escape_exposure[0]
        == 1
    )

    assert (
        observations
        .raw_infection_exposure[0]
        == 1
    )

    assert (
        observations
        .weighted_escape_exposure[0]
        == pytest.approx(3.0)
    )

    assert (
        observations
        .weighted_infection_exposure[0]
        == pytest.approx(3.0)
    )


def test_fit_model_a_hazard_known_solution() -> None:
    # One infection:
    #
    # escaped exposure = 1
    # infection-step exposure = 1
    #
    # Therefore:
    #
    # l(lambda)
    #   = -lambda
    #     + log(1 - exp(-lambda))
    #
    # Setting the derivative to zero:
    #
    #   -1 + 1 / (exp(lambda) - 1) = 0
    #
    # gives:
    #
    #   lambda = log(2)

    infected = np.array(
        [
            True,
            False,
        ],
        dtype=np.bool_,
    )

    escape_exposure = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )

    infection_exposure = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )

    lambda_hat = fit_model_a_hazard(
        infected,
        escape_exposure,
        infection_exposure,
    )

    assert lambda_hat is not None

    assert lambda_hat == pytest.approx(
        np.log(2.0),
        rel=1e-10,
    )


def test_fit_model_a_hazard_no_infections() -> None:
    infected = np.array(
        [
            False,
            False,
        ],
        dtype=np.bool_,
    )

    escape_exposure = np.array(
        [
            2.0,
            3.0,
        ],
        dtype=np.float64,
    )

    infection_exposure = np.array(
        [
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    lambda_hat = fit_model_a_hazard(
        infected,
        escape_exposure,
        infection_exposure,
    )

    assert lambda_hat == 0.0