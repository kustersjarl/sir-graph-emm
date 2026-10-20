import numpy as np
import pytest

from sir_graph.analysis import (
    descendant_counts,
    model_b_observations,
    opportunity_contact_steps,
    weighted_opportunity_contact_steps,
)
from sir_graph.dynamics import SimulationResult
from sir_graph.graph import Graph


def make_graph(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> Graph:
    return Graph.from_edges(
        n_nodes,
        np.asarray(
            edges,
            dtype=np.int64,
        ).reshape(-1, 2),
    )


def make_simple_run() -> SimulationResult:
    """
    Graph:

        0 -- 1 -- 2

    Transmission:

        0 -> 1

    Node 0:
        infected at t=0
        recovers at t=2

    Node 1:
        infected at t=1
        recovers at t=3

    Node 2:
        never infected
    """

    return SimulationResult(
        counts=np.array(
            [
                [2, 1, 0],
                [1, 2, 0],
                [1, 1, 1],
                [1, 0, 2],
            ],
            dtype=np.int64,
        ),
        infection_time=np.array(
            [0, 1, -1],
            dtype=np.int64,
        ),
        recovery_time=np.array(
            [2, 3, -1],
            dtype=np.int64,
        ),
        infector=np.array(
            [-1, 0, -1],
            dtype=np.int64,
        ),
        seeds=np.array(
            [0],
            dtype=np.int64,
        ),
    )


def test_opportunity_contact_steps() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    opportunity = opportunity_contact_steps(
        graph,
        run,
    )

    # Node 0:
    # node 1 is susceptible only during t=0.
    #
    # Node 1:
    # node 2 remains susceptible during t=1 and t=2.
    #
    # Node 2:
    # never infected.
    expected = np.array(
        [1, 2, 0],
        dtype=np.int64,
    )

    assert np.array_equal(
        opportunity,
        expected,
    )


def test_weighted_opportunity_contact_steps() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    susceptibility = np.array(
        [
            0.20,
            0.50,
            0.25,
        ],
        dtype=np.float64,
    )

    weighted = (
        weighted_opportunity_contact_steps(
            graph,
            run,
            susceptibility,
        )
    )

    # Node 0:
    # one contact-step with node 1:
    #
    #     1 * 0.50 = 0.50
    #
    # Node 1:
    # two contact-steps with node 2:
    #
    #     2 * 0.25 = 0.50
    #
    # Node 2:
    # never infectious.
    expected = np.array(
        [
            0.50,
            0.50,
            0.00,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        weighted,
        expected,
    )


def test_weighted_equals_raw_when_susceptibility_is_one() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    raw = opportunity_contact_steps(
        graph,
        run,
    )

    weighted = (
        weighted_opportunity_contact_steps(
            graph,
            run,
            np.ones(
                3,
                dtype=np.float64,
            ),
        )
    )

    np.testing.assert_allclose(
        weighted,
        raw.astype(np.float64),
    )


def test_weighted_opportunity_rejects_invalid_susceptibility() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    with pytest.raises(ValueError):
        weighted_opportunity_contact_steps(
            graph,
            run,
            np.array(
                [0.5, 0.5],
                dtype=np.float64,
            ),
        )

    with pytest.raises(ValueError):
        weighted_opportunity_contact_steps(
            graph,
            run,
            np.array(
                [
                    0.5,
                    -0.2,
                    0.5,
                ],
                dtype=np.float64,
            ),
        )


def test_descendant_counts_chain() -> None:
    """
    Transmission tree:

        0 -> 1 -> 2 -> 3

    Descendants:

        0: 3
        1: 2
        2: 1
        3: 0
    """

    run = SimulationResult(
        counts=np.zeros(
            (6, 3),
            dtype=np.int64,
        ),
        infection_time=np.array(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        recovery_time=np.array(
            [2, 3, 4, 5],
            dtype=np.int64,
        ),
        infector=np.array(
            [-1, 0, 1, 2],
            dtype=np.int64,
        ),
        seeds=np.array(
            [0],
            dtype=np.int64,
        ),
    )

    descendants = descendant_counts(
        run
    )

    expected = np.array(
        [3, 2, 1, 0],
        dtype=np.int64,
    )

    assert np.array_equal(
        descendants,
        expected,
    )


def test_model_b_observations_include_weighted_opportunity() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    susceptibility = np.array(
        [
            0.20,
            0.50,
            0.25,
        ],
        dtype=np.float64,
    )

    observations = model_b_observations(
        graph,
        [run],
        susceptibility=susceptibility,
    )

    # Node 0 is the artificial seed and
    # must therefore not be included.
    #
    # Only node 1 was naturally infected.
    assert np.array_equal(
        observations.run_index,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        observations.node_index,
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        observations.offspring,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        observations.opportunity,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )

    np.testing.assert_allclose(
        observations.weighted_opportunity,
        np.array(
            [0.50],
            dtype=np.float64,
        ),
    )

    assert np.array_equal(
        observations.descendants,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )


def test_model_b_observations_rejects_wrong_susceptibility_size() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    run = make_simple_run()

    with pytest.raises(ValueError):
        model_b_observations(
            graph,
            [run],
            susceptibility=np.ones(
                2,
                dtype=np.float64,
            ),
        )