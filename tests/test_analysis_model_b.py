import numpy as np
import pytest

from sir_graph.analysis import (
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
        seed, infected at t=0
        recovers at t=2

    Node 1:
        naturally infected at t=1
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
    # Node 1 is susceptible only during t=0.
    #
    # Node 1:
    # Node 2 remains susceptible during t=1 and t=2.
    #
    # Node 2:
    # Never infectious.
    expected = np.array(
        [1, 2, 0],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
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

    weighted = weighted_opportunity_contact_steps(
        graph,
        run,
        susceptibility,
    )

    # Node 0:
    # 1 contact-step with node 1:
    # 1 * 0.50 = 0.50
    #
    # Node 1:
    # 2 contact-steps with node 2:
    # 2 * 0.25 = 0.50
    #
    # Node 2:
    # Never infectious.
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

    weighted = weighted_opportunity_contact_steps(
        graph,
        run,
        np.ones(
            3,
            dtype=np.float64,
        ),
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

    # Wrong number of susceptibility values.
    with pytest.raises(ValueError):
        weighted_opportunity_contact_steps(
            graph,
            run,
            np.array(
                [0.5, 0.5],
                dtype=np.float64,
            ),
        )

    # Negative susceptibility.
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


def test_weighted_opportunity_respects_edge_multipliers() -> None:
    """
    Verify that Model B opportunity includes contact-specific edge weights.
    """

    graph = make_graph(
        3,
        [
            (0, 1),
            (0, 2),
        ],
    )

    # Node 2 is the seed and infects node 0.
    # Node 0 is then infectious for two steps while
    # node 1 remains susceptible.
    run = SimulationResult(
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
            [1, -1, 0],
            dtype=np.int64,
        ),
        recovery_time=np.array(
            [3, -1, 2],
            dtype=np.int64,
        ),
        infector=np.array(
            [2, -1, -1],
            dtype=np.int64,
        ),
        seeds=np.array(
            [2],
            dtype=np.int64,
        ),
    )

    susceptibility = np.ones(
        3,
        dtype=np.float64,
    )

    edge_multipliers = np.ones(
        graph.indices.size,
        dtype=np.float64,
    )

    # Give edge 0 <-> 1 multiplier 2 in both CSR directions.
    for source, target in [
        (0, 1),
        (1, 0),
    ]:
        start = graph.indptr[source]
        stop = graph.indptr[source + 1]

        local_index = np.flatnonzero(
            graph.indices[start:stop] == target
        )

        assert local_index.size == 1

        edge_multipliers[
            start + local_index[0]
        ] = 2.0

    ordinary = weighted_opportunity_contact_steps(
        graph,
        run,
        susceptibility,
    )

    household_weighted = weighted_opportunity_contact_steps(
        graph,
        run,
        susceptibility,
        edge_multipliers=edge_multipliers,
    )

    # Node 0 is infectious from t=1 until t=3:
    # 2 susceptible contact-steps with node 1.
    assert ordinary[0] == pytest.approx(2.0)

    # 2 contact-steps * susceptibility 1 * edge weight 2 = 4.
    assert household_weighted[0] == pytest.approx(4.0)

    # Verify model_b_observations passes the edge multipliers through.
    observations = model_b_observations(
        graph,
        [run],
        susceptibility=susceptibility,
        edge_multipliers=edge_multipliers,
    )

    # Node 0 is the only naturally infected node.
    # Node 2 is the artificial seed.
    assert observations.node_index.tolist() == [0]

    assert (
        observations.weighted_opportunity[0]
        == pytest.approx(4.0)
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

    # Node 0 is the artificial seed and is excluded.
    # Only node 1 was naturally infected.
    np.testing.assert_array_equal(
        observations.run_index,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        observations.node_index,
        np.array(
            [1],
            dtype=np.int64,
        ),
    )

    # Node 1 caused no direct secondary infections.
    np.testing.assert_array_equal(
        observations.offspring,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )

    # Node 1 has two susceptible-neighbour contact-steps with node 2.
    np.testing.assert_array_equal(
        observations.opportunity,
        np.array(
            [2],
            dtype=np.int64,
        ),
    )

    # 2 contact-steps * susceptibility(node 2) = 2 * 0.25 = 0.50.
    np.testing.assert_allclose(
        observations.weighted_opportunity,
        np.array(
            [0.50],
            dtype=np.float64,
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


def test_model_b_observations_rejects_empty_runs() -> None:
    graph = make_graph(
        3,
        [
            (0, 1),
            (1, 2),
        ],
    )

    with pytest.raises(ValueError):
        model_b_observations(
            graph,
            [],
            susceptibility=np.ones(
                3,
                dtype=np.float64,
            ),
        )