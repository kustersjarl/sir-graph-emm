from itertools import combinations

import numpy as np
import pytest

from sir_graph.dynamics import SimulationResult, simulate, step
from sir_graph.graph import Graph
from sir_graph.parameters import NodeParams
from sir_graph.states import STATE_DTYPE, State


def make_graph(
    n: int,
    edges: list[tuple[int, int]],
) -> Graph:
    """Construct a simple undirected graph from an edge list."""
    return Graph.from_edges(
        n,
        np.asarray(
            edges,
            dtype=np.int64,
        ).reshape(-1, 2),
    )


def path_graph(n: int) -> Graph:
    """Path 0--1--...--(n-1)."""
    return make_graph(
        n,
        [(i, i + 1) for i in range(n - 1)],
    )


def complete_graph(n: int) -> Graph:
    """Complete graph on n nodes."""
    return make_graph(
        n,
        list(combinations(range(n), 2)),
    )


def test_contact_graph(n: int = 50) -> Graph:
    """
    Deterministic sparse graph used for reproducibility and invariant tests.

    Each node is connected to its next one and next two neighbours whenever
    those nodes exist.
    """
    edges = []

    for i in range(n):
        if i + 1 < n:
            edges.append((i, i + 1))
        if i + 2 < n:
            edges.append((i, i + 2))

    return make_graph(
        n,
        edges,
    )


def assert_consistent(
    graph: Graph,
    result: SimulationResult,
) -> None:
    """Check invariants every completed simulation must satisfy."""

    n = graph.n_nodes
    counts = result.counts

    assert counts.shape == (
        result.n_steps + 1,
        3,
    )

    assert np.all(
        counts.sum(axis=1) == n
    )

    # Susceptible count cannot increase.
    assert np.all(
        np.diff(
            counts[:, State.S]
        )
        <= 0
    )

    # Recovered count cannot decrease.
    assert np.all(
        np.diff(
            counts[:, State.R]
        )
        >= 0
    )

    # A completed run has no infectious nodes left.
    assert counts[-1, State.I] == 0

    # Initial infectious count equals the number of seeds.
    assert counts[0, State.I] == len(
        result.seeds
    )

    infection_time = result.infection_time
    recovery_time = result.recovery_time
    infector = result.infector

    # Seeds are infected at t=0 and have no recorded infector.
    assert np.all(
        infection_time[
            result.seeds
        ]
        == 0
    )

    assert np.all(
        infector[
            result.seeds
        ]
        == -1
    )

    # At the end of the epidemic, everyone ever infected has recovered.
    assert np.array_equal(
        infection_time >= 0,
        recovery_time >= 0,
    )

    # Every infected node remains infectious for at least one step.
    assert np.all(
        recovery_time[
            infection_time >= 0
        ]
        > infection_time[
            infection_time >= 0
        ]
    )

    # Recorded infectors must be adjacent and infectious when transmission
    # occurs.
    for target in np.flatnonzero(
        infector >= 0
    ):
        source = int(
            infector[target]
        )

        assert graph.has_edge(
            source,
            int(target),
        )

        assert (
            infection_time[source]
            < infection_time[target]
            <= recovery_time[source]
        )

    assert result.final_size == int(
        np.sum(
            infection_time >= 0
        )
    )

    # Every non-seed infection has exactly one recorded infector.
    assert (
        result.offspring.sum()
        == result.final_size
        - len(result.seeds)
    )


# ============================================================================
# step()
# ============================================================================


def test_step_does_not_modify_state() -> None:
    graph = path_graph(5)

    state = np.zeros(
        5,
        dtype=STATE_DTYPE,
    )
    state[2] = State.I

    before = state.copy()

    step(
        graph,
        state,
        NodeParams.homogeneous(5),
        1.0,
        np.random.default_rng(0),
    )

    assert np.array_equal(
        state,
        before,
    )


def test_step_p1_infects_all_susceptible_neighbours() -> None:
    graph = path_graph(5)

    state = np.array(
        [
            State.S,
            State.R,
            State.I,
            State.S,
            State.S,
        ],
        dtype=STATE_DTYPE,
    )

    events = step(
        graph,
        state,
        NodeParams.homogeneous(5),
        1.0,
        np.random.default_rng(0),
    )

    # Node 1 is already recovered, so only node 3 can be infected by node 2.
    assert events.newly_infected.tolist() == [
        3
    ]

    assert events.infectors.tolist() == [
        2
    ]

    # Homogeneous parameters use recovery probability 1 by default.
    assert events.newly_recovered.tolist() == [
        2
    ]


def test_step_one_infector_per_target() -> None:
    # Star graph: centre 0 susceptible, leaves 1..5 infectious.
    graph = make_graph(
        6,
        [
            (0, node)
            for node in range(1, 6)
        ],
    )

    state = np.full(
        6,
        State.I,
        dtype=STATE_DTYPE,
    )
    state[0] = State.S

    events = step(
        graph,
        state,
        NodeParams.homogeneous(6),
        1.0,
        np.random.default_rng(0),
    )

    assert events.newly_infected.tolist() == [
        0
    ]

    assert len(
        events.infectors
    ) == 1

    assert (
        1
        <= events.infectors[0]
        <= 5
    )


# ============================================================================
# simulate()
# ============================================================================


def test_path_p1_is_breadth_first_search() -> None:
    result = simulate(
        path_graph(5),
        [0],
        1.0,
        rng=0,
    )

    assert result.infection_time.tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert result.recovery_time.tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert result.infector.tolist() == [
        -1,
        0,
        1,
        2,
        3,
    ]

    assert result.n_steps == 5
    assert result.final_size == 5

    assert result.offspring.tolist() == [
        1,
        1,
        1,
        1,
        0,
    ]


def test_p0_only_seeds() -> None:
    result = simulate(
        complete_graph(10),
        [2, 7],
        0.0,
        rng=0,
    )

    assert result.final_size == 2
    assert result.n_steps == 1

    assert result.ever_infected.tolist() == [
        i in (2, 7)
        for i in range(10)
    ]


def test_isolated_nodes_never_infected() -> None:
    # Nodes 3, 4 and 5 are isolated.
    graph = make_graph(
        6,
        [
            (0, 1),
            (1, 2),
        ],
    )

    result = simulate(
        graph,
        [0],
        1.0,
        rng=0,
    )

    assert not result.ever_infected[
        3:
    ].any()


def test_zero_susceptibility_protects_group() -> None:
    n = 10

    protected = (
        np.arange(n) >= 5
    )

    params = (
        NodeParams
        .homogeneous(n)
        .scaled(
            protected,
            susceptibility=0.0,
        )
    )

    result = simulate(
        complete_graph(n),
        [0],
        1.0,
        params,
        rng=0,
    )

    assert result.ever_infected.tolist() == (
        [True] * 5
        + [False] * 5
    )


def test_zero_infectiousness_prevents_secondary_infections() -> None:
    graph = path_graph(3)

    mask = np.array(
        [
            True,
            False,
            False,
        ],
        dtype=np.bool_,
    )

    params = (
        NodeParams
        .homogeneous(3)
        .scaled(
            mask,
            infectiousness=0.0,
        )
    )

    result = simulate(
        graph,
        [0],
        1.0,
        params,
        rng=0,
    )

    assert result.final_size == 1
    assert result.offspring.sum() == 0


def test_clipping_large_factors() -> None:
    params = (
        NodeParams
        .homogeneous(5)
        .scaled(
            np.ones(
                5,
                dtype=np.bool_,
            ),
            susceptibility=10.0,
            infectiousness=10.0,
        )
    )

    result = simulate(
        path_graph(5),
        [0],
        0.5,
        params,
        rng=0,
    )

    # p * tau * sigma = 0.5 * 10 * 10 = 50, clipped to 1.
    assert result.final_size == 5


def test_reproducible() -> None:
    graph = test_contact_graph(50)

    params = NodeParams.homogeneous(
        50,
        recovery=0.5,
    )

    first = simulate(
        graph,
        [0, 1],
        0.3,
        params,
        rng=9,
    )

    second = simulate(
        graph,
        [0, 1],
        0.3,
        params,
        rng=9,
    )

    np.testing.assert_array_equal(
        first.counts,
        second.counts,
    )

    np.testing.assert_array_equal(
        first.infection_time,
        second.infection_time,
    )

    np.testing.assert_array_equal(
        first.recovery_time,
        second.recovery_time,
    )

    np.testing.assert_array_equal(
        first.infector,
        second.infector,
    )


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (0.7, 1.0),
        (0.2, 0.3),
        (0.05, 0.5),
    ],
)
def test_simulation_invariants(
    p: float,
    q: float,
) -> None:
    graph = test_contact_graph(50)

    params = NodeParams.homogeneous(
        50,
        recovery=q,
    )

    rng = np.random.default_rng(2)

    for _ in range(5):
        seeds = rng.choice(
            50,
            size=2,
            replace=False,
        )

        result = simulate(
            graph,
            seeds,
            p,
            params,
            rng=rng,
        )

        assert_consistent(
            graph,
            result,
        )


@pytest.mark.parametrize(
    "seeds",
    [
        [0, 0],
        [10],
        [-1],
    ],
)
def test_invalid_seeds(
    seeds: list[int],
) -> None:
    with pytest.raises(
        ValueError
    ):
        simulate(
            path_graph(5),
            seeds,
            0.5,
            rng=0,
        )


@pytest.mark.parametrize(
    "p",
    [
        -0.1,
        1.1,
    ],
)
def test_invalid_p(
    p: float,
) -> None:
    with pytest.raises(
        ValueError
    ):
        simulate(
            path_graph(5),
            [0],
            p,
            rng=0,
        )


def test_params_size_mismatch() -> None:
    with pytest.raises(
        ValueError
    ):
        simulate(
            path_graph(5),
            [0],
            0.5,
            NodeParams.homogeneous(4),
            rng=0,
        )


def test_max_steps_raises() -> None:
    params = NodeParams.homogeneous(
        50,
        recovery=1e-6,
    )

    with pytest.raises(
        RuntimeError
    ):
        simulate(
            path_graph(50),
            [0],
            1.0,
            params,
            rng=0,
            max_steps=10,
        )


def test_invalid_edge_multiplier_shape() -> None:
    graph = path_graph(3)

    # The CSR representation contains four directed entries for two
    # undirected edges, so a length-two multiplier array is invalid.
    with pytest.raises(
        ValueError
    ):
        simulate(
            graph,
            [0],
            0.5,
            edge_multipliers=np.ones(
                2,
                dtype=np.float64,
            ),
            rng=0,
        )


def test_negative_edge_multiplier_rejected() -> None:
    graph = make_graph(
        2,
        [
            (0, 1),
        ],
    )

    with pytest.raises(
        ValueError
    ):
        simulate(
            graph,
            [0],
            0.5,
            edge_multipliers=np.array(
                [
                    1.0,
                    -1.0,
                ],
                dtype=np.float64,
            ),
            rng=0,
        )


# ============================================================================
# Transmission behaviour
# ============================================================================


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (0.3, 0.4),
        (0.7, 1.0),
        (0.1, 0.2),
    ],
)
def test_transmissibility_single_edge(
    p: float,
    q: float,
) -> None:
    """
    For a single edge with homogeneous parameters, eventual transmissibility is

        T = p / (p + q - p*q).
    """

    graph = make_graph(
        2,
        [
            (0, 1),
        ],
    )

    params = NodeParams.homogeneous(
        2,
        recovery=q,
    )

    rng = np.random.default_rng(11)

    n_runs = 10_000

    hits = sum(
        simulate(
            graph,
            [0],
            p,
            params,
            rng=rng,
        ).ever_infected[1]
        for _ in range(
            n_runs
        )
    )

    expected = (
        p
        / (
            p
            + q
            - p * q
        )
    )

    assert (
        hits / n_runs
        == pytest.approx(
            expected,
            abs=0.02,
        )
    )


def test_edge_multiplier_changes_transmission() -> None:
    graph = make_graph(
        2,
        [
            (0, 1),
        ],
    )

    params = NodeParams.homogeneous(
        2,
        recovery=1.0,
    )

    # CSR contains both directed entries:
    # 0 -> 1 and 1 -> 0.
    edge_multipliers = np.array(
        [
            2.0,
            2.0,
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(123)

    n_runs = 20_000

    hits = sum(
        simulate(
            graph,
            [0],
            0.2,
            params,
            edge_multipliers=
                edge_multipliers,
            rng=rng,
        ).ever_infected[1]
        for _ in range(
            n_runs
        )
    )

    observed = (
        hits / n_runs
    )

    # Recovery is certain after one step, so the infection probability is
    # simply 0.2 * 2 = 0.4.
    assert observed == pytest.approx(
        0.4,
        abs=0.02,
    )
