from itertools import combinations

import numpy as np
import pytest

from sir_graph.dynamics import SimulationResult, run_ensemble, simulate, step
from sir_graph.geometric import random_geometric
from sir_graph.graph import Graph
from sir_graph.parameters import NodeParams
from sir_graph.states import STATE_DTYPE, State


def make_graph(n: int, edges: list[tuple[int, int]]) -> Graph:
    """Single place to adapt if your Graph.from_edges signature differs."""
    return Graph.from_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2))


def path_graph(n: int) -> Graph:
    return make_graph(n, [(i, i + 1) for i in range(n - 1)])


def complete_graph(n: int) -> Graph:
    return make_graph(n, list(combinations(range(n), 2)))


def assert_consistent(graph: Graph, res: SimulationResult) -> None:
    """Invariants every run must satisfy, whatever the randomness."""
    n = graph.n_nodes
    counts = res.counts
    assert counts.shape == (res.n_steps + 1, 3)
    assert np.all(counts.sum(axis=1) == n)
    assert np.all(np.diff(counts[:, State.S]) <= 0)  # S never grows
    assert np.all(np.diff(counts[:, State.R]) >= 0)  # R never shrinks
    assert counts[-1, State.I] == 0  # run ends with no infected
    assert counts[0, State.I] == len(res.seeds)

    inf_t, rec_t, who = res.infection_time, res.recovery_time, res.infector
    assert np.all(inf_t[res.seeds] == 0)
    assert np.all(who[res.seeds] == -1)
    # infected <=> recovered (no one is still I at the end), and I lasts >= 1 step
    assert np.array_equal(inf_t >= 0, rec_t >= 0)
    assert np.all(rec_t[inf_t >= 0] > inf_t[inf_t >= 0])
    # transmission tree: infector is a neighbour that was infectious at the time
    for j in np.flatnonzero(who >= 0):
        i = who[j]
        assert graph.has_edge(i, j)
        assert inf_t[i] < inf_t[j] <= rec_t[i]
    assert res.final_size == int(np.sum(inf_t >= 0))
    assert res.offspring.sum() == res.final_size - len(res.seeds)


# step


def test_step_does_not_modify_state() -> None:
    graph = path_graph(5)
    state = np.zeros(5, dtype=STATE_DTYPE)
    state[2] = State.I
    before = state.copy()
    step(graph, state, NodeParams.homogeneous(5), 1.0, np.random.default_rng(0))
    assert np.array_equal(state, before)


def test_step_p1_infects_all_susceptible_neighbours() -> None:
    graph = path_graph(5)
    state = np.array([State.S, State.R, State.I, State.S, State.S], dtype=STATE_DTYPE)
    ev = step(graph, state, NodeParams.homogeneous(5), 1.0, np.random.default_rng(0))
    assert ev.newly_infected.tolist() == [3]  # node 1 is R, so only node 3
    assert ev.infectors.tolist() == [2]
    assert ev.newly_recovered.tolist() == [2]  # q = 1


def test_step_one_infector_per_target() -> None:
    # star: centre 0 is susceptible, leaves 1..5 infected, p = 1
    graph = make_graph(6, [(0, k) for k in range(1, 6)])
    state = np.full(6, State.I, dtype=STATE_DTYPE)
    state[0] = State.S
    ev = step(graph, state, NodeParams.homogeneous(6), 1.0, np.random.default_rng(0))
    assert ev.newly_infected.tolist() == [0]
    assert len(ev.infectors) == 1 and 1 <= ev.infectors[0] <= 5


#  simulate


def test_path_p1_is_breadth_first_search() -> None:
    res = simulate(path_graph(5), [0], 1.0, rng=0)
    assert res.infection_time.tolist() == [0, 1, 2, 3, 4]
    assert res.recovery_time.tolist() == [1, 2, 3, 4, 5]  # q = 1
    assert res.infector.tolist() == [-1, 0, 1, 2, 3]
    assert res.n_steps == 5
    assert res.final_size == 5
    assert res.offspring.tolist() == [1, 1, 1, 1, 0]


def test_p0_only_seeds() -> None:
    res = simulate(complete_graph(10), [2, 7], 0.0, rng=0)
    assert res.final_size == 2
    assert res.n_steps == 1
    assert res.ever_infected.tolist() == [i in (2, 7) for i in range(10)]


def test_isolated_nodes_never_infected() -> None:
    graph = make_graph(6, [(0, 1), (1, 2)])  # nodes 3, 4, 5 isolated
    res = simulate(graph, [0], 1.0, rng=0)
    assert not res.ever_infected[3:].any()


def test_zero_susceptibility_protects_group() -> None:
    n = 10
    mask = np.arange(n) >= 5
    params = NodeParams.homogeneous(n).scaled(mask, susceptibility=0.0)
    res = simulate(complete_graph(n), [0], 1.0, params, rng=0)
    assert res.ever_infected.tolist() == [True] * 5 + [False] * 5


def test_clipping_large_factors() -> None:
    params = NodeParams.homogeneous(5).scaled(
        np.ones(5, dtype=bool), susceptibility=10.0, infectiousness=10.0
    )
    res = simulate(path_graph(5), [0], 0.5, params, rng=0)
    assert res.final_size == 5  # p_ij = min(1, 50) = 1


def test_reproducible() -> None:
    graph, _ = random_geometric(200, 0.15, rng=0)
    a = simulate(graph, [0, 1], 0.3, NodeParams.homogeneous(200, recovery=0.5), rng=9)
    b = simulate(graph, [0, 1], 0.3, NodeParams.homogeneous(200, recovery=0.5), rng=9)
    assert np.array_equal(a.counts, b.counts)
    assert np.array_equal(a.infector, b.infector)


@pytest.mark.parametrize(("p", "q"), [(0.7, 1.0), (0.2, 0.3), (0.05, 0.5)])
def test_invariants_on_geometric_graph(p: float, q: float) -> None:
    graph, _ = random_geometric(250, 0.15, rng=1)
    params = NodeParams.homogeneous(250, recovery=q)
    rng = np.random.default_rng(2)
    for _ in range(5):
        seeds = rng.choice(250, size=2, replace=False)
        assert_consistent(graph, simulate(graph, seeds, p, params, rng=rng))


@pytest.mark.parametrize("seeds", [[0, 0], [10], [-1]])
def test_invalid_seeds(seeds: list[int]) -> None:
    with pytest.raises(ValueError):
        simulate(path_graph(5), seeds, 0.5, rng=0)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_invalid_p(p: float) -> None:
    with pytest.raises(ValueError):
        simulate(path_graph(5), [0], p, rng=0)


def test_params_size_mismatch() -> None:
    with pytest.raises(ValueError):
        simulate(path_graph(5), [0], 0.5, NodeParams.homogeneous(4), rng=0)


def test_max_steps_raises() -> None:
    params = NodeParams.homogeneous(50, recovery=1e-6)  # practically never recovers
    with pytest.raises(RuntimeError):
        simulate(path_graph(50), [0], 1.0, params, rng=0, max_steps=10)


# theory


@pytest.mark.parametrize(("p", "q"), [(0.3, 0.4), (0.7, 1.0), (0.1, 0.2)])
def test_transmissibility_single_edge(p: float, q: float) -> None:
    """P(infection passes along one edge) = T = p / (p + q - p q)."""
    graph = make_graph(2, [(0, 1)])
    params = NodeParams.homogeneous(2, recovery=q)
    rng = np.random.default_rng(11)
    n_runs = 10_000
    hits = sum(
        simulate(graph, [0], p, params, rng=rng).ever_infected[1] for _ in range(n_runs)
    )
    expected = p / (p + q - p * q)
    assert hits / n_runs == pytest.approx(expected, abs=0.02)


# ensemble


def test_run_ensemble() -> None:
    graph, _ = random_geometric(100, 0.2, rng=0)
    results = run_ensemble(graph, 20, 2, 0.3, rng=5)
    assert len(results) == 20
    assert all(len(r.seeds) == 2 for r in results)
    assert len({tuple(r.seeds.tolist()) for r in results}) > 1  # fresh seeds per run
    again = run_ensemble(graph, 20, 2, 0.3, rng=5)
    assert [r.final_size for r in results] == [r.final_size for r in again]


def test_edge_multiplier_changes_transmission() -> None:
    graph = make_graph(
        2,
        [(0, 1)],
    )

    params = NodeParams.homogeneous(
        2,
        recovery=1.0,
    )

    # CSR contains both:
    # 0 -> 1
    # 1 -> 0
    edge_multipliers = np.array(
        [2.0, 2.0],
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        123
    )

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
        hits
        / n_runs
    )

    # Recovery is certain after one step,
    # so transmission probability is
    # simply 0.2 * 2 = 0.4.
    assert observed == pytest.approx(
        0.4,
        abs=0.02,
    )