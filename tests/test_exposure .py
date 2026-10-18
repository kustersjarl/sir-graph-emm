import numpy as np
import pytest

from sir_graph.exposure import contact_steps, exposure_opportunity, run_metrics
from sir_graph.dynamics import SimulationResult, apply_events, simulate, step
from sir_graph.graph import Graph
from sir_graph.havel_hakimi import havel_hakimi, randomize_degree_preserving
from sir_graph.parameters import NodeParams
from sir_graph.states import STATE_DTYPE, State, choose_seeds


def _path_result():
    # path 0 - 1 - 2, seed 0:  0 infectious t=0,1  1 infectious t=1,2  2 infectious t=3
    graph = Graph.from_edges(3, np.array([[0, 1], [1, 2]]))
    result = SimulationResult(
        counts=np.zeros((5, 3), dtype=np.int64),
        infection_time=np.array([0, 1, 3]),
        recovery_time=np.array([2, 3, 4]),
        infector=np.array([-1, 0, 1]),
        seeds=np.array([0]),
    )
    return graph, result


def test_hand_computed_path():
    graph, result = _path_result()
    exposure, opportunity = exposure_opportunity(graph, result)
    # node 1 is exposed at t=0 only (node 0 infectious); node 2 at t=1,2 (node 1 infectious)
    assert exposure.tolist() == [0, 1, 2]
    # node 0 had node 1 susceptible at t=0; node 1 had node 2 susceptible at t=1,2
    assert opportunity.tolist() == [1, 2, 0]


def test_contact_steps_per_edge():
    graph, result = _path_result()
    src, dst, c = contact_steps(graph, result)
    steps = {(int(s), int(d)): int(x) for s, d, x in zip(src, dst, c)}
    assert steps == {(0, 1): 1, (1, 0): 0, (1, 2): 2, (2, 1): 0}


def _replay(graph, seeds, p, params, rng):
    """instrumented simulation: count exposure and opportunity step by step"""
    n = graph.n_nodes
    state = np.full(n, State.S, dtype=STATE_DTYPE)
    state[seeds] = State.I
    exposure = np.zeros(n, dtype=np.int64)
    opportunity = np.zeros(n, dtype=np.int64)
    while np.any(state == State.I):
        for v in range(n):
            neighbours = graph.neighbours(v)
            if state[v] == State.S:
                exposure[v] += np.sum(state[neighbours] == State.I)
            elif state[v] == State.I:
                opportunity[v] += np.sum(state[neighbours] == State.S)
        apply_events(state, step(graph, state, params, p, rng))
    return exposure, opportunity


@pytest.fixture(scope="module")
def graph():
    return randomize_degree_preserving(havel_hakimi([4] * 60), 200, rng=0)


@pytest.mark.parametrize("run_seed", range(8))
def test_matches_instrumented_replay(graph, run_seed):
    n = graph.n_nodes
    rng = np.random.default_rng(run_seed)
    params = NodeParams(rng.uniform(0.3, 1.0, n), rng.uniform(0.5, 1.5, n), np.full(n, 0.4))
    seeds = choose_seeds(n, 3, rng=run_seed)

    result = simulate(graph, seeds, 0.2, params, rng=100 + run_seed)
    expected = _replay(graph, seeds, 0.2, params, np.random.default_rng(100 + run_seed))
    exposure, opportunity = exposure_opportunity(graph, result)

    assert np.array_equal(exposure, expected[0])
    assert np.array_equal(opportunity, expected[1])
    assert exposure.sum() == opportunity.sum()
    assert np.all(exposure[result.seeds] == 0)


def test_run_metrics_shapes_and_masks(graph):
    n = graph.n_nodes
    params = NodeParams.homogeneous(n, recovery=0.5)
    runs = [simulate(graph, choose_seeds(n, 2, rng=i), 0.2, params, rng=i) for i in range(5)]
    metrics = run_metrics(graph, runs)

    assert metrics.natural.shape == metrics.exposure.shape == (5, n)
    assert metrics.n_runs == 5 and metrics.n_nodes == n
    assert np.all(metrics.is_seed.sum(axis=1) == 2)
    assert not np.any(metrics.natural & metrics.is_seed)
    # every natural infection was exposed at least once
    assert np.all(metrics.exposure[metrics.natural] >= 1)


def test_run_metrics_rejects_empty_and_mismatched(graph):
    with pytest.raises(ValueError):
        run_metrics(graph, [])
    _, other = _path_result()
    with pytest.raises(ValueError):
        run_metrics(graph, [other])
