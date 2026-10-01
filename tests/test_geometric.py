import math

import numpy as np
import pytest

from sir_graph.geometric import expected_mean_degree, random_geometric


def test_shapes_and_bounds() -> None:
    graph, pos = random_geometric(100, 0.2, rng=0)
    assert graph.n_nodes == 100
    assert pos.shape == (100, 2)
    assert np.all((pos >= 0.0) & (pos <= 1.0))


def test_edges_match_distances_exactly() -> None:
    """
    brute force test: edge (i, j) exists iff dist(i, j) < radius
    """
    n, r = 60, 0.25
    graph, pos = random_geometric(n, r, rng=1)
    for i in range(n):
        for j in range(i + 1, n):
            close = np.hypot(*(pos[i] - pos[j])) < r
            assert graph.has_edge(i, j) == close, (i, j)


def test_reproducible() -> None:
    g1, p1 = random_geometric(80, 0.2, rng=123)
    g2, p2 = random_geometric(80, 0.2, rng=123)
    assert np.array_equal(p1, p2)
    assert g1 == g2


@pytest.mark.parametrize(("n", "r"), [(0, 0.1), (1, 0.1), (50, 0.0)])
def test_edge_cases_have_no_edges(n: int, r: float) -> None:
    graph, pos = random_geometric(n, r, rng=0)
    assert graph.n_nodes == n
    assert graph.n_edges == 0
    assert pos.shape == (n, 2)


def test_radius_covering_square_gives_complete_graph() -> None:
    n = 30
    graph, _ = random_geometric(n, math.sqrt(2) + 0.01, rng=0)
    assert graph.n_edges == n * (n - 1) // 2


@pytest.mark.parametrize(("n", "r"), [(-1, 0.1), (10, -0.1)])
def test_invalid_arguments(n: int, r: float) -> None:
    with pytest.raises(ValueError):
        random_geometric(n, r)


def test_expected_mean_degree_formula() -> None:
    # (n - 1) * (pi r^2 - 8/3 r^3 + r^4 / 2) for the assignment's parameters
    assert expected_mean_degree(250, 0.15) == pytest.approx(15.4228, rel=1e-4)
    assert expected_mean_degree(250, 0.0) == 0.0
    # r = 1: A(1) = pi - 8/3 + 1/2
    assert expected_mean_degree(2, 1.0) == pytest.approx(math.pi - 8 / 3 + 0.5)


def test_expected_mean_degree_rejects_large_radius() -> None:
    with pytest.raises(ValueError):
        expected_mean_degree(10, 1.5)


def test_empirical_mean_degree_matches_theory() -> None:
    """
    averaged over graphs, 2m/n should be close to (n - 1) A(r)
    """
    n, r = 250, 0.15
    rng = np.random.default_rng(7)
    mean_degrees = [
        2 * random_geometric(n, r, rng=rng)[0].n_edges / n for _ in range(20)
    ]
    assert np.mean(mean_degrees) == pytest.approx(expected_mean_degree(n, r), rel=0.03)
