import numpy as np
import pytest

from sir_graph.graph import Graph

# helpers


def assert_valid_simple_graph(g: Graph) -> None:
    """check invariants that __post_init__ skips"""
    assert g.degree().sum() == 2 * g.n_edges == g.indices.size
    for u in range(g.n_nodes):
        nbrs = g.neighbours(u)
        # Sorted and no duplicates.
        assert np.all(np.diff(nbrs) > 0), f"node {u}: unsorted or duplicate"
        # No self-loops.
        assert u not in nbrs, f"node {u}: self-loop"
        # Symmetry.
        for v in nbrs:
            assert g.has_edge(v, u), f"edge ({u}, {v}) not symmetric"


# fixtures


@pytest.fixture
def small_graph() -> Graph:
    """edges {0,1}, {1,2}; node 3 isolated"""
    return Graph(indptr=np.array([0, 1, 3, 4, 4]), indices=np.array([1, 0, 2, 1]))


@pytest.fixture
def random_graph() -> Graph:
    rng = np.random.default_rng(42)
    edges = rng.integers(0, 50, size=(300, 2))  # includes loops and duplicates
    return Graph.from_edges(50, edges)


#  from_edges


def test_from_edges_small_example() -> None:
    g = Graph.from_edges(4, np.array([(0, 1), (1, 0), (2, 2), (1, 2)]))
    np.testing.assert_array_equal(g.indptr, [0, 1, 3, 4, 4])
    np.testing.assert_array_equal(g.indices, [1, 0, 2, 1])


def test_from_edges_accepts_list() -> None:
    g = Graph.from_edges(3, [(0, 1), (1, 2)])
    assert g.n_edges == 2


def test_from_edges_empty_case() -> None:
    g = Graph.from_edges(3, np.array([]))
    np.testing.assert_array_equal(g.indptr, [0, 0, 0, 0])
    assert len(g.indices) == 0
    assert g.n_edges == 0


def test_from_edges_zero_nodes() -> None:
    g = Graph.from_edges(0, np.array([]))
    assert g.n_nodes == 0
    assert g.n_edges == 0


@pytest.mark.parametrize(
    ("n_nodes", "edges", "match"),
    [
        (3, [(0, 3)], "out of range"),
        (2, [(0, -1)], "out of range"),
        (-1, [], "non-negative"),
        (2, [0, 1, 2], "pairs"),
    ],
)
def test_from_edges_rejects_bad_input(n_nodes: int, edges: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Graph.from_edges(n_nodes, np.array(edges))


def test_from_edges_random_is_valid(random_graph: Graph) -> None:
    assert_valid_simple_graph(random_graph)


#  validation


@pytest.mark.parametrize(
    ("indptr", "indices", "match"),
    [
        ([], [], "at least one entry"),
        ([2, 3, 5], [0, 1, 0, 1, 0], "must start at 0"),
        ([0, 3, 1, 4], [1, 2, 0, 0], "non-decreasing"),
        ([0, 1, 2], [1, 0, 1, 0], "must equal the length of indices"),
        ([0, 1, 3, 4, 4], [1, 0, 7, 1], "out of range"),
    ],
)
def test_invalid_csr_rejected(
    indptr: list[int], indices: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Graph(
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.array(indices, dtype=np.int64),
        )


#  immutability


def test_arrays_are_read_only(small_graph: Graph) -> None:
    with pytest.raises(ValueError, match="read-only"):
        small_graph.indices[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        small_graph.indptr[0] = 99


def test_neighbours_view_is_read_only(small_graph: Graph) -> None:
    with pytest.raises(ValueError, match="read-only"):
        small_graph.neighbours(1)[0] = 99


# equality


def test_equal_graphs(small_graph: Graph) -> None:
    assert small_graph == Graph.from_edges(4, [(0, 1), (1, 2)])


def test_unequal_graphs(small_graph: Graph) -> None:
    triangle = Graph.from_edges(3, [(0, 1), (1, 2), (0, 2)])
    assert small_graph != triangle


def test_not_equal_to_other_types(small_graph: Graph) -> None:
    assert small_graph != "not a graph"


#  structure


def test_n_nodes(small_graph: Graph) -> None:
    assert small_graph.n_nodes == 4


def test_n_edges(small_graph: Graph) -> None:
    assert small_graph.n_edges == 2


def test_degree(small_graph: Graph) -> None:
    np.testing.assert_array_equal(small_graph.degree(), [1, 2, 1, 0])


def test_neighbours_values(small_graph: Graph) -> None:
    np.testing.assert_array_equal(small_graph.neighbours(1), [0, 2])
    assert len(small_graph.neighbours(3)) == 0


def test_neighbours_is_view(small_graph: Graph) -> None:
    assert np.shares_memory(small_graph.neighbours(1), small_graph.indices)


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [(0, 1, True), (1, 0, True), (1, 2, True), (0, 2, False), (3, 0, False)],
)
def test_has_edge(small_graph: Graph, u: int, v: int, expected: bool) -> None:
    assert small_graph.has_edge(u, v) is expected


def test_edges(small_graph: Graph) -> None:
    np.testing.assert_array_equal(small_graph.edges(), [[0, 1], [1, 2]])


def test_edges_empty() -> None:
    assert Graph.from_edges(3, []).edges().shape == (0, 2)


def test_edges_round_trip(small_graph: Graph, random_graph: Graph) -> None:
    for g in (small_graph, random_graph):
        assert Graph.from_edges(g.n_nodes, g.edges()) == g


# statistics


def test_degree_distribution(small_graph: Graph) -> None:
    # degrees [1, 2, 1, 0] expect one node with k=0, two with k=1, one with k=2
    np.testing.assert_allclose(small_graph.degree_distribution(), [0.25, 0.5, 0.25])


def test_degree_distribution_sums_to_one(random_graph: Graph) -> None:
    assert random_graph.degree_distribution().sum() == pytest.approx(1.0)


def test_moments(small_graph: Graph) -> None:
    # <k> = (1+2+1+0)/4 = 1.0;  <k^2> = (1+4+1+0)/4 = 1.5
    mean_k, mean_k2 = small_graph.moments()
    assert mean_k == pytest.approx(1.0)
    assert mean_k2 == pytest.approx(1.5)


def test_mean_degree_is_2m_over_n(random_graph: Graph) -> None:
    mean_k, _ = random_graph.moments()
    assert mean_k == pytest.approx(2 * random_graph.n_edges / random_graph.n_nodes)


def test_statistics_undefined_for_empty_graph() -> None:
    g = Graph.from_edges(0, [])
    with pytest.raises(ValueError, match="empty graph"):
        g.degree_distribution()
    with pytest.raises(ValueError, match="empty graph"):
        g.moments()
