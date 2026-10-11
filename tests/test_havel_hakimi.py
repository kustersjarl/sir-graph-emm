import numpy as np
import pytest

from sir_graph.havel_hakimi import (
    havel_hakimi,
    randomize_degree_preserving,
)


def test_havel_hakimi_preserves_degree_sequence():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    np.testing.assert_array_equal(
        graph.degree(),
        degrees,
    )


def test_havel_hakimi_number_of_edges():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    assert graph.n_edges == degrees.sum() // 2


def test_havel_hakimi_rejects_odd_degree_sum():
    degrees = np.array([2, 2, 1])

    with pytest.raises(ValueError):
        havel_hakimi(degrees)


def test_havel_hakimi_rejects_negative_degree():
    degrees = np.array([2, -1, 1])

    with pytest.raises(ValueError):
        havel_hakimi(degrees)


def test_randomization_preserves_degrees():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    randomized = randomize_degree_preserving(
        graph,
        n_swaps=20,
        rng=42,
    )

    np.testing.assert_array_equal(
        randomized.degree(),
        graph.degree(),
    )


def test_randomization_preserves_number_of_edges():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    randomized = randomize_degree_preserving(
        graph,
        n_swaps=20,
        rng=42,
    )

    assert randomized.n_edges == graph.n_edges


def test_randomization_is_reproducible():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    a = randomize_degree_preserving(
        graph,
        n_swaps=20,
        rng=123,
    )

    b = randomize_degree_preserving(
        graph,
        n_swaps=20,
        rng=123,
    )

    assert a == b


def test_zero_swaps_returns_same_graph():
    degrees = np.array([3, 3, 2, 2, 2, 2])

    graph = havel_hakimi(degrees)

    randomized = randomize_degree_preserving(
        graph,
        n_swaps=0,
        rng=42,
    )

    assert randomized == graph