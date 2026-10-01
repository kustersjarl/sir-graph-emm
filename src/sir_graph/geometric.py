import math

import numpy as np
from numpy.typing import NDArray

from sir_graph.graph import Graph

__all__ = ["expected_mean_degree", "random_geometric"]


def random_geometric(
    n_nodes: int,
    radius: float,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[Graph, NDArray[np.float64]]:
    """
    random geometric graph on the unit square

    places `n_nodes` uniform points in [0, 1]^2 and
    connects pairs at distance < `radius`
    returns the graph and the `(n_nodes, 2)` positions
    """

    # ensure validity
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative: got {n_nodes} instead")
    if radius < 0:
        raise ValueError(f"radius must be non-negative: got {radius} instead")

    # set up generator
    rng = np.random.default_rng(rng)

    # sample postions
    positions = rng.random((n_nodes, 2))
    # skip loop ; vectorized operation to find difference in positions
    diff = positions[:, None, :] - positions[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)
    close = sq_dist < radius**2
    # use upper triangular properties to drop self-loops
    edges = np.argwhere(np.triu(close, k=1))
    graph = Graph.from_edges(n_nodes, edges)
    return (graph, positions)


def expected_mean_degree(n_nodes: int, radius: float) -> float:
    """
    expected mean degree of `random_geometric`, including boundary effects

    equals `(n_nodes - 1) * A(r)` with `A(r) = pi r^2 - 8/3 r^3 + r^4/2` s.t.
    the probability that two uniform points are within distance r (0 <= r <= 1)

    """
    # ensure validity
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative: got {n_nodes} instead")
    if not (0 <= radius <= 1):
        raise ValueError(f"radius must be in [0, 1], got {radius} instead")

    # compute mean
    if n_nodes == 0:
        return 0.0
    p_edge = math.pi * radius**2 - 8 / 3 * radius**3 + radius**4 / 2
    return (n_nodes - 1) * p_edge
