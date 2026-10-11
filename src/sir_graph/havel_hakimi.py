import numpy as np

from sir_graph.graph import Graph


def havel_hakimi(degrees) -> Graph:
    """
    Construct a simple graph with exactly the supplied degree sequence.
    """
    degrees = np.asarray(degrees, dtype=np.int64)

    if degrees.ndim != 1:
        raise ValueError("degrees must be a 1-D array")

    if np.any(degrees < 0):
        raise ValueError("degrees must be non-negative")

    n = len(degrees)

    if np.any(degrees >= n):
        raise ValueError("degree must be smaller than number of nodes")

    if degrees.sum() % 2 != 0:
        raise ValueError("sum of degrees must be even")

    # [remaining degree, original node id]
    remaining = [
        [int(degrees[i]), i]
        for i in range(n)
    ]

    edges = []

    while True:
        remaining.sort(
            key=lambda x: x[0],
            reverse=True
        )

        if not remaining or remaining[0][0] == 0:
            break

        d, node = remaining.pop(0)

        if d > len(remaining):
            raise ValueError("degree sequence is not graphical")

        for i in range(d):
            if remaining[i][0] <= 0:
                raise ValueError("degree sequence is not graphical")

            neighbour = remaining[i][1]

            edges.append((node, neighbour))

            remaining[i][0] -= 1

    return Graph.from_edges(
        n,
        np.asarray(edges, dtype=np.int64)
    )


def randomize_degree_preserving(
    graph: Graph,
    n_swaps: int,
    *,
    rng=None,
    max_attempts=None,
) -> Graph:
    """
    Randomize graph edges while preserving every node's degree.

    Uses double-edge swaps:

        A -- B        A -- D
        C -- D   ->   C -- B

    Self-loops and duplicate edges are rejected.
    """
    rng = np.random.default_rng(rng)

    edge_list = [
        tuple(map(int, edge))
        for edge in graph.edges()
    ]

    if n_swaps < 0:
        raise ValueError("n_swaps must be non-negative")

    if n_swaps == 0:
        return graph

    if len(edge_list) < 2:
        raise ValueError("graph needs at least two edges")

    # Normalize undirected edges
    edge_list = [
        (min(a, b), max(a, b))
        for a, b in edge_list
    ]

    edge_set = set(edge_list)

    if max_attempts is None:
        max_attempts = max(1000, n_swaps * 50)

    successful = 0
    attempts = 0

    while successful < n_swaps and attempts < max_attempts:
        attempts += 1

        i, j = rng.choice(
            len(edge_list),
            size=2,
            replace=False
        )

        a, b = edge_list[i]
        c, d = edge_list[j]

        # Require four different nodes
        if len({a, b, c, d}) < 4:
            continue

        # Randomly choose one of the two possible rewiring patterns
        if rng.random() < 0.5:
            new1 = (a, d)
            new2 = (c, b)
        else:
            new1 = (a, c)
            new2 = (b, d)

        new1 = (min(new1), max(new1))
        new2 = (min(new2), max(new2))

        if new1 == new2:
            continue

        # No duplicate edges
        if new1 in edge_set or new2 in edge_set:
            continue

        old1 = edge_list[i]
        old2 = edge_list[j]

        edge_set.remove(old1)
        edge_set.remove(old2)

        edge_set.add(new1)
        edge_set.add(new2)

        edge_list[i] = new1
        edge_list[j] = new2

        successful += 1

    if successful < n_swaps:
        raise RuntimeError(
            f"Only completed {successful}/{n_swaps} swaps "
            f"after {attempts} attempts"
        )

    return Graph.from_edges(
        graph.n_nodes,
        np.asarray(edge_list, dtype=np.int64)
    )