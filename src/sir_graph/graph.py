from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Graph:
    """undirected, unweighted graph as a compressed sparse matrix format"""

    indptr: IntArray  # shape (n_nodes + 1,)
    indices: IntArray  # shape (2 * n_edges,)

    def __post_init__(self) -> None:
        """
        validate CSR invariants, then make the arrays read-only

        pass copy if in need of mutable graph
        """
        if self.indptr.ndim != 1 or self.indptr.size == 0:
            raise ValueError("indptr must be 1D with at least one entry")
        if self.indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if self.indptr[0] != 0:
            raise ValueError(f"indptr must start at 0, got {self.indptr[0]}")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        if self.indptr[-1] != self.indices.size:
            raise ValueError(
                f"indptr[-1] ({self.indptr[-1]}) must equal the length of indices "
                f"({self.indices.size})"
            )
        if self.indices.size > 0 and (
            self.indices.min() < 0 or self.indices.max() >= self.n_nodes
        ):
            raise ValueError(f"indices out of range [0, {self.n_nodes})")

        self.indptr.flags.writeable = False
        self.indices.flags.writeable = False

    def __eq__(self, other: object) -> bool:
        """
        fix equal checking not working properly
        """
        if not isinstance(other, Graph):
            return NotImplemented
        return np.array_equal(self.indptr, other.indptr) and np.array_equal(
            self.indices, other.indices
        )

    @classmethod
    def from_edges(cls, n_nodes: int, edges: IntArray) -> "Graph":
        """edges: shape (m, 2); symmetrise, drop self-loops/duplicates, build CSR"""
        if n_nodes < 0:
            raise ValueError(
                f"non-negative amount of nodes allowed, got {n_nodes} nodes"
            )

        edges = np.asarray(edges, dtype=np.int64)
        if (edges.size % 2) != 0:  # size not divisible by 2
            raise ValueError(
                f"needs to be a multiple of 2 pairs, got a shape of {edges.shape}"
            )
        edges = edges.reshape(-1, 2)

        if edges.size > 0 and (edges.min() < 0 or edges.max() >= n_nodes):
            raise ValueError(
                f"node ids out of range [0, {n_nodes}), "
                f"got min{edges.min()} and max {edges.max()}"
            )

        u, v = edges[:, 0], edges[:, 1]
        keep = u != v
        u, v = u[keep], v[keep]

        u, v = np.concatenate([u, v]), np.concatenate([v, u])

        key = u * n_nodes + v
        key = np.unique(key)
        u, v = np.divmod(key, n_nodes)

        counts = np.bincount(u, minlength=n_nodes)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        indices = v

        return cls(indptr=indptr, indices=indices)

    @property
    def n_nodes(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_edges(self) -> int:
        return len(self.indices) // 2

    def degree(self) -> IntArray:
        return np.diff(self.indptr)

    def neighbours(self, node: int) -> IntArray:
        return self.indices[self.indptr[node] : self.indptr[node + 1]]

    def edges(self) -> IntArray:
        """
        return each undirected edge once, as an (n_edges, 2) array with u < v
        """
        u = np.repeat(
            np.arange(self.n_nodes), self.degree()
        )  # what node belongs to wich index
        v = self.indices
        keep = u < v
        return np.column_stack([u[keep], v[keep]])

    def has_edge(self, u: int, v: int) -> bool:
        """true if {u, v} is an edge"""
        nbrs = self.neighbours(u)
        i = np.searchsorted(nbrs, v)
        return bool(i < nbrs.size and nbrs[i] == v)

    def degree_distribution(self) -> FloatArray:
        """Empirical p_k for k = 0 .. k_max, shape (k_max + 1,)."""
        if self.n_nodes == 0:
            raise ValueError("degree distribution is undefined for an empty graph")
        return np.bincount(self.degree()) / self.n_nodes

    def moments(self) -> tuple[float, float]:
        """First and second moments of the degree distribution: (<k>, <k^2>)."""
        if self.n_nodes == 0:
            raise ValueError("moments are undefined for an empty graph")
        k = self.degree().astype(np.float64)
        return float(k.mean()), float((k**2).mean())
