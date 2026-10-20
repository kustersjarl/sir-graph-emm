"""
we model the per node epidemic paramters here for the heterogenous SIR model
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["NodeParams", "transmission_probability"]


@dataclass(frozen=True, eq=False)
class NodeParams:
    """
    per node susceptibility, infectiousness and recovery probability
    forced as read-only arrays
    """

    susceptibility: NDArray[np.float64]
    infectiousness: NDArray[np.float64]
    recovery: NDArray[np.float64]

    def __post_init__(self) -> None:
        """
        validate and freeze array
        """
        arrays = {}
        for name in ("susceptibility", "infectiousness", "recovery"):
            arr = np.array(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1d array, got {arr.ndim} instead")
            arrays[name] = arr

        sizes = {name: arr.size for name, arr in arrays.items()}
        if len(set(map(len, arrays.values()))) > 1:
            raise ValueError(f"mismatch in NodeParams, not equally sized got {sizes}")

        s, f, q = arrays["susceptibility"], arrays["infectiousness"], arrays["recovery"]

        if not np.all(np.isfinite(s) & (s >= 0)):
            raise ValueError(f"{s} must be finite and non-negative")

        if not np.all(np.isfinite(f) & (f >= 0)):
            raise ValueError(f"{f} must be finite and non-negative")

        if not np.all((q > 0) & (q <= 1)):
            raise ValueError(f"{q} must be in (0,1]")

        for name, arr in arrays.items():
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    @property
    def n_nodes(self) -> int:
        """
        amount of nodes, any param should suffice after __post_init__ tests
        """
        return self.recovery.size

    @classmethod
    def homogeneous(cls, n_nodes: int, *, recovery: float = 1.0) -> Self:
        """
        build all nodes: s=f=1 and q = recovery
        """
        return cls(np.ones(n_nodes), np.ones(n_nodes), np.full(n_nodes, recovery))

    def scaled(
        self,
        mask: NDArray[np.bool_],
        *,
        susceptibility: float = 1.0,
        infectiousness: float = 1.0,
        recovery: float | None = None,
    ) -> Self:
        """
        use a mask to determine a change in the node structure
        """
        mask = np.asarray(mask)
        if mask.shape != (self.n_nodes,):
            raise ValueError(
                f"mask must have shape ({self.n_nodes},), got {mask.shape}"
            )
        if mask.dtype != np.bool_:
            raise TypeError(f"mask must be boolean, got dtype {mask.dtype}")

        s = self.susceptibility.copy()
        f = self.infectiousness.copy()
        q = self.recovery.copy()

        s[mask] *= susceptibility
        f[mask] *= infectiousness
        if recovery is not None:
            q[mask] = recovery

        return type(self)(s, f, q)


def transmission_probability(
    p: float,
    params: NodeParams,
    src: NDArray[np.int64],
    dst: NDArray[np.int64],
    edge_multiplier: float | NDArray[np.float64] = 1.0,
) -> NDArray[np.float64]:
    """
    Per-edge transmission probabilities.

    p_ij = min(
        1,
        p
        * infectiousness_i
        * susceptibility_j
        * edge_multiplier_ij
    )
    """

    edge_multiplier = np.asarray(
        edge_multiplier,
        dtype=np.float64,
    )

    if np.any(
        ~np.isfinite(edge_multiplier)
    ):
        raise ValueError(
            "edge_multiplier must be finite"
        )

    if np.any(
        edge_multiplier < 0
    ):
        raise ValueError(
            "edge_multiplier cannot be negative"
        )

    return np.minimum(
        1.0,
        p
        * params.infectiousness[src]
        * params.susceptibility[dst]
        * edge_multiplier,
    )