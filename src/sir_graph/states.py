from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

__all__ = ["STATE_DTYPE", "State", "choose_seeds"]

STATE_DTYPE = np.int8


class State(IntEnum):
    """
    integer value that showcases what state a node is in --
    doubles as column indices in simulation results and the bincount gives the amount of S,I,R counts
    """

    S = 0  # susceptible
    I = 1  # infected and infectious
    R = 2  # recovered and no longer infectious


def choose_seeds(
    n_nodes: int,
    n_seeds: int,
    *,
    rng: np.random.Generator | int | None = None,
    eligble: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    raise NotImplemented
