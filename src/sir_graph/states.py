from enum import IntEnum
from multiprocessing.sharedctypes import Value

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
    eligible: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """
    we pick `n_seeds`, distinct nodes uniformly at random

    the args * inputs the decorater to stay modular over files

    rng is either an object or the int to setup a seed

    seeds are only drawn from nodes where the boolean is true (optional), this will give
    us  flexibility in seeding and reseeding
    """
    rng = np.random.default_rng(rng)

    if eligible is None:
        candidates = np.arange(n_nodes)
    else:
        eligible = np.asarray(eligible)
        if eligible.shape != (n_nodes,):
            raise ValueError(f"expected shape: {(n_nodes,)} but got: {eligible.shape}")
        if eligible.dtype != np.bool_:  # so if not a boolean array
            raise TypeError(
                f"eligible must be a boolean mask, got dtype {eligible.dtype}"
            )
        candidates = np.flatnonzero(eligible)

    # ensure valid size
    if not 0 <= n_seeds <= candidates.size:
        raise ValueError(
            f"n_seeds must be between 0 and {candidates.size}, got {n_seeds}"
        )

    # sample and sort
    seeds = rng.choice(candidates, size=n_seeds, replace=False)
    return np.sort(seeds).astype(np.int64)
