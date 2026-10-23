from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sir_graph.graph import Graph
from sir_graph.parameters import NodeParams, transmission_probability
from sir_graph.states import STATE_DTYPE, State


__all__ = [
    "StepEvents",
    "SimulationResult",
    "step",
    "apply_events",
    "simulate",
]


_EMPTY_INT = np.empty(0, dtype=np.int64)
_EMPTY_FLOAT = np.empty(0, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class StepEvents:
    """
    Events generated during one simulation step.

    ``infectors[k]`` is the source responsible for
    ``newly_infected[k]``.
    """

    newly_infected: NDArray[np.int64]
    infectors: NDArray[np.int64]
    newly_recovered: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """
    Complete record of one SIR simulation.

    ``counts[t]`` stores the numbers of susceptible, infectious, and
    recovered individuals at simulation step t.

    Per-node arrays use -1 for "never":
      - infection_time == -1: never infected
      - recovery_time == -1: never recovered / never infected
      - infector == -1: no recorded infector, including initial seeds

    Initial seed infections have infection_time == 0.
    """

    counts: NDArray[np.int64]
    infection_time: NDArray[np.int64]
    recovery_time: NDArray[np.int64]
    infector: NDArray[np.int64]
    seeds: NDArray[np.int64]

    @property
    def n_nodes(self) -> int:
        return self.infection_time.size

    @property
    def n_steps(self) -> int:
        """Number of simulation steps until no infectious nodes remain."""
        return len(self.counts) - 1

    @property
    def ever_infected(self) -> NDArray[np.bool_]:
        """Boolean mask of nodes infected at any point, including seeds."""
        return self.infection_time >= 0

    @property
    def final_size(self) -> int:
        """Number of nodes ever infected, including the initial seeds."""
        return int(np.count_nonzero(self.ever_infected))

    @property
    def offspring(self) -> NDArray[np.int64]:
        """Number of direct secondary infections caused by each node."""
        infectors = self.infector[self.infector >= 0]
        return np.bincount(
            infectors,
            minlength=self.n_nodes,
        ).astype(np.int64)


def _edge_multiplier_array(
    graph: Graph,
    edge_multipliers: ArrayLike | None,
) -> NDArray[np.float64]:
    """
    Validate and return one multiplier per directed CSR graph entry.
    """

    if edge_multipliers is None:
        return np.ones(
            graph.indices.size,
            dtype=np.float64,
        )

    values = np.asarray(
        edge_multipliers,
        dtype=np.float64,
    )

    if values.shape != (graph.indices.size,):
        raise ValueError(
            "edge_multipliers must contain one value per directed "
            "graph entry"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("edge_multipliers must be finite")

    if np.any(values < 0):
        raise ValueError("edge_multipliers cannot be negative")

    return values


def step(
    graph: Graph,
    state: NDArray[np.int8],
    params: NodeParams,
    p: float,
    rng: np.random.Generator,
    edge_multipliers: ArrayLike | None = None,
) -> StepEvents:
    """
    Compute epidemic events for one step t -> t+1.

    The input state is not modified.

    For every infectious-susceptible edge (u, v), transmission is sampled
    independently with probability

        min(1, p * tau_u * sigma_v * w_uv).

    If several infectious neighbours successfully infect the same target
    during the same step, successful pairs are randomly permuted and the
    first success is retained as the recorded infector.
    """

    edge_multiplier_values = _edge_multiplier_array(
        graph,
        edge_multipliers,
    )

    infected = np.flatnonzero(
        state == State.I
    )

    src_parts: list[NDArray[np.int64]] = []
    dst_parts: list[NDArray[np.int64]] = []
    multiplier_parts: list[NDArray[np.float64]] = []

    for source in infected:
        start = graph.indptr[source]
        stop = graph.indptr[source + 1]

        neighbours = graph.indices[start:stop]
        local_multipliers = edge_multiplier_values[start:stop]

        susceptible_mask = (
            state[neighbours] == State.S
        )

        susceptible = neighbours[susceptible_mask]
        susceptible_multipliers = local_multipliers[
            susceptible_mask
        ]

        if susceptible.size == 0:
            continue

        src_parts.append(
            np.full(
                susceptible.size,
                source,
                dtype=np.int64,
            )
        )

        dst_parts.append(
            susceptible.astype(
                np.int64,
                copy=False,
            )
        )

        multiplier_parts.append(
            susceptible_multipliers.astype(
                np.float64,
                copy=False,
            )
        )

    src = (
        np.concatenate(src_parts)
        if src_parts
        else _EMPTY_INT
    )

    dst = (
        np.concatenate(dst_parts)
        if dst_parts
        else _EMPTY_INT
    )

    pair_multipliers = (
        np.concatenate(multiplier_parts)
        if multiplier_parts
        else _EMPTY_FLOAT
    )

    probabilities = transmission_probability(
        p,
        params,
        src,
        dst,
        edge_multiplier=pair_multipliers,
    )

    successful = (
        rng.random(src.size)
        < probabilities
    )

    successful_src = src[successful]
    successful_dst = dst[successful]

    # Resolve multiple successful infectors for the same target randomly.
    order = rng.permutation(
        successful_src.size
    )

    successful_src = successful_src[order]
    successful_dst = successful_dst[order]

    newly_infected, first = np.unique(
        successful_dst,
        return_index=True,
    )

    infectors = successful_src[first]

    recovers = (
        rng.random(infected.size)
        < params.recovery[infected]
    )

    newly_recovered = infected[
        recovers
    ].astype(
        np.int64,
        copy=False,
    )

    return StepEvents(
        newly_infected=newly_infected.astype(
            np.int64,
            copy=False,
        ),
        infectors=infectors.astype(
            np.int64,
            copy=False,
        ),
        newly_recovered=newly_recovered,
    )


def apply_events(
    state: NDArray[np.int8],
    events: StepEvents,
) -> None:
    """Apply one step's infection and recovery events in place."""

    state[events.newly_infected] = State.I
    state[events.newly_recovered] = State.R


def simulate(
    graph: Graph,
    seeds: ArrayLike,
    p: float,
    params: NodeParams | None = None,
    *,
    edge_multipliers: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    max_steps: int = 10_000,
) -> SimulationResult:
    """
    Run one discrete-time SIR epidemic until no infectious nodes remain.

    Parameters
    ----------
    graph:
        Contact network.

    seeds:
        Initial infectious node IDs.

    p:
        Global transmission parameter.

    params:
        Per-node susceptibility, infectiousness, and recovery probability.
        If omitted, homogeneous parameters are used.

    edge_multipliers:
        One contact-specific transmission multiplier per directed CSR entry.

    rng:
        NumPy generator or integer seed.

    max_steps:
        Safety limit on epidemic duration.
    """

    n = graph.n_nodes

    if not 0 <= p <= 1:
        raise ValueError(
            f"p must be in [0, 1], got {p}"
        )

    seed_nodes = np.asarray(
        seeds,
        dtype=np.int64,
    )

    if seed_nodes.ndim != 1:
        raise ValueError(
            f"seeds must be 1-D, got {seed_nodes.ndim}-D"
        )

    if np.any(
        (seed_nodes < 0)
        | (seed_nodes >= n)
    ):
        raise ValueError(
            f"seeds must be in [0, {n}), got {seed_nodes}"
        )

    if np.unique(seed_nodes).size != seed_nodes.size:
        raise ValueError(
            f"seeds must be distinct, got {seed_nodes}"
        )

    if params is None:
        params = NodeParams.homogeneous(n)

    if params.n_nodes != n:
        raise ValueError(
            f"params has {params.n_nodes} nodes, graph has {n}"
        )

    # Validate once here so invalid edge weights fail before simulation.
    edge_multiplier_values = _edge_multiplier_array(
        graph,
        edge_multipliers,
    )

    rng = np.random.default_rng(rng)

    state = np.full(
        n,
        State.S,
        dtype=STATE_DTYPE,
    )
    state[seed_nodes] = State.I

    infection_time = np.full(
        n,
        -1,
        dtype=np.int64,
    )
    recovery_time = np.full(
        n,
        -1,
        dtype=np.int64,
    )
    infector = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    infection_time[seed_nodes] = 0

    counts = [
        np.bincount(
            state,
            minlength=3,
        )
    ]

    t = 0

    while np.any(state == State.I):
        if t >= max_steps:
            n_infected = int(
                counts[-1][State.I]
            )
            raise RuntimeError(
                f"still {n_infected} infected after {t} steps"
            )

        events = step(
            graph,
            state,
            params,
            p,
            rng,
            edge_multipliers=edge_multiplier_values,
        )

        apply_events(
            state,
            events,
        )

        t += 1

        infection_time[
            events.newly_infected
        ] = t

        infector[
            events.newly_infected
        ] = events.infectors

        recovery_time[
            events.newly_recovered
        ] = t

        counts.append(
            np.bincount(
                state,
                minlength=3,
            )
        )

    return SimulationResult(
        counts=np.asarray(
            counts,
            dtype=np.int64,
        ),
        infection_time=infection_time,
        recovery_time=recovery_time,
        infector=infector,
        seeds=np.sort(seed_nodes),
    )
