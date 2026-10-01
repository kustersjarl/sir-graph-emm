from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sir_graph.graph import Graph
from sir_graph.parameters import NodeParams, transmission_probability
from sir_graph.states import STATE_DTYPE, State, choose_seeds

_EMPTY = np.empty(0, dtype=np.int64)


@dataclass(frozen=True, eq=False)
class StepEvents:
    """
    Events of one step, computed from the state at time t.

    `infectors[k]` is the infector of `newly_infected[k]`.
    """

    newly_infected: NDArray[np.int64]
    infectors: NDArray[np.int64]
    newly_recovered: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """
    Full record of one run.

    `counts[t]` holds (S, I, R) at time t. Per-node arrays use -1 for
    "never": `infection_time`, `recovery_time`, and `infector`
    (also -1 for seeds). Together, `infector` forms the transmission tree
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
        """Steps until no node was infected."""
        return len(self.counts) - 1

    @property
    def ever_infected(self) -> NDArray[np.bool_]:
        return self.infection_time >= 0

    @property
    def final_size(self) -> int:
        """Number of nodes ever infected, seeds included."""
        return int(np.count_nonzero(self.ever_infected))

    @property
    def offspring(self) -> NDArray[np.int64]:
        """Secondary infections caused by each node."""
        infectors = self.infector[self.infector >= 0]
        return np.bincount(infectors, minlength=self.n_nodes).astype(np.int64)


def step(
    graph: Graph,
    state: NDArray[np.int8],
    params: NodeParams,
    p: float,
    rng: np.random.Generator,
) -> StepEvents:
    """Compute the events of one step t -> t+1. Does not modify ``state``"""
    infected = np.flatnonzero(state == State.I)

    # All (src, dst) pairs with src infected and dst a susceptible neighbour
    src_parts, dst_parts = [], []
    for i in infected:
        neighbours = graph.indices[graph.indptr[i] : graph.indptr[i + 1]]
        susceptible = neighbours[state[neighbours] == State.S]
        src_parts.append(np.full(susceptible.size, i, dtype=np.int64))
        dst_parts.append(susceptible.astype(np.int64))
    src = np.concatenate(src_parts) if src_parts else _EMPTY
    dst = np.concatenate(dst_parts) if dst_parts else _EMPTY

    # One Bernoulli trial per pair
    success = rng.random(src.size) < transmission_probability(p, params, src, dst)
    src, dst = src[success], dst[success]

    # only keep first infected per target
    order = rng.permutation(src.size)
    src, dst = src[order], dst[order]
    newly_infected, first = np.unique(dst, return_index=True)

    recovers = rng.random(infected.size) < params.recovery[infected]

    return StepEvents(
        newly_infected=newly_infected.astype(np.int64),
        infectors=src[first],
        newly_recovered=infected[recovers].astype(np.int64),
    )


def apply_events(state: NDArray[np.int8], events: StepEvents) -> None:
    """apply `events` to `state` in place"""
    state[events.newly_infected] = State.I
    state[events.newly_recovered] = State.R


def simulate(
    graph: Graph,
    seeds: ArrayLike,
    p: float,
    params: NodeParams | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    max_steps: int = 10_000,
) -> SimulationResult:
    """Run one epidemic from `seeds` until no node is infected"""
    n = graph.n_nodes
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError(f"seeds must be 1-D, got {seeds.ndim}-D")
    if np.any((seeds < 0) | (seeds >= n)):
        raise ValueError(f"seeds must be in [0, {n}), got {seeds}")
    if np.unique(seeds).size != seeds.size:
        raise ValueError(f"seeds must be distinct, got {seeds}")
    if params is None:
        params = NodeParams.homogeneous(n)
    if params.n_nodes != n:
        raise ValueError(f"params has {params.n_nodes} nodes, graph has {n}")

    rng = np.random.default_rng(rng)

    state = np.full(n, State.S, dtype=STATE_DTYPE)
    state[seeds] = State.I
    infection_time = np.full(n, -1, dtype=np.int64)
    recovery_time = np.full(n, -1, dtype=np.int64)
    infector = np.full(n, -1, dtype=np.int64)
    infection_time[seeds] = 0

    counts = [np.bincount(state, minlength=3)]
    t = 0
    while np.any(state == State.I):
        if t >= max_steps:
            n_infected = counts[-1][State.I]
            raise RuntimeError(f"still {n_infected} infected after {t} steps")
        events = step(graph, state, params, p, rng)
        apply_events(state, events)
        t += 1
        infection_time[events.newly_infected] = t
        infector[events.newly_infected] = events.infectors
        recovery_time[events.newly_recovered] = t
        counts.append(np.bincount(state, minlength=3))

    return SimulationResult(
        counts=np.array(counts, dtype=np.int64),
        infection_time=infection_time,
        recovery_time=recovery_time,
        infector=infector,
        seeds=np.sort(seeds),
    )


def run_ensemble(
    graph: Graph,
    n_runs: int,
    n_seeds: int,
    p: float,
    params: NodeParams | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    max_steps: int = 10_000,
) -> list[SimulationResult]:
    """
    Run `n_runs` epidemics on the same graph, with fresh random seeds each

    one shared generator makes the whole ensemble reproducible from `rng`
    """
    rng = np.random.default_rng(rng)
    return [
        simulate(
            graph,
            choose_seeds(graph.n_nodes, n_seeds, rng=rng),
            p,
            params,
            rng=rng,
            max_steps=max_steps,
        )
        for _ in range(n_runs)
    ]
