import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence
from dataclasses import dataclass
from sir_graph.dynamics import SimulationResult
from sir_graph.graph import Graph

_BIG = np.iinfo(np.int64).max // 4

def contact_steps(graph: Graph, result: SimulationResult) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """
    contact-steps per directed edge, reconstructed from a finished run.

    returns (src, dst, c): c[e] = number of time steps in which src[e] was
    infectious while dst[e] was still susceptible.
    """
    src = np.repeat(np.arange(graph.n_nodes), graph.degree())
    dst = graph.indices

    inf = result.infection_time
    rec = np.where(result.recovery_time < 0, _BIG, result.recovery_time)
    inf_dst = np.where(inf < 0, _BIG, inf)

    c = np.maximum(0, np.minimum(rec[src], inf_dst[dst]) - inf[src])
    c[inf[src] < 0] = 0  # never infected, never infectious
    return src, dst, c.astype(np.int64)


def exposure_opportunity(graph: Graph, result: SimulationResult) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    per-node (exposure, opportunity) of one run.

    exposure[v]:sum over steps of the number of infectious neighbours of v while v was susceptible
    opportunity[j]: sum over steps of the number of susceptible neighbours of j while j was infectious
    """
    src, dst, c = contact_steps(graph, result)
    n = graph.n_nodes
    exposure = np.bincount(dst, weights=c, minlength=n)
    opportunity = np.bincount(src, weights=c, minlength=n)
    return np.rint(exposure).astype(np.int64), np.rint(opportunity).astype(np.int64)


@dataclass(frozen=True, eq=False)
class RunMetrics:
    natural: NDArray[np.bool_] # node was infected by transmission (not a seed)
    is_seed: NDArray[np.bool_] # node was a seed in this run
    offspring: NDArray[np.int64] # direct secondary infections caused by the node
    exposure: NDArray[np.int64]
    opportunity: NDArray[np.int64]

    @property
    def n_runs(self) -> int:
        return self.natural.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.natural.shape[1]


def run_metrics(graph: Graph, runs: Sequence[SimulationResult]) -> RunMetrics:
    if len(runs) == 0:
        raise ValueError("runs must contain at least one simulation")
    if any(run.n_nodes != graph.n_nodes for run in runs):
        raise ValueError("all simulation results must match the graph size")

    times = np.stack([run.infection_time for run in runs])
    pairs = [exposure_opportunity(graph, run) for run in runs]
    return RunMetrics(
        natural=times > 0,
        is_seed=times == 0,
        offspring=np.stack([run.offspring for run in runs]),
        exposure=np.stack([e for e, _ in pairs]),
        opportunity=np.stack([o for _, o in pairs]),
    )
