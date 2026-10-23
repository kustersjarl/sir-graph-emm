from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from sir_graph.dynamics import SimulationResult
from sir_graph.graph import Graph


__all__ = [
    "ModelAObservations",
    "ModelBObservations",
    "opportunity_contact_steps",
    "weighted_opportunity_contact_steps",
    "model_b_observations",
    "model_a_exposure_components",
    "model_a_observations",
    "fit_model_a_hazard",
]


@dataclass(frozen=True)
class ModelAObservations:
    """
    Run-person observations used by Model A.

    Each row corresponds to one non-seed person in one simulation run.

    infected:
        True if the person became naturally infected during the run.

    raw_escape_exposure:
        Number of infectious-neighbour contact-steps survived before
        infection or before the epidemic ended.

    raw_infection_exposure:
        Number of infectious neighbours present on the infection step.
        Zero for never-infected people.

    weighted_escape_exposure:
        Escape exposure weighted by source infectiousness and edge multiplier.

    weighted_infection_exposure:
        Infection-step exposure weighted by source infectiousness and
        edge multiplier.
    """

    run_index: NDArray[np.int64]
    node_index: NDArray[np.int64]
    infected: NDArray[np.bool_]

    raw_escape_exposure: NDArray[np.int64]
    raw_infection_exposure: NDArray[np.int64]

    weighted_escape_exposure: NDArray[np.float64]
    weighted_infection_exposure: NDArray[np.float64]


@dataclass(frozen=True)
class ModelBObservations:
    """
    Run-person observations used by Model B.

    Each row corresponds to one naturally infected, non-seed person in one
    simulation run.

    offspring:
        Number of direct secondary infections caused by the person.

    opportunity:
        Raw susceptible-neighbour contact-steps while infectious.

    weighted_opportunity:
        Susceptible-neighbour contact-steps weighted by recipient
        susceptibility and, when supplied, the contact-specific edge
        multiplier.
    """

    run_index: NDArray[np.int64]
    node_index: NDArray[np.int64]
    offspring: NDArray[np.int64]
    opportunity: NDArray[np.int64]
    weighted_opportunity: NDArray[np.float64]


# ============================================================================
# Model B: raw susceptible-contact opportunity
# ============================================================================


def opportunity_contact_steps(
    graph: Graph,
    run: SimulationResult,
) -> NDArray[np.int64]:
    """
    Count susceptible-neighbour contact-steps for every node in one run.

    A contact-step is counted whenever node i is infectious at the start of
    a simulation step and one of its neighbours is still susceptible. The
    same neighbour may contribute multiple contact-steps if it remains
    susceptible across multiple infectious steps.
    """

    if graph.n_nodes != run.n_nodes:
        raise ValueError(
            f"graph has {graph.n_nodes} nodes but run has {run.n_nodes}"
        )

    infection_time = run.infection_time
    recovery_time = run.recovery_time

    opportunities = np.zeros(run.n_nodes, dtype=np.int64)

    infected_nodes = np.flatnonzero(infection_time >= 0)

    for node in infected_nodes:
        start = int(infection_time[node])
        stop = int(recovery_time[node])

        if stop <= start:
            raise ValueError(
                f"node {node} has invalid infection/recovery times "
                f"({start}, {stop})"
            )

        neighbours = graph.neighbours(int(node))
        if neighbours.size == 0:
            continue

        neighbour_infection_time = infection_time[neighbours]

        # A never-infected neighbour remains susceptible for the full
        # infectious interval. Otherwise it contributes until its infection.
        susceptible_until = np.where(
            neighbour_infection_time < 0,
            stop,
            np.minimum(stop, neighbour_infection_time),
        )

        contact_steps = np.maximum(0, susceptible_until - start)

        opportunities[node] = int(contact_steps.sum())

    return opportunities


# ============================================================================
# Model A: infectious exposure reconstruction
# ============================================================================


def model_a_exposure_components(
    graph: Graph,
    run: SimulationResult,
    infectiousness: ArrayLike,
    *,
    edge_multipliers: ArrayLike | None = None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Reconstruct infectious exposure for every node in one epidemic run.

    For target v at simulation step t, weighted exposure is

        x_v(t) = sum_u tau_u * w_uv

    over infectious neighbours u.

    Returns
    -------
    raw_escape_exposure:
        Infectious-neighbour contact-steps survived before infection/end.

    raw_infection_exposure:
        Number of infectious neighbours on the infection-causing step.

    weighted_escape_exposure:
        Escape exposure weighted by source infectiousness and edge multiplier.

    weighted_infection_exposure:
        Infection-step exposure weighted by source infectiousness and
        edge multiplier.

    Artificial seed nodes are left at zero because they are not valid Model A
    target observations.
    """

    if graph.n_nodes != run.n_nodes:
        raise ValueError(
            f"graph has {graph.n_nodes} nodes but run has {run.n_nodes}"
        )

    infectiousness_array = np.asarray(infectiousness, dtype=np.float64)

    if infectiousness_array.ndim != 1:
        raise ValueError("infectiousness must be a 1-D array")

    if infectiousness_array.shape != (graph.n_nodes,):
        raise ValueError(
            "infectiousness must contain one value per graph node"
        )

    if not np.all(np.isfinite(infectiousness_array)):
        raise ValueError("infectiousness must contain only finite values")

    if np.any(infectiousness_array < 0):
        raise ValueError("infectiousness cannot be negative")

    if edge_multipliers is None:
        edge_multipliers_array = np.ones(
            graph.indices.size,
            dtype=np.float64,
        )
    else:
        edge_multipliers_array = np.asarray(
            edge_multipliers,
            dtype=np.float64,
        )

        if edge_multipliers_array.shape != (graph.indices.size,):
            raise ValueError(
                "edge_multipliers must contain one value per directed "
                "graph entry"
            )

        if not np.all(np.isfinite(edge_multipliers_array)):
            raise ValueError(
                "edge_multipliers must contain only finite values"
            )

        if np.any(edge_multipliers_array < 0):
            raise ValueError("edge_multipliers cannot be negative")

    infection_time = run.infection_time
    recovery_time = run.recovery_time

    raw_total_exposure = np.zeros(graph.n_nodes, dtype=np.int64)
    weighted_total_exposure = np.zeros(graph.n_nodes, dtype=np.float64)

    raw_infection_exposure = np.zeros(graph.n_nodes, dtype=np.int64)
    weighted_infection_exposure = np.zeros(graph.n_nodes, dtype=np.float64)

    infected_sources = np.flatnonzero(infection_time >= 0)

    for source in infected_sources:
        source_start = int(infection_time[source])
        source_stop = int(recovery_time[source])

        if source_stop <= source_start:
            raise ValueError(
                f"node {source} has invalid infection/recovery times "
                f"({source_start}, {source_stop})"
            )

        csr_start = graph.indptr[source]
        csr_stop = graph.indptr[source + 1]

        targets = graph.indices[csr_start:csr_stop]
        source_edge_multipliers = edge_multipliers_array[
            csr_start:csr_stop
        ]

        if targets.size == 0:
            continue

        target_infection_time = infection_time[targets]

        # Seeds are not valid receiving-side Model A observations.
        valid_target = target_infection_time != 0
        if not np.any(valid_target):
            continue

        targets = targets[valid_target]
        target_infection_time = target_infection_time[valid_target]
        source_edge_multipliers = source_edge_multipliers[valid_target]

        # A naturally infected target is susceptible until its infection time.
        # A never-infected target remains susceptible until the epidemic ends.
        target_stop = np.where(
            target_infection_time > 0,
            target_infection_time,
            run.n_steps,
        )

        contact_steps = np.maximum(
            0,
            np.minimum(source_stop, target_stop) - source_start,
        ).astype(np.int64)

        raw_total_exposure[targets] += contact_steps

        transmission_strength = (
            infectiousness_array[source] * source_edge_multipliers
        )

        weighted_total_exposure[targets] += (
            contact_steps * transmission_strength
        )

        # If target infection time is T, infection occurred during T-1 -> T.
        infection_step = target_infection_time - 1

        exposed_on_infection_step = (
            (target_infection_time > 0)
            & (source_start <= infection_step)
            & (infection_step < source_stop)
        )

        infection_targets = targets[exposed_on_infection_step]

        raw_infection_exposure[infection_targets] += 1
        weighted_infection_exposure[infection_targets] += (
            transmission_strength[exposed_on_infection_step]
        )

    raw_escape_exposure = (
        raw_total_exposure - raw_infection_exposure
    )

    weighted_escape_exposure = (
        weighted_total_exposure - weighted_infection_exposure
    )

    if np.any(raw_escape_exposure < 0):
        raise RuntimeError("raw escape exposure became negative")

    if np.any(weighted_escape_exposure < -1e-12):
        raise RuntimeError("weighted escape exposure became negative")

    weighted_escape_exposure = np.maximum(
        weighted_escape_exposure,
        0.0,
    )

    return (
        raw_escape_exposure,
        raw_infection_exposure,
        weighted_escape_exposure,
        weighted_infection_exposure,
    )


def model_a_observations(
    graph: Graph,
    runs: Sequence[SimulationResult],
    *,
    infectiousness: ArrayLike,
    edge_multipliers: ArrayLike | None = None,
) -> ModelAObservations:
    """
    Build Model A run-person observations.

    One observation is created for every non-seed person in every run.
    Naturally infected people have infected=True; never-infected people have
    infected=False. Artificial seed infections are excluded.
    """

    if len(runs) == 0:
        raise ValueError("runs must contain at least one simulation")

    if any(run.n_nodes != graph.n_nodes for run in runs):
        raise ValueError("all simulation results must match the graph size")

    infectiousness_array = np.asarray(
        infectiousness,
        dtype=np.float64,
    )

    if infectiousness_array.shape != (graph.n_nodes,):
        raise ValueError(
            "infectiousness must contain one value per graph node"
        )

    run_parts: list[NDArray[np.int64]] = []
    node_parts: list[NDArray[np.int64]] = []
    infected_parts: list[NDArray[np.bool_]] = []

    raw_escape_parts: list[NDArray[np.int64]] = []
    raw_infection_parts: list[NDArray[np.int64]] = []

    weighted_escape_parts: list[NDArray[np.float64]] = []
    weighted_infection_parts: list[NDArray[np.float64]] = []

    for run_index, run in enumerate(runs):
        (
            raw_escape,
            raw_infection,
            weighted_escape,
            weighted_infection,
        ) = model_a_exposure_components(
            graph,
            run,
            infectiousness_array,
            edge_multipliers=edge_multipliers,
        )

        # infection_time == 0 denotes an artificial seed.
        nodes = np.flatnonzero(
            run.infection_time != 0
        ).astype(np.int64)

        naturally_infected = run.infection_time[nodes] > 0

        # A natural infection must have positive infectious exposure on the
        # infection-causing step.
        if np.any(
            naturally_infected
            & (weighted_infection[nodes] <= 0)
        ):
            raise ValueError(
                "naturally infected node has zero infection-step exposure"
            )

        run_parts.append(
            np.full(nodes.size, run_index, dtype=np.int64)
        )
        node_parts.append(nodes)
        infected_parts.append(
            naturally_infected.astype(np.bool_)
        )

        raw_escape_parts.append(raw_escape[nodes])
        raw_infection_parts.append(raw_infection[nodes])

        weighted_escape_parts.append(weighted_escape[nodes])
        weighted_infection_parts.append(weighted_infection[nodes])

    return ModelAObservations(
        run_index=np.concatenate(run_parts),
        node_index=np.concatenate(node_parts),
        infected=np.concatenate(infected_parts),
        raw_escape_exposure=np.concatenate(raw_escape_parts),
        raw_infection_exposure=np.concatenate(raw_infection_parts),
        weighted_escape_exposure=np.concatenate(weighted_escape_parts),
        weighted_infection_exposure=np.concatenate(
            weighted_infection_parts
        ),
    )


# ============================================================================
# Model A: susceptibility hazard estimation
# ============================================================================


def fit_model_a_hazard(
    infected: ArrayLike,
    escape_exposure: ArrayLike,
    infection_exposure: ArrayLike,
) -> float | None:
    """
    Estimate the Model A susceptibility hazard lambda.

    The likelihood is

        l(lambda)
          = -lambda * sum(X_escape)
            + sum_infected log(1 - exp(-lambda * x_infection))

    Returns 0.0 when there are no infections and None when no finite maximum
    exists.
    """

    infected_array = np.asarray(infected, dtype=np.bool_)
    escape = np.asarray(escape_exposure, dtype=np.float64)
    infection = np.asarray(infection_exposure, dtype=np.float64)

    if (
        infected_array.ndim != 1
        or escape.ndim != 1
        or infection.ndim != 1
    ):
        raise ValueError(
            "infected and exposure arrays must be 1-D"
        )

    if not (
        infected_array.shape
        == escape.shape
        == infection.shape
    ):
        raise ValueError(
            "infected and exposure arrays must have the same shape"
        )

    if not np.all(np.isfinite(escape)):
        raise ValueError(
            "escape_exposure must contain only finite values"
        )

    if not np.all(np.isfinite(infection)):
        raise ValueError(
            "infection_exposure must contain only finite values"
        )

    if np.any(escape < 0):
        raise ValueError("escape_exposure cannot be negative")

    if np.any(infection < 0):
        raise ValueError("infection_exposure cannot be negative")

    if np.any(infection[~infected_array] != 0):
        raise ValueError(
            "never-infected observations must have zero "
            "infection_exposure"
        )

    infection_exposure_observed = infection[infected_array]

    if np.any(infection_exposure_observed <= 0):
        raise ValueError(
            "infected observations must have positive "
            "infection_exposure"
        )

    n_infections = int(infected_array.sum())

    if n_infections == 0:
        return 0.0

    total_escape_exposure = float(escape.sum())

    # If nobody escaped any exposure, the likelihood increases toward
    # lambda = infinity and there is no finite MLE.
    if total_escape_exposure <= 0:
        return None

    def score(hazard: float) -> float:
        z = hazard * infection_exposure_observed

        with np.errstate(
            over="ignore",
            divide="ignore",
            invalid="ignore",
        ):
            denominator = np.expm1(z)
            infection_term = np.divide(
                infection_exposure_observed,
                denominator,
                out=np.zeros_like(
                    infection_exposure_observed
                ),
                where=denominator != 0,
            )

        return float(
            -total_escape_exposure + infection_term.sum()
        )

    lower = 1e-12
    upper = 1.0

    while score(upper) > 0:
        upper *= 2.0
        if upper > 1e12:
            return None

    lambda_hat = brentq(
        score,
        lower,
        upper,
        xtol=1e-12,
        rtol=1e-12,
    )

    return float(lambda_hat)


# ============================================================================
# Model B: susceptibility- and edge-weighted opportunity
# ============================================================================


def weighted_opportunity_contact_steps(
    graph: Graph,
    run: SimulationResult,
    susceptibility: ArrayLike,
    *,
    edge_multipliers: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute weighted transmission opportunity for every node in one run.

    For infectious node i,

        O*_i = sum_j contact_steps_ij * sigma_j * w_ij

    where j ranges over susceptible neighbours during i's infectious period.
    """

    if graph.n_nodes != run.n_nodes:
        raise ValueError(
            f"graph has {graph.n_nodes} nodes but run has {run.n_nodes}"
        )

    susceptibility_array = np.asarray(
        susceptibility,
        dtype=np.float64,
    )

    if susceptibility_array.ndim != 1:
        raise ValueError("susceptibility must be a 1-D array")

    if susceptibility_array.shape != (run.n_nodes,):
        raise ValueError(
            "susceptibility must contain one value per graph node"
        )

    if not np.all(np.isfinite(susceptibility_array)):
        raise ValueError(
            "susceptibility must contain only finite values"
        )

    if np.any(susceptibility_array < 0):
        raise ValueError("susceptibility cannot be negative")

    if edge_multipliers is None:
        edge_multipliers_array = np.ones(
            graph.indices.size,
            dtype=np.float64,
        )
    else:
        edge_multipliers_array = np.asarray(
            edge_multipliers,
            dtype=np.float64,
        )

        if edge_multipliers_array.shape != (graph.indices.size,):
            raise ValueError(
                "edge_multipliers must contain one value per directed "
                "graph entry"
            )

        if not np.all(np.isfinite(edge_multipliers_array)):
            raise ValueError(
                "edge_multipliers must contain only finite values"
            )

        if np.any(edge_multipliers_array < 0):
            raise ValueError("edge_multipliers cannot be negative")

    infection_time = run.infection_time
    recovery_time = run.recovery_time

    weighted_opportunities = np.zeros(
        run.n_nodes,
        dtype=np.float64,
    )

    infected_nodes = np.flatnonzero(infection_time >= 0)

    for node in infected_nodes:
        start = int(infection_time[node])
        stop = int(recovery_time[node])

        if stop <= start:
            raise ValueError(
                f"node {node} has invalid infection/recovery times "
                f"({start}, {stop})"
            )

        start_index = graph.indptr[int(node)]
        stop_index = graph.indptr[int(node) + 1]

        neighbours = graph.indices[start_index:stop_index]
        neighbour_edge_multipliers = edge_multipliers_array[
            start_index:stop_index
        ]

        if neighbours.size == 0:
            continue

        neighbour_infection_time = infection_time[neighbours]

        susceptible_until = np.where(
            neighbour_infection_time < 0,
            stop,
            np.minimum(stop, neighbour_infection_time),
        )

        contact_steps = np.maximum(
            0,
            susceptible_until - start,
        ).astype(np.float64)

        neighbour_susceptibility = susceptibility_array[
            neighbours
        ]

        weighted_opportunities[node] = float(
            np.sum(
                contact_steps
                * neighbour_susceptibility
                * neighbour_edge_multipliers
            )
        )

    return weighted_opportunities


def model_b_observations(
    graph: Graph,
    runs: Sequence[SimulationResult],
    *,
    susceptibility: ArrayLike,
    edge_multipliers: ArrayLike | None = None,
) -> ModelBObservations:
    """
    Build Model B run-person observations.

    One observation is created for every naturally infected, non-seed person
    in every run. Model B uses direct secondary infections as the outcome and
    realised susceptible-neighbour opportunity as the denominator.

    Zero-opportunity observations are retained.
    """

    if len(runs) == 0:
        raise ValueError("runs must contain at least one simulation")

    if any(run.n_nodes != graph.n_nodes for run in runs):
        raise ValueError("all simulation results must match the graph size")

    susceptibility_array = np.asarray(
        susceptibility,
        dtype=np.float64,
    )

    if susceptibility_array.ndim != 1:
        raise ValueError("susceptibility must be a 1-D array")

    if susceptibility_array.shape != (graph.n_nodes,):
        raise ValueError(
            "susceptibility must contain one value per graph node"
        )

    if not np.all(np.isfinite(susceptibility_array)):
        raise ValueError(
            "susceptibility must contain only finite values"
        )

    if np.any(susceptibility_array < 0):
        raise ValueError("susceptibility cannot be negative")

    run_parts: list[NDArray[np.int64]] = []
    node_parts: list[NDArray[np.int64]] = []
    offspring_parts: list[NDArray[np.int64]] = []
    opportunity_parts: list[NDArray[np.int64]] = []
    weighted_opportunity_parts: list[NDArray[np.float64]] = []

    for run_index, run in enumerate(runs):
        # infection_time == 0 denotes an artificial seed.
        natural_nodes = np.flatnonzero(
            run.infection_time > 0
        ).astype(np.int64)

        if natural_nodes.size == 0:
            continue

        opportunities = opportunity_contact_steps(
            graph,
            run,
        )

        weighted_opportunities = weighted_opportunity_contact_steps(
            graph,
            run,
            susceptibility_array,
            edge_multipliers=edge_multipliers,
        )

        offspring = run.offspring

        run_parts.append(
            np.full(
                natural_nodes.size,
                run_index,
                dtype=np.int64,
            )
        )
        node_parts.append(natural_nodes)
        offspring_parts.append(offspring[natural_nodes])
        opportunity_parts.append(opportunities[natural_nodes])
        weighted_opportunity_parts.append(
            weighted_opportunities[natural_nodes]
        )

    if not node_parts:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)

        return ModelBObservations(
            run_index=empty_int.copy(),
            node_index=empty_int.copy(),
            offspring=empty_int.copy(),
            opportunity=empty_int.copy(),
            weighted_opportunity=empty_float.copy(),
        )

    return ModelBObservations(
        run_index=np.concatenate(run_parts),
        node_index=np.concatenate(node_parts),
        offspring=np.concatenate(offspring_parts),
        opportunity=np.concatenate(opportunity_parts),
        weighted_opportunity=np.concatenate(
            weighted_opportunity_parts
        ),
    )
