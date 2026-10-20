from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from sir_graph.dynamics import SimulationResult
from sir_graph.graph import Graph


__all__ = [
    "DegreeBaseline",
    "ModelBObservations",
    "CascadeObservations",
    "per_node_spreading_metrics",
    "fit_degree_baseline",
    "opportunity_contact_steps",
    "weighted_opportunity_contact_steps",
    "descendant_counts",
    "model_b_observations",
    "cascade_observations",
    "aggregate_cascade_by_person",
]


@dataclass(frozen=True)
class DegreeBaseline:
    """
    Linear baseline for expected spreading behaviour from node degree.
    """

    slope: float
    intercept: float
    expected: NDArray[np.float64]
    residuals: NDArray[np.float64]


@dataclass(frozen=True)
class ModelBObservations:
    """
    Run-level observations used by Model B.

    Each row corresponds to one naturally infected node
    in one simulation run.

    Artificial seed infections are excluded.

    opportunity:
        Raw susceptible-neighbour contact-steps.

    weighted_opportunity:
        Susceptibility-weighted susceptible-neighbour
        contact-steps.

    offspring:
        Number of direct secondary infections.

    descendants:
        Number of all downstream infections in the
        transmission tree.
    """

    run_index: NDArray[np.int64]
    node_index: NDArray[np.int64]
    offspring: NDArray[np.int64]

    opportunity: NDArray[np.int64]

    weighted_opportunity: NDArray[np.float64]

    descendants: NDArray[np.int64]


@dataclass(frozen=True)
class CascadeObservations:
    """
    Run-person observations for downstream cascade analysis.

    Each row corresponds to one naturally infected node in one
    simulation run. Artificial seed infections are excluded.

    descendants:
        All downstream infections in the transmission tree, including
        direct offspring.

    indirect_descendants:
        Downstream infections beyond the first generation:

            descendants - offspring

    timing_adjusted:
        log(1 + descendants) minus the pooled mean among observations
        infected at the same simulation time.

    timing_adjusted_indirect:
        log(1 + indirect_descendants) minus the pooled mean among
        observations infected at the same simulation time.
    """

    run_index: NDArray[np.int64]
    node_index: NDArray[np.int64]
    infection_time: NDArray[np.int64]

    descendants: NDArray[np.int64]
    indirect_descendants: NDArray[np.int64]

    log_descendants: NDArray[np.float64]
    log_indirect_descendants: NDArray[np.float64]

    expected_log_descendants: NDArray[np.float64]
    expected_log_indirect_descendants: NDArray[np.float64]

    timing_adjusted: NDArray[np.float64]
    timing_adjusted_indirect: NDArray[np.float64]


# ============================================================
# Original per-node spreading baseline
# ============================================================


def per_node_spreading_metrics(
    runs: Sequence[SimulationResult],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """
    Aggregate spreading behaviour across epidemic simulations.

    Seeds are excluded because they were artificially infected.

    Returns
    -------
    n_natural_infections:
        Number of runs in which each node became
        naturally infected.

    mean_secondary_when_infected:
        Mean number of direct secondary infections
        caused by each node, conditional on that node
        being naturally infected.
    """

    if len(runs) == 0:
        raise ValueError(
            "runs must contain at least one simulation"
        )

    n_nodes = runs[0].n_nodes

    if any(
        run.n_nodes != n_nodes
        for run in runs
    ):
        raise ValueError(
            "all simulation results must have "
            "the same number of nodes"
        )

    infection_times = np.stack(
        [
            run.infection_time
            for run in runs
        ]
    )

    offspring = np.stack(
        [
            run.offspring
            for run in runs
        ]
    )

    # > 0 means naturally infected.
    #
    # 0 means the node was artificially
    # selected as a seed.
    naturally_infected = (
        infection_times > 0
    )

    n_natural_infections = (
        naturally_infected.sum(
            axis=0,
            dtype=np.int64,
        )
    )

    total_secondary = np.where(
        naturally_infected,
        offspring,
        0,
    ).sum(
        axis=0,
        dtype=np.int64,
    )

    mean_secondary_when_infected = np.divide(
        total_secondary,
        n_natural_infections,
        out=np.full(
            n_nodes,
            np.nan,
            dtype=np.float64,
        ),
        where=n_natural_infections > 0,
    )

    return (
        n_natural_infections,
        mean_secondary_when_infected,
    )


def fit_degree_baseline(
    degrees: ArrayLike,
    spread: ArrayLike,
) -> DegreeBaseline:
    """
    Fit expected spreading behaviour from node degree.

    Fits:

        expected_spread =
            intercept + slope * degree

    Residual:

        actual_spread - expected_spread

    Positive residual:
        node spreads more than expected
        for its degree.

    Negative residual:
        node spreads less than expected
        for its degree.
    """

    x = np.asarray(
        degrees,
        dtype=np.float64,
    )

    y = np.asarray(
        spread,
        dtype=np.float64,
    )

    if (
        x.ndim != 1
        or y.ndim != 1
    ):
        raise ValueError(
            "degrees and spread must be "
            "1-D arrays"
        )

    if x.shape != y.shape:
        raise ValueError(
            f"degrees and spread must have "
            f"the same shape, got "
            f"{x.shape} and {y.shape}"
        )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 2:
        raise ValueError(
            "at least two valid observations "
            "are required"
        )

    if (
        np.unique(
            x[valid]
        ).size < 2
    ):
        raise ValueError(
            "degree must contain at least "
            "two distinct values"
        )

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    expected = (
        intercept
        + slope * x
    )

    residuals = (
        y
        - expected
    )

    return DegreeBaseline(
        slope=float(slope),
        intercept=float(intercept),
        expected=expected,
        residuals=residuals,
    )


# ============================================================
# Model B:
# raw susceptible-contact opportunity
# ============================================================


def opportunity_contact_steps(
    graph: Graph,
    run: SimulationResult,
) -> NDArray[np.int64]:
    """
    Count susceptible-neighbour contact-steps
    for every node in one simulation run.

    A contact-step is counted whenever node i
    is infectious at the start of a simulation
    step and one of its neighbours is still
    susceptible.

    The same neighbour may contribute multiple
    contact-steps if it remains susceptible
    across multiple infectious time steps.

    This reconstructs opportunity from recorded
    infection and recovery times.
    """

    if graph.n_nodes != run.n_nodes:
        raise ValueError(
            f"graph has {graph.n_nodes} nodes "
            f"but run has {run.n_nodes}"
        )

    infection_time = (
        run.infection_time
    )

    recovery_time = (
        run.recovery_time
    )

    opportunities = np.zeros(
        run.n_nodes,
        dtype=np.int64,
    )

    infected_nodes = np.flatnonzero(
        infection_time >= 0
    )

    for node in infected_nodes:

        start = int(
            infection_time[node]
        )

        stop = int(
            recovery_time[node]
        )

        if stop <= start:
            raise ValueError(
                f"node {node} has invalid "
                f"infection/recovery times "
                f"({start}, {stop})"
            )

        neighbours = graph.neighbours(
            int(node)
        )

        if neighbours.size == 0:
            continue

        neighbour_infection_time = (
            infection_time[
                neighbours
            ]
        )

        # If neighbour j is infected at
        # time t_j, it was susceptible
        # during states:
        #
        #     start, ..., t_j - 1
        #
        # A never-infected neighbour remains
        # susceptible for the entire infectious
        # interval of node i.
        susceptible_until = np.where(
            neighbour_infection_time < 0,
            stop,
            np.minimum(
                stop,
                neighbour_infection_time,
            ),
        )

        contact_steps = np.maximum(
            0,
            susceptible_until - start,
        )

        opportunities[node] = int(
            contact_steps.sum()
        )

    return opportunities


# ============================================================
# Model B:
# susceptibility-weighted opportunity
# ============================================================


def weighted_opportunity_contact_steps(
    graph: Graph,
    run: SimulationResult,
    susceptibility: ArrayLike,
    *,
    edge_multipliers: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute simulator-aligned weighted transmission opportunity
    for every node in one run.

    For infectious node i:

        O*_i =
            sum_j (
                contact_steps_ij
                * susceptibility_j
                * edge_multiplier_ij
            )

    where j ranges over neighbours of i.

    ``edge_multipliers`` must follow the same directed CSR-entry
    ordering as ``graph.indices``. If it is omitted, all edges are
    assigned multiplier 1.0, preserving the previous behaviour.

    This matches the simulator more closely because transmission
    probability is proportional to both the receiving node's
    susceptibility and the contact-specific edge multiplier.
    """

    if graph.n_nodes != run.n_nodes:
        raise ValueError(
            f"graph has {graph.n_nodes} nodes "
            f"but run has {run.n_nodes}"
        )

    susceptibility = np.asarray(
        susceptibility,
        dtype=np.float64,
    )

    if susceptibility.ndim != 1:
        raise ValueError(
            "susceptibility must be a 1-D array"
        )

    if susceptibility.shape != (
        run.n_nodes,
    ):
        raise ValueError(
            "susceptibility must contain "
            "one value per graph node"
        )

    if not np.all(
        np.isfinite(
            susceptibility
        )
    ):
        raise ValueError(
            "susceptibility must contain "
            "only finite values"
        )

    if np.any(
        susceptibility < 0
    ):
        raise ValueError(
            "susceptibility cannot be negative"
        )

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

        if edge_multipliers_array.shape != (
            graph.indices.size,
        ):
            raise ValueError(
                "edge_multipliers must contain "
                "one value per directed graph entry"
            )

        if not np.all(
            np.isfinite(
                edge_multipliers_array
            )
        ):
            raise ValueError(
                "edge_multipliers must contain "
                "only finite values"
            )

        if np.any(
            edge_multipliers_array < 0
        ):
            raise ValueError(
                "edge_multipliers cannot be negative"
            )

    infection_time = (
        run.infection_time
    )

    recovery_time = (
        run.recovery_time
    )

    weighted_opportunities = np.zeros(
        run.n_nodes,
        dtype=np.float64,
    )

    infected_nodes = np.flatnonzero(
        infection_time >= 0
    )

    for node in infected_nodes:

        start = int(
            infection_time[node]
        )

        stop = int(
            recovery_time[node]
        )

        if stop <= start:
            raise ValueError(
                f"node {node} has invalid "
                f"infection/recovery times "
                f"({start}, {stop})"
            )

        start_index = graph.indptr[
            int(node)
        ]

        stop_index = graph.indptr[
            int(node) + 1
        ]

        neighbours = graph.indices[
            start_index:stop_index
        ]

        neighbour_edge_multipliers = (
            edge_multipliers_array[
                start_index:stop_index
            ]
        )

        if neighbours.size == 0:
            continue

        neighbour_infection_time = (
            infection_time[
                neighbours
            ]
        )

        susceptible_until = np.where(
            neighbour_infection_time < 0,
            stop,
            np.minimum(
                stop,
                neighbour_infection_time,
            ),
        )

        contact_steps = np.maximum(
            0,
            susceptible_until - start,
        ).astype(
            np.float64
        )

        neighbour_susceptibility = (
            susceptibility[
                neighbours
            ]
        )

        weighted_opportunities[node] = float(
            np.sum(
                contact_steps
                * neighbour_susceptibility
                * neighbour_edge_multipliers
            )
        )

    return weighted_opportunities


# ============================================================
# Transmission-tree cascade outcome
# ============================================================


def descendant_counts(
    run: SimulationResult,
) -> NDArray[np.int64]:
    """
    Count all downstream transmission-tree
    descendants of every node.

    Direct offspring count as descendants,
    as do infections in all later generations
    descending from those offspring.

    Never-infected nodes have zero descendants.
    """

    descendants = np.zeros(
        run.n_nodes,
        dtype=np.int64,
    )

    infected_nodes = np.flatnonzero(
        run.infection_time >= 0
    )

    # Parents are always infected before
    # children.
    #
    # Working backwards allows each child
    # to pass its complete subtree size
    # to its parent.
    order = infected_nodes[
        np.argsort(
            run.infection_time[
                infected_nodes
            ]
        )[::-1]
    ]

    for node in order:

        parent = int(
            run.infector[node]
        )

        if parent >= 0:
            descendants[parent] += (
                1
                + descendants[node]
            )

    return descendants


# ============================================================
# Build Model B run-person observations
# ============================================================


def model_b_observations(
    graph: Graph,
    runs: Sequence[SimulationResult],
    *,
    susceptibility: ArrayLike,
    edge_multipliers: ArrayLike | None = None,
) -> ModelBObservations:
    """
    Build run-level Model B observations
    from an epidemic ensemble.

    One observation is created for every
    naturally infected node in every run.

    Artificial seed infections are excluded.

    Returned outcomes:

        offspring
            Direct secondary infections.

        opportunity
            Raw susceptible-neighbour
            contact-steps.

        weighted_opportunity
            Contact-steps weighted by each
            susceptible neighbour's susceptibility
            and, when supplied, the corresponding
            contact-specific edge multiplier.

        descendants
            Full downstream cascade size.

    Zero-opportunity observations are retained.

    ``edge_multipliers`` must follow the same directed CSR-entry
    ordering as ``graph.indices``. If omitted, all edges have
    multiplier 1.0.
    """

    if len(runs) == 0:
        raise ValueError(
            "runs must contain at least "
            "one simulation"
        )

    if any(
        run.n_nodes != graph.n_nodes
        for run in runs
    ):
        raise ValueError(
            "all simulation results must "
            "match the graph size"
        )

    susceptibility = np.asarray(
        susceptibility,
        dtype=np.float64,
    )

    if susceptibility.ndim != 1:
        raise ValueError(
            "susceptibility must be a "
            "1-D array"
        )

    if susceptibility.shape != (
        graph.n_nodes,
    ):
        raise ValueError(
            "susceptibility must contain "
            "one value per graph node"
        )

    if not np.all(
        np.isfinite(
            susceptibility
        )
    ):
        raise ValueError(
            "susceptibility must contain "
            "only finite values"
        )

    if np.any(
        susceptibility < 0
    ):
        raise ValueError(
            "susceptibility cannot be negative"
        )

    if edge_multipliers is not None:
        edge_multipliers = np.asarray(
            edge_multipliers,
            dtype=np.float64,
        )

        if edge_multipliers.shape != (
            graph.indices.size,
        ):
            raise ValueError(
                "edge_multipliers must contain "
                "one value per directed graph entry"
            )

        if not np.all(
            np.isfinite(
                edge_multipliers
            )
        ):
            raise ValueError(
                "edge_multipliers must contain "
                "only finite values"
            )

        if np.any(
            edge_multipliers < 0
        ):
            raise ValueError(
                "edge_multipliers cannot be negative"
            )

    run_parts: list[
        NDArray[np.int64]
    ] = []

    node_parts: list[
        NDArray[np.int64]
    ] = []

    offspring_parts: list[
        NDArray[np.int64]
    ] = []

    opportunity_parts: list[
        NDArray[np.int64]
    ] = []

    weighted_opportunity_parts: list[
        NDArray[np.float64]
    ] = []

    descendant_parts: list[
        NDArray[np.int64]
    ] = []

    for run_index, run in enumerate(
        runs
    ):

        # Naturally infected only.
        #
        # infection_time == 0 denotes
        # an artificial seed.
        natural_nodes = np.flatnonzero(
            run.infection_time > 0
        ).astype(
            np.int64
        )

        if natural_nodes.size == 0:
            continue

        opportunities = (
            opportunity_contact_steps(
                graph,
                run,
            )
        )

        weighted_opportunities = (
            weighted_opportunity_contact_steps(
                graph,
                run,
                susceptibility,
                edge_multipliers=
                    edge_multipliers,
            )
        )

        descendants = (
            descendant_counts(
                run
            )
        )

        offspring = (
            run.offspring
        )

        run_parts.append(
            np.full(
                natural_nodes.size,
                run_index,
                dtype=np.int64,
            )
        )

        node_parts.append(
            natural_nodes
        )

        offspring_parts.append(
            offspring[
                natural_nodes
            ]
        )

        opportunity_parts.append(
            opportunities[
                natural_nodes
            ]
        )

        weighted_opportunity_parts.append(
            weighted_opportunities[
                natural_nodes
            ]
        )

        descendant_parts.append(
            descendants[
                natural_nodes
            ]
        )

    if not node_parts:

        empty_int = np.empty(
            0,
            dtype=np.int64,
        )

        empty_float = np.empty(
            0,
            dtype=np.float64,
        )

        return ModelBObservations(
            run_index=empty_int.copy(),
            node_index=empty_int.copy(),
            offspring=empty_int.copy(),
            opportunity=empty_int.copy(),
            weighted_opportunity=(
                empty_float.copy()
            ),
            descendants=empty_int.copy(),
        )

    return ModelBObservations(
        run_index=np.concatenate(
            run_parts
        ),
        node_index=np.concatenate(
            node_parts
        ),
        offspring=np.concatenate(
            offspring_parts
        ),
        opportunity=np.concatenate(
            opportunity_parts
        ),
        weighted_opportunity=np.concatenate(
            weighted_opportunity_parts
        ),
        descendants=np.concatenate(
            descendant_parts
        ),
    )

# ============================================================
# Cascade observations and person-level aggregation
# ============================================================


def cascade_observations(
    runs: Sequence[SimulationResult],
) -> CascadeObservations:
    """
    Build timing-adjusted downstream-cascade observations.

    One row is created for every naturally infected node in every run.
    Seeds are excluded because their infection was imposed by the
    experiment.

    For observation (i, r):

        D_ir = all descendants of i in the transmission tree
        I_ir = D_ir - direct offspring of i

    The transformed outcomes are:

        Z_ir = log(1 + D_ir)
        ZI_ir = log(1 + I_ir)

    To account for the strong dependence of cascade size on epidemic
    timing, the baseline is pooled over all supplied runs by exact
    infection time t:

        B(t) = mean(Z_ir | infection_time = t)
        BI(t) = mean(ZI_ir | infection_time = t)

    and the timing-adjusted outcomes are:

        C_ir = Z_ir - B(t)
        CI_ir = ZI_ir - BI(t)

    This is a cascade-impact outcome, not a direct infectiousness
    parameter.
    """

    if len(runs) == 0:
        raise ValueError(
            "runs must contain at least one simulation"
        )

    n_nodes = runs[0].n_nodes

    if any(
        run.n_nodes != n_nodes
        for run in runs
    ):
        raise ValueError(
            "all simulation results must have "
            "the same number of nodes"
        )

    run_parts: list[
        NDArray[np.int64]
    ] = []

    node_parts: list[
        NDArray[np.int64]
    ] = []

    infection_time_parts: list[
        NDArray[np.int64]
    ] = []

    descendant_parts: list[
        NDArray[np.int64]
    ] = []

    indirect_parts: list[
        NDArray[np.int64]
    ] = []

    for run_index, run in enumerate(runs):
        natural_nodes = np.flatnonzero(
            run.infection_time > 0
        ).astype(
            np.int64
        )

        if natural_nodes.size == 0:
            continue

        descendants = descendant_counts(
            run
        )

        offspring = run.offspring

        natural_descendants = descendants[
            natural_nodes
        ].astype(
            np.int64,
            copy=False,
        )

        natural_offspring = offspring[
            natural_nodes
        ].astype(
            np.int64,
            copy=False,
        )

        indirect_descendants = (
            natural_descendants
            - natural_offspring
        )

        if np.any(
            indirect_descendants < 0
        ):
            raise ValueError(
                "descendants cannot be smaller "
                "than direct offspring"
            )

        run_parts.append(
            np.full(
                natural_nodes.size,
                run_index,
                dtype=np.int64,
            )
        )

        node_parts.append(
            natural_nodes
        )

        infection_time_parts.append(
            run.infection_time[
                natural_nodes
            ].astype(
                np.int64,
                copy=False,
            )
        )

        descendant_parts.append(
            natural_descendants
        )

        indirect_parts.append(
            indirect_descendants
        )

    if not node_parts:
        empty_int = np.empty(
            0,
            dtype=np.int64,
        )

        empty_float = np.empty(
            0,
            dtype=np.float64,
        )

        return CascadeObservations(
            run_index=empty_int.copy(),
            node_index=empty_int.copy(),
            infection_time=empty_int.copy(),
            descendants=empty_int.copy(),
            indirect_descendants=empty_int.copy(),
            log_descendants=empty_float.copy(),
            log_indirect_descendants=(
                empty_float.copy()
            ),
            expected_log_descendants=(
                empty_float.copy()
            ),
            expected_log_indirect_descendants=(
                empty_float.copy()
            ),
            timing_adjusted=empty_float.copy(),
            timing_adjusted_indirect=(
                empty_float.copy()
            ),
        )

    run_index = np.concatenate(
        run_parts
    )

    node_index = np.concatenate(
        node_parts
    )

    infection_time = np.concatenate(
        infection_time_parts
    )

    descendants = np.concatenate(
        descendant_parts
    )

    indirect_descendants = np.concatenate(
        indirect_parts
    )

    log_descendants = np.log1p(
        descendants.astype(
            np.float64
        )
    )

    log_indirect_descendants = np.log1p(
        indirect_descendants.astype(
            np.float64
        )
    )

    # Pooled exact-infection-time baseline.
    max_time = int(
        infection_time.max()
    )

    time_counts = np.bincount(
        infection_time,
        minlength=max_time + 1,
    ).astype(
        np.int64
    )

    descendant_sums = np.bincount(
        infection_time,
        weights=log_descendants,
        minlength=max_time + 1,
    )

    indirect_sums = np.bincount(
        infection_time,
        weights=log_indirect_descendants,
        minlength=max_time + 1,
    )

    expected_by_time = np.divide(
        descendant_sums,
        time_counts,
        out=np.full(
            max_time + 1,
            np.nan,
            dtype=np.float64,
        ),
        where=time_counts > 0,
    )

    expected_indirect_by_time = np.divide(
        indirect_sums,
        time_counts,
        out=np.full(
            max_time + 1,
            np.nan,
            dtype=np.float64,
        ),
        where=time_counts > 0,
    )

    expected_log_descendants = (
        expected_by_time[
            infection_time
        ]
    )

    expected_log_indirect_descendants = (
        expected_indirect_by_time[
            infection_time
        ]
    )

    timing_adjusted = (
        log_descendants
        - expected_log_descendants
    )

    timing_adjusted_indirect = (
        log_indirect_descendants
        - expected_log_indirect_descendants
    )

    return CascadeObservations(
        run_index=run_index,
        node_index=node_index,
        infection_time=infection_time,
        descendants=descendants,
        indirect_descendants=(
            indirect_descendants
        ),
        log_descendants=log_descendants,
        log_indirect_descendants=(
            log_indirect_descendants
        ),
        expected_log_descendants=(
            expected_log_descendants
        ),
        expected_log_indirect_descendants=(
            expected_log_indirect_descendants
        ),
        timing_adjusted=timing_adjusted,
        timing_adjusted_indirect=(
            timing_adjusted_indirect
        ),
    )


def aggregate_cascade_by_person(
    people: pd.DataFrame,
    observations: CascadeObservations,
) -> pd.DataFrame:
    """
    Aggregate cascade observations to one row per person.

    Returns a copy of ``people`` with:

        cascade_natural_infections
        cascade_mean_descendants
        cascade_mean_log_descendants
        cascade_mean_indirect_descendants
        cascade_mean_log_indirect_descendants
        cascade_mean_timing_adjusted
        cascade_mean_timing_adjusted_indirect

    Means are conditional on natural infection. A person never naturally
    infected receives NaN for the mean outcomes.
    """

    result = people.copy()
    n_people = len(result)

    node_index = np.asarray(
        observations.node_index,
        dtype=np.int64,
    )

    if node_index.ndim != 1:
        raise ValueError(
            "observations.node_index must be 1-D"
        )

    if (
        np.any(node_index < 0)
        or np.any(
            node_index >= n_people
        )
    ):
        raise ValueError(
            "observation node indices are "
            "incompatible with the people dataframe"
        )

    arrays = {
        "descendants": np.asarray(
            observations.descendants,
            dtype=np.float64,
        ),
        "log_descendants": np.asarray(
            observations.log_descendants,
            dtype=np.float64,
        ),
        "indirect_descendants": np.asarray(
            observations.indirect_descendants,
            dtype=np.float64,
        ),
        "log_indirect_descendants": np.asarray(
            observations.log_indirect_descendants,
            dtype=np.float64,
        ),
        "timing_adjusted": np.asarray(
            observations.timing_adjusted,
            dtype=np.float64,
        ),
        "timing_adjusted_indirect": np.asarray(
            observations.timing_adjusted_indirect,
            dtype=np.float64,
        ),
    }

    for name, values in arrays.items():
        if values.shape != node_index.shape:
            raise ValueError(
                f"{name} and node_index must "
                "have the same shape"
            )

        if np.any(
            ~np.isfinite(values)
        ):
            raise ValueError(
                f"{name} must contain only "
                "finite values"
            )

    natural_infections = np.bincount(
        node_index,
        minlength=n_people,
    ).astype(
        np.int64
    )

    def mean_by_person(
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        totals = np.bincount(
            node_index,
            weights=values,
            minlength=n_people,
        )

        return np.divide(
            totals,
            natural_infections,
            out=np.full(
                n_people,
                np.nan,
                dtype=np.float64,
            ),
            where=natural_infections > 0,
        )

    result[
        "cascade_natural_infections"
    ] = natural_infections

    result[
        "cascade_mean_descendants"
    ] = mean_by_person(
        arrays[
            "descendants"
        ]
    )

    result[
        "cascade_mean_log_descendants"
    ] = mean_by_person(
        arrays[
            "log_descendants"
        ]
    )

    result[
        "cascade_mean_indirect_descendants"
    ] = mean_by_person(
        arrays[
            "indirect_descendants"
        ]
    )

    result[
        "cascade_mean_log_indirect_descendants"
    ] = mean_by_person(
        arrays[
            "log_indirect_descendants"
        ]
    )

    result[
        "cascade_mean_timing_adjusted"
    ] = mean_by_person(
        arrays[
            "timing_adjusted"
        ]
    )

    result[
        "cascade_mean_timing_adjusted_indirect"
    ] = mean_by_person(
        arrays[
            "timing_adjusted_indirect"
        ]
    )

    return result
