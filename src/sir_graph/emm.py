from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sir_graph.analysis import (
    ModelAObservations,
    ModelBObservations,
    fit_model_a_hazard,
)


__all__ = [
    "Candidate",
    "Description",
    "equal_frequency_cut_points",
    "generate_descriptions",
    "search_descriptions",
    "score_model_a_subgroup",
    "search_model_a_subgroups",
    "score_model_b_subgroup",
    "search_model_b_subgroups",
]


@dataclass(frozen=True)
class Candidate:
    """One possible subgroup condition."""

    description: str
    attribute: str
    mask: NDArray[np.bool_]


@dataclass(frozen=True)
class Description:
    """One evaluable subgroup description of depth 1 or 2."""

    description: str
    depth: int
    mask: NDArray[np.bool_]


def _mask_key(mask: NDArray[np.bool_]) -> bytes:
    """
    Return a stable identity for a subgroup's actual member set.

    Syntactically different descriptions selecting exactly the same people
    therefore receive the same key.
    """

    return np.asarray(mask, dtype=np.bool_).tobytes()


# ============================================================================
# Final local equal-frequency description space
# ============================================================================


def _categorical_conditions(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> list[Candidate]:
    """Generate equality conditions for all categorical descriptors."""

    candidates: list[Candidate] = []

    for column in categorical_columns:
        if column not in df.columns:
            raise KeyError(f"{column!r} not found in dataframe")

        for value in df[column].dropna().unique():
            candidates.append(
                Candidate(
                    description=f"{column} == {value}",
                    attribute=column,
                    mask=(df[column] == value).to_numpy(dtype=np.bool_),
                )
            )

    return candidates


def equal_frequency_cut_points(
    values: NDArray,
    n_bins: int,
    operator: str = ">=",
) -> list[float]:
    """
    Compute equal-frequency cut points following Algorithm 2 of
    Meeng and Knobbe (2021).

    For b = 1, ..., B-1:
        x = floor(n * b / B)

    For the upper-bound operator (<=), x is decreased by one when
    n * b is exactly divisible by B. Bounds are inclusive.

    Duplicate cut points are removed.
    """

    if operator not in (">=", "<="):
        raise ValueError("operator must be '>=' or '<='")

    a = np.sort(np.asarray(values, dtype=np.float64))
    n = a.size

    if n == 0 or n_bins < 2:
        return []

    cut_points: set[float] = set()

    for b in range(1, n_bins):
        x = (n * b) // n_bins

        if operator == "<=" and (n * b) % n_bins == 0:
            x -= 1

        if 0 <= x < n:
            cut_points.add(float(a[x]))

    return sorted(cut_points)


def _local_age_conditions(
    ages: NDArray[np.float64],
    domain: NDArray[np.float64],
    age_column: str,
    n_bins: int,
) -> list[Candidate]:
    """
    Generate local half-bounded age conditions.

    Cut points are recomputed on ``domain``, i.e. on the population covered
    by the subgroup being refined. Conditions that do not split the current
    domain are discarded.
    """

    conditions: list[Candidate] = []

    for operator, select in (
        (">=", np.greater_equal),
        ("<=", np.less_equal),
    ):
        for cut_point in equal_frequency_cut_points(
            domain,
            n_bins,
            operator,
        ):
            in_domain = select(domain, cut_point)

            if in_domain.all() or not in_domain.any():
                continue

            conditions.append(
                Candidate(
                    description=(
                        f"{age_column} {operator} {cut_point:g}"
                    ),
                    attribute=age_column,
                    mask=select(ages, cut_point).astype(
                        np.bool_,
                        copy=False,
                    ),
                )
            )

    return conditions


def generate_descriptions(
    df: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    age_column: str = "age",
    max_depth: int = 2,
    n_bins: int = 10,
) -> list[Description]:
    """
    Enumerate the final local equal-frequency subgroup description space.

    Numeric refinement uses local equal-frequency discretisation. At depth 1,
    age cut points are computed on the whole population. When age refines a
    categorical condition at depth 2, the cut points are recomputed on the
    ages inside that condition.

    Descriptions have depth at most two. Two conditions on the same attribute
    are never combined. Identical person covers are deduplicated.
    """

    if len(df) == 0:
        raise ValueError("df must contain at least one person")

    if max_depth not in (1, 2):
        raise ValueError(
            "current implementation supports max_depth 1 or 2"
        )

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    if age_column not in df.columns:
        raise KeyError(f"{age_column!r} not found in dataframe")

    ages = df[age_column].to_numpy(dtype=np.float64)

    categorical = _categorical_conditions(
        df,
        categorical_columns,
    )

    root_age = _local_age_conditions(
        ages,
        ages,
        age_column,
        n_bins,
    )

    descriptions: list[Description] = []
    seen: set[bytes] = set()

    def add(
        description: str,
        depth: int,
        mask: NDArray[np.bool_],
    ) -> None:
        mask = np.asarray(mask, dtype=np.bool_)
        key = _mask_key(mask)

        if key in seen:
            return

        seen.add(key)

        descriptions.append(
            Description(
                description=description,
                depth=depth,
                mask=mask,
            )
        )

    # Depth 1.
    for condition in categorical:
        add(
            condition.description,
            1,
            condition.mask,
        )

    for condition in root_age:
        add(
            condition.description,
            1,
            condition.mask,
        )

    if max_depth == 1:
        return descriptions

    def pair(a: Candidate, b: Candidate) -> None:
        add(
            f"{a.description} AND {b.description}",
            2,
            a.mask & b.mask,
        )

    # Depth 2: categorical x categorical.
    for a, b in combinations(categorical, 2):
        if a.attribute != b.attribute:
            pair(a, b)

    # Depth 2: a categorical subgroup refined locally by age.
    for categorical_condition in categorical:
        domain = ages[categorical_condition.mask]

        for age_condition in _local_age_conditions(
            ages,
            domain,
            age_column,
            n_bins,
        ):
            pair(
                categorical_condition,
                age_condition,
            )

    # Depth 2: root-level age condition refined by a categorical one.
    # Membership-based deduplication removes equivalent covers.
    for age_condition in root_age:
        for categorical_condition in categorical:
            pair(
                age_condition,
                categorical_condition,
            )

    return descriptions


def search_descriptions(
    descriptions: Sequence[Description],
    scorer,
) -> pd.DataFrame:
    """
    Score every description and return results ordered by decreasing quality.

    ``scorer`` must accept a person-level boolean mask and return either a
    dictionary containing ``quality`` or None.
    """

    results = []

    for description in descriptions:
        score = scorer(description.mask)

        if score is None:
            continue

        results.append(
            {
                "depth": description.depth,
                "description": description.description,
                **score,
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(
            "quality",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _directional_effect(
    log_rate_ratio: float,
    direction: str,
) -> float:
    """Orient a log-rate ratio toward the requested search direction."""

    if direction == "high":
        return log_rate_ratio

    if direction == "low":
        return -log_rate_ratio

    raise ValueError("direction must be 'high' or 'low'")


# ============================================================================
# Model A: exposure-adjusted susceptibility
# ============================================================================


def score_model_a_subgroup(
    mask: NDArray[np.bool_],
    observations: ModelAObservations,
    *,
    min_size: int = 200,
    direction: str = "high",
    exposure_type: str = "weighted",
) -> dict | None:
    """
    Score a subgroup using Model A.

    Model A fits a susceptibility hazard to the subgroup and its complement
    using realised infectious exposure.

    exposure_type="raw":
        Count infectious-neighbour contact-steps.

    exposure_type="weighted":
        Weight exposure by source infectiousness and contact-specific edge
        multiplier.

    Quality:
        sqrt(|G| / N) * oriented log(RR)
    """

    mask = np.asarray(mask, dtype=np.bool_)

    if mask.ndim != 1:
        raise ValueError("mask must be a 1-D person-level mask")

    n_people = len(mask)
    n_group = int(mask.sum())
    n_rest = n_people - n_group

    if n_group < min_size or n_rest < min_size:
        return None

    node_index = np.asarray(
        observations.node_index,
        dtype=np.int64,
    )
    infected = np.asarray(
        observations.infected,
        dtype=np.bool_,
    )

    if node_index.ndim != 1:
        raise ValueError(
            "observations.node_index must be 1-D"
        )

    if node_index.size == 0:
        return None

    if (
        np.any(node_index < 0)
        or np.any(node_index >= n_people)
    ):
        raise ValueError(
            "observation node indices are incompatible with "
            "the person-level subgroup mask"
        )

    if exposure_type == "raw":
        escape_exposure = np.asarray(
            observations.raw_escape_exposure,
            dtype=np.float64,
        )
        infection_exposure = np.asarray(
            observations.raw_infection_exposure,
            dtype=np.float64,
        )

    elif exposure_type == "weighted":
        escape_exposure = np.asarray(
            observations.weighted_escape_exposure,
            dtype=np.float64,
        )
        infection_exposure = np.asarray(
            observations.weighted_infection_exposure,
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "exposure_type must be 'raw' or 'weighted'"
        )

    if not (
        infected.shape
        == node_index.shape
        == escape_exposure.shape
        == infection_exposure.shape
    ):
        raise ValueError(
            "Model A observation arrays must have matching shapes"
        )

    if not np.all(np.isfinite(escape_exposure)):
        raise ValueError(
            "escape exposure must contain only finite values"
        )

    if not np.all(np.isfinite(infection_exposure)):
        raise ValueError(
            "infection exposure must contain only finite values"
        )

    if np.any(escape_exposure < 0):
        raise ValueError("escape exposure cannot be negative")

    if np.any(infection_exposure < 0):
        raise ValueError("infection exposure cannot be negative")

    observation_mask = mask[node_index]

    group_hazard = fit_model_a_hazard(
        infected[observation_mask],
        escape_exposure[observation_mask],
        infection_exposure[observation_mask],
    )

    rest_hazard = fit_model_a_hazard(
        infected[~observation_mask],
        escape_exposure[~observation_mask],
        infection_exposure[~observation_mask],
    )

    # A finite log rate ratio requires positive finite estimates on both sides.
    if (
        group_hazard is None
        or rest_hazard is None
        or group_hazard <= 0
        or rest_hazard <= 0
    ):
        return None

    rate_ratio = group_hazard / rest_hazard
    log_rate_ratio = float(np.log(rate_ratio))

    effect = _directional_effect(
        log_rate_ratio,
        direction,
    )

    support = n_group / n_people
    quality = np.sqrt(support) * effect

    return {
        "size": n_group,
        "support": support,
        "observations": int(observation_mask.sum()),
        "rest_observations": int((~observation_mask).sum()),
        "infections": int(
            infected[observation_mask].sum()
        ),
        "rest_infections": int(
            infected[~observation_mask].sum()
        ),
        "exposure_type": exposure_type,
        "group_escape_exposure": float(
            escape_exposure[observation_mask].sum()
        ),
        "rest_escape_exposure": float(
            escape_exposure[~observation_mask].sum()
        ),
        "group_hazard": float(group_hazard),
        "rest_hazard": float(rest_hazard),
        "rate_ratio": float(rate_ratio),
        "log_rate_ratio": log_rate_ratio,
        "quality": float(quality),
    }


def search_model_a_subgroups(
    df: pd.DataFrame,
    observations: ModelAObservations,
    *,
    categorical_columns: Sequence[str],
    age_column: str = "age",
    min_size: int = 200,
    max_depth: int = 2,
    direction: str = "high",
    exposure_type: str = "weighted",
    n_bins: int = 10,
    descriptions: Sequence[Description] | None = None,
) -> pd.DataFrame:
    """
    Search for exceptional susceptibility subgroups using Model A.

    The final local equal-frequency description space is generated unless a
    precomputed description list is supplied.
    """

    if len(df) == 0:
        raise ValueError("df must contain at least one person")

    if exposure_type not in ("raw", "weighted"):
        raise ValueError(
            "exposure_type must be 'raw' or 'weighted'"
        )

    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'")

    if descriptions is None:
        descriptions = generate_descriptions(
            df,
            categorical_columns=categorical_columns,
            age_column=age_column,
            max_depth=max_depth,
            n_bins=n_bins,
        )

    return search_descriptions(
        descriptions,
        lambda mask: score_model_a_subgroup(
            mask,
            observations,
            min_size=min_size,
            direction=direction,
            exposure_type=exposure_type,
        ),
    )


# ============================================================================
# Model B: opportunity-adjusted infectiousness
# ============================================================================


def score_model_b_subgroup(
    mask: NDArray[np.bool_],
    observations: ModelBObservations,
    *,
    min_size: int = 200,
    direction: str = "high",
    opportunity_type: str = "weighted",
) -> dict | None:
    """
    Score a subgroup using Model B.

    Model B compares direct secondary infections with realised opportunities
    to transmit to susceptible neighbours.

    opportunity_type="raw":
        Susceptible-neighbour contact-steps.

    opportunity_type="weighted":
        Contact-steps weighted by recipient susceptibility and, when present
        in the observation construction, the contact-specific edge multiplier.

    Quality:
        sqrt(|G| / N) * oriented log(RR)
    """

    mask = np.asarray(mask, dtype=np.bool_)

    if mask.ndim != 1:
        raise ValueError("mask must be a 1-D person-level mask")

    n_people = len(mask)
    n_group = int(mask.sum())
    n_rest = n_people - n_group

    if n_group < min_size or n_rest < min_size:
        return None

    node_index = np.asarray(
        observations.node_index,
        dtype=np.int64,
    )

    if node_index.ndim != 1:
        raise ValueError(
            "observations.node_index must be 1-D"
        )

    if node_index.size == 0:
        return None

    if (
        np.any(node_index < 0)
        or np.any(node_index >= n_people)
    ):
        raise ValueError(
            "observation node indices are incompatible with "
            "the person-level subgroup mask"
        )

    offspring = np.asarray(
        observations.offspring,
        dtype=np.float64,
    )

    if opportunity_type == "raw":
        opportunity = np.asarray(
            observations.opportunity,
            dtype=np.float64,
        )

    elif opportunity_type == "weighted":
        opportunity = np.asarray(
            observations.weighted_opportunity,
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "opportunity_type must be 'raw' or 'weighted'"
        )

    if offspring.shape != node_index.shape:
        raise ValueError(
            "offspring and node_index must have the same shape"
        )

    if opportunity.shape != node_index.shape:
        raise ValueError(
            "opportunity and node_index must have the same shape"
        )

    if not np.all(np.isfinite(offspring)):
        raise ValueError(
            "offspring must contain only finite values"
        )

    if not np.all(np.isfinite(opportunity)):
        raise ValueError(
            "opportunity must contain only finite values"
        )

    if np.any(offspring < 0):
        raise ValueError("offspring cannot be negative")

    if np.any(opportunity < 0):
        raise ValueError("opportunity cannot be negative")

    observation_mask = mask[node_index]

    group_offspring = float(
        offspring[observation_mask].sum()
    )
    group_opportunity = float(
        opportunity[observation_mask].sum()
    )

    rest_offspring = float(
        offspring[~observation_mask].sum()
    )
    rest_opportunity = float(
        opportunity[~observation_mask].sum()
    )

    if group_opportunity <= 0 or rest_opportunity <= 0:
        return None

    group_rate = group_offspring / group_opportunity
    rest_rate = rest_offspring / rest_opportunity

    # log(RR) is undefined if either transmission rate is zero.
    if group_rate <= 0 or rest_rate <= 0:
        return None

    rate_ratio = group_rate / rest_rate
    log_rate_ratio = float(np.log(rate_ratio))

    effect = _directional_effect(
        log_rate_ratio,
        direction,
    )

    support = n_group / n_people
    quality = np.sqrt(support) * effect

    return {
        "size": n_group,
        "support": support,
        "observations": int(observation_mask.sum()),
        "rest_observations": int((~observation_mask).sum()),
        "opportunity_type": opportunity_type,
        "group_offspring": group_offspring,
        "group_opportunity": group_opportunity,
        "group_rate": group_rate,
        "rest_offspring": rest_offspring,
        "rest_opportunity": rest_opportunity,
        "rest_rate": rest_rate,
        "rate_ratio": float(rate_ratio),
        "log_rate_ratio": log_rate_ratio,
        "quality": float(quality),
    }


def search_model_b_subgroups(
    df: pd.DataFrame,
    observations: ModelBObservations,
    *,
    categorical_columns: Sequence[str],
    age_column: str = "age",
    min_size: int = 200,
    max_depth: int = 2,
    direction: str = "high",
    opportunity_type: str = "weighted",
    n_bins: int = 10,
    descriptions: Sequence[Description] | None = None,
) -> pd.DataFrame:
    """
    Search for exceptional infectiousness subgroups using Model B.

    The final local equal-frequency description space is generated unless a
    precomputed description list is supplied.
    """

    if len(df) == 0:
        raise ValueError("df must contain at least one person")

    if opportunity_type not in ("raw", "weighted"):
        raise ValueError(
            "opportunity_type must be 'raw' or 'weighted'"
        )

    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'")

    if descriptions is None:
        descriptions = generate_descriptions(
            df,
            categorical_columns=categorical_columns,
            age_column=age_column,
            max_depth=max_depth,
            n_bins=n_bins,
        )

    return search_descriptions(
        descriptions,
        lambda mask: score_model_b_subgroup(
            mask,
            observations,
            min_size=min_size,
            direction=direction,
            opportunity_type=opportunity_type,
        ),
    )
