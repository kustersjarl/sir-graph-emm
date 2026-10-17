from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sir_graph.analysis import ModelBObservations


@dataclass(frozen=True)
class Candidate:
    """
    One possible subgroup condition.

    Examples:
        smoking == 1
        age >= 60
    """

    description: str
    attribute: str
    mask: NDArray[np.bool_]


def _mask_key(
    mask: NDArray[np.bool_],
) -> bytes:
    """
    Return a stable identity for a subgroup's actual member set.

    Syntactically different subgroup descriptions that select exactly
    the same people therefore receive the same key.
    """

    return np.asarray(
        mask,
        dtype=np.bool_,
    ).tobytes()


# ============================================================
# Candidate generation
# ============================================================


def generate_candidates(
    df: pd.DataFrame,
    *,
    categorical_columns: list[str],
    age_column: str = "age",
    age_thresholds=range(20, 81, 5),
) -> list[Candidate]:
    """
    Generate simple one-condition subgroup descriptions.

    Categorical examples:
        sex == Female
        smoking == 1

    Numeric age examples:
        age >= 60
        age < 60
    """

    candidates: list[Candidate] = []

    for column in categorical_columns:
        if column not in df.columns:
            raise KeyError(
                f"{column!r} not found in dataframe"
            )

        for value in df[column].dropna().unique():
            candidates.append(
                Candidate(
                    description=(
                        f"{column} == {value}"
                    ),
                    attribute=column,
                    mask=(
                        df[column] == value
                    ).to_numpy(),
                )
            )

    if age_column not in df.columns:
        raise KeyError(
            f"{age_column!r} not found in dataframe"
        )

    for threshold in age_thresholds:
        candidates.append(
            Candidate(
                description=(
                    f"{age_column} >= {threshold}"
                ),
                attribute=age_column,
                mask=(
                    df[age_column] >= threshold
                ).to_numpy(),
            )
        )

        candidates.append(
            Candidate(
                description=(
                    f"{age_column} < {threshold}"
                ),
                attribute=age_column,
                mask=(
                    df[age_column] < threshold
                ).to_numpy(),
            )
        )

    return candidates


# ============================================================
# BASELINE EMM
# Residual-based method used in the previous experiments
# ============================================================


def score_subgroup(
    mask: NDArray[np.bool_],
    target: NDArray[np.float64],
    *,
    min_size: int = 20,
    direction: str = "high",
) -> dict | None:
    """
    Score one subgroup using a numeric target.

    Quality:
        sqrt(|G| / N) * effect

    For direction="high":
        effect = mean(G) - mean(not G)

    For direction="low":
        effect = mean(not G) - mean(G)

    For direction="both":
        effect = abs(mean(G) - mean(not G))

    This is the original residual-based EMM
    quality measure.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if mask.shape != target.shape:
        raise ValueError(
            f"mask and target must have same shape, "
            f"got {mask.shape} and {target.shape}"
        )

    valid = np.isfinite(target)

    mask = mask[valid]
    target = target[valid]

    n = len(target)
    n_group = int(mask.sum())

    if n_group < min_size:
        return None

    if n - n_group < min_size:
        return None

    mean_group = float(
        target[mask].mean()
    )

    mean_rest = float(
        target[~mask].mean()
    )

    raw_difference = (
        mean_group
        - mean_rest
    )

    if direction == "high":
        effect = raw_difference

    elif direction == "low":
        effect = -raw_difference

    elif direction == "both":
        effect = abs(
            raw_difference
        )

    else:
        raise ValueError(
            "direction must be "
            "'high', 'low', or 'both'"
        )

    support = (
        n_group
        / n
    )

    quality = (
        np.sqrt(support)
        * effect
    )

    return {
        "size": n_group,
        "support": support,
        "mean_residual": mean_group,
        "rest_mean_residual": mean_rest,
        "difference": raw_difference,
        "quality": quality,
    }


def search_single_subgroups(
    candidates: list[Candidate],
    target: NDArray[np.float64],
    *,
    min_size: int = 20,
    direction: str = "high",
) -> pd.DataFrame:
    """
    Evaluate all depth-1 subgroup descriptions
    using the original numeric target.
    """

    results = []

    for candidate in candidates:
        score = score_subgroup(
            candidate.mask,
            target,
            min_size=min_size,
            direction=direction,
        )

        if score is None:
            continue

        results.append(
            {
                "description":
                    candidate.description,
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
        )
        .reset_index(drop=True)
    )


def search_pair_subgroups(
    candidates: list[Candidate],
    target: NDArray[np.float64],
    *,
    min_size: int = 20,
    direction: str = "high",
) -> pd.DataFrame:
    """
    Evaluate all depth-2 conjunctions
    using the original numeric target.

    Conditions using the same attribute
    are not combined.
    """

    results = []

    # A depth-2 subgroup is redundant if it selects exactly the same
    # people as an existing depth-1 candidate. Also keep only the first
    # depth-2 description for any repeated member set.
    seen_masks = {
        _mask_key(candidate.mask)
        for candidate in candidates
    }

    for a, b in combinations(
        candidates,
        2,
    ):
        if a.attribute == b.attribute:
            continue

        mask = (
            a.mask
            & b.mask
        )

        mask_key = _mask_key(mask)

        if mask_key in seen_masks:
            continue

        seen_masks.add(mask_key)

        score = score_subgroup(
            mask,
            target,
            min_size=min_size,
            direction=direction,
        )

        if score is None:
            continue

        results.append(
            {
                "description": (
                    f"{a.description} "
                    f"AND {b.description}"
                ),
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
        )
        .reset_index(drop=True)
    )


def search_subgroups(
    df: pd.DataFrame,
    *,
    target_column: str,
    categorical_columns: list[str],
    age_column: str = "age",
    age_thresholds=range(20, 81, 5),
    min_size: int = 20,
    max_depth: int = 2,
    direction: str = "high",
) -> pd.DataFrame:
    """
    Run the original residual-based EMM search.

    Supports subgroup descriptions
    of depth 1 or 2.
    """

    if target_column not in df.columns:
        raise KeyError(
            f"{target_column!r} "
            "not found in dataframe"
        )

    if max_depth not in (1, 2):
        raise ValueError(
            "current implementation supports "
            "max_depth 1 or 2"
        )

    candidates = generate_candidates(
        df,
        categorical_columns=
            categorical_columns,
        age_column=age_column,
        age_thresholds=age_thresholds,
    )

    target = df[
        target_column
    ].to_numpy(
        dtype=np.float64
    )

    single_results = (
        search_single_subgroups(
            candidates,
            target,
            min_size=min_size,
            direction=direction,
        )
    )

    if not single_results.empty:
        single_results.insert(
            0,
            "depth",
            1,
        )

    if max_depth == 1:
        return single_results

    pair_results = (
        search_pair_subgroups(
            candidates,
            target,
            min_size=min_size,
            direction=direction,
        )
    )

    if not pair_results.empty:
        pair_results.insert(
            0,
            "depth",
            2,
        )

    result_frames = [
        frame
        for frame in (
            single_results,
            pair_results,
        )
        if not frame.empty
    ]

    if not result_frames:
        return pd.DataFrame()

    return (
        pd.concat(
            result_frames,
            ignore_index=True,
        )
        .sort_values(
            "quality",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# MODEL B
# Opportunity-adjusted direct offspring
# ============================================================


def score_model_b_subgroup(
    mask: NDArray[np.bool_],
    observations: ModelBObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    opportunity_type: str = "raw",
) -> dict | None:
    """
    Score a person-level subgroup using Model B.

    Model B measures direct transmission relative
    to actual transmission opportunity.

    Opportunity types
    -----------------
    raw:
        Susceptible-neighbour contact-steps.

    weighted:
        Susceptible-neighbour contact-steps weighted
        by the susceptibility of the receiving node.

    For subgroup G:

        r_G =
            total offspring in G
            --------------------
            total opportunity in G

        r_rest =
            total offspring outside G
            -------------------------
            total opportunity outside G

        rate_ratio =
            r_G / r_rest

        model effect =
            log(rate_ratio)

    Quality:

        sqrt(|G| / N) * effect

    The subgroup support is based on unique
    people, not run-person observations.
    """

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.ndim != 1:
        raise ValueError(
            "mask must be a 1-D "
            "person-level mask"
        )

    n_people = len(mask)
    n_group = int(mask.sum())

    if n_group < min_size:
        return None

    if (
        n_people - n_group
        < min_size
    ):
        return None

    node_index = np.asarray(
        observations.node_index,
        dtype=np.int64,
    )

    if node_index.ndim != 1:
        raise ValueError(
            "observations.node_index "
            "must be 1-D"
        )

    if len(node_index) == 0:
        return None

    if (
        np.any(node_index < 0)
        or np.any(
            node_index >= n_people
        )
    ):
        raise ValueError(
            "observation node indices "
            "are incompatible with the "
            "person-level subgroup mask"
        )

    offspring = np.asarray(
        observations.offspring,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Choose opportunity definition
    # --------------------------------------------------------

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
            "opportunity_type must be "
            "'raw' or 'weighted'"
        )

    # --------------------------------------------------------
    # Validate arrays
    # --------------------------------------------------------

    if (
        offspring.shape
        != node_index.shape
    ):
        raise ValueError(
            "offspring and node_index "
            "must have the same shape"
        )

    if (
        opportunity.shape
        != node_index.shape
    ):
        raise ValueError(
            "opportunity and node_index "
            "must have the same shape"
        )

    if np.any(
        ~np.isfinite(offspring)
    ):
        raise ValueError(
            "offspring must contain "
            "only finite values"
        )

    if np.any(
        ~np.isfinite(opportunity)
    ):
        raise ValueError(
            "opportunity must contain "
            "only finite values"
        )

    if np.any(
        offspring < 0
    ):
        raise ValueError(
            "offspring cannot be negative"
        )

    if np.any(
        opportunity < 0
    ):
        raise ValueError(
            "opportunity cannot be negative"
        )

    # --------------------------------------------------------
    # Map person-level subgroup membership
    # to run-person observations
    # --------------------------------------------------------

    observation_mask = mask[
        node_index
    ]

    group_observations = int(
        observation_mask.sum()
    )

    rest_observations = int(
        (~observation_mask).sum()
    )

    # --------------------------------------------------------
    # Aggregate offspring and opportunity
    # --------------------------------------------------------

    group_offspring = float(
        offspring[
            observation_mask
        ].sum()
    )

    group_opportunity = float(
        opportunity[
            observation_mask
        ].sum()
    )

    rest_offspring = float(
        offspring[
            ~observation_mask
        ].sum()
    )

    rest_opportunity = float(
        opportunity[
            ~observation_mask
        ].sum()
    )

    # A rate cannot be estimated
    # without positive opportunity.
    if (
        group_opportunity <= 0
        or rest_opportunity <= 0
    ):
        return None

    group_rate = (
        group_offspring
        / group_opportunity
    )

    rest_rate = (
        rest_offspring
        / rest_opportunity
    )

    # log(RR) is undefined if either
    # transmission rate is zero.
    if (
        group_rate <= 0
        or rest_rate <= 0
    ):
        return None

    # --------------------------------------------------------
    # Model B effect
    # --------------------------------------------------------

    rate_ratio = (
        group_rate
        / rest_rate
    )

    log_rate_ratio = float(
        np.log(rate_ratio)
    )

    if direction == "high":

        effect = log_rate_ratio

    elif direction == "low":

        effect = -log_rate_ratio

    elif direction == "both":

        effect = abs(
            log_rate_ratio
        )

    else:
        raise ValueError(
            "direction must be "
            "'high', 'low', or 'both'"
        )

    # --------------------------------------------------------
    # EMM quality
    # --------------------------------------------------------

    support = (
        n_group
        / n_people
    )

    quality = (
        np.sqrt(support)
        * effect
    )

    return {
        "size":
            n_group,

        "support":
            support,

        "observations":
            group_observations,

        "rest_observations":
            rest_observations,

        "opportunity_type":
            opportunity_type,

        "group_offspring":
            group_offspring,

        "group_opportunity":
            group_opportunity,

        "group_rate":
            group_rate,

        "rest_offspring":
            rest_offspring,

        "rest_opportunity":
            rest_opportunity,

        "rest_rate":
            rest_rate,

        "rate_ratio":
            rate_ratio,

        "log_rate_ratio":
            log_rate_ratio,

        "quality":
            quality,
    }


def search_model_b_single_subgroups(
    candidates: list[Candidate],
    observations: ModelBObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    opportunity_type: str = "raw",
) -> pd.DataFrame:
    """
    Evaluate all depth-1 subgroup descriptions
    using Model B.
    """

    results = []

    for candidate in candidates:

        score = score_model_b_subgroup(
            candidate.mask,
            observations,
            min_size=min_size,
            direction=direction,
            opportunity_type=
                opportunity_type,
        )

        if score is None:
            continue

        results.append(
            {
                "description":
                    candidate.description,
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
        )
        .reset_index(drop=True)
    )


def search_model_b_pair_subgroups(
    candidates: list[Candidate],
    observations: ModelBObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    opportunity_type: str = "raw",
) -> pd.DataFrame:
    """
    Evaluate all depth-2 conjunctions
    using Model B.
    """

    results = []

    # Use the same membership-based deduplication as the baseline EMM.
    # This prevents equivalent descriptions from occupying several ranks.
    seen_masks = {
        _mask_key(candidate.mask)
        for candidate in candidates
    }

    for a, b in combinations(
        candidates,
        2,
    ):

        # Do not combine two conditions
        # using the same attribute.
        if a.attribute == b.attribute:
            continue

        mask = (
            a.mask
            & b.mask
        )

        mask_key = _mask_key(mask)

        if mask_key in seen_masks:
            continue

        seen_masks.add(mask_key)

        score = score_model_b_subgroup(
            mask,
            observations,
            min_size=min_size,
            direction=direction,
            opportunity_type=
                opportunity_type,
        )

        if score is None:
            continue

        results.append(
            {
                "description": (
                    f"{a.description} "
                    f"AND {b.description}"
                ),
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
        )
        .reset_index(drop=True)
    )


def search_model_b_subgroups(
    df: pd.DataFrame,
    observations: ModelBObservations,
    *,
    categorical_columns: list[str],
    age_column: str = "age",
    age_thresholds=range(20, 81, 5),
    min_size: int = 20,
    max_depth: int = 2,
    direction: str = "high",
    opportunity_type: str = "raw",
) -> pd.DataFrame:
    """
    Run EMM using Model B.

    Subgroup descriptions are defined over
    the person-level dataframe.

    opportunity_type="raw":
        Use ordinary susceptible-neighbour
        contact-steps.

    opportunity_type="weighted":
        Weight susceptible contact-steps by
        the susceptibility of the receiving
        node.

    Exceptionalness is measured by the
    opportunity-adjusted transmission
    log-rate ratio.

    Supports depth 1 and depth 2.
    """

    if len(df) == 0:
        raise ValueError(
            "df must contain at least "
            "one person"
        )

    if max_depth not in (
        1,
        2,
    ):
        raise ValueError(
            "current implementation supports "
            "max_depth 1 or 2"
        )

    if opportunity_type not in (
        "raw",
        "weighted",
    ):
        raise ValueError(
            "opportunity_type must be "
            "'raw' or 'weighted'"
        )

    candidates = generate_candidates(
        df,
        categorical_columns=
            categorical_columns,
        age_column=age_column,
        age_thresholds=age_thresholds,
    )

    # --------------------------------------------------------
    # Depth 1
    # --------------------------------------------------------

    single_results = (
        search_model_b_single_subgroups(
            candidates,
            observations,
            min_size=min_size,
            direction=direction,
            opportunity_type=
                opportunity_type,
        )
    )

    if not single_results.empty:
        single_results.insert(
            0,
            "depth",
            1,
        )

    if max_depth == 1:
        return single_results

    # --------------------------------------------------------
    # Depth 2
    # --------------------------------------------------------

    pair_results = (
        search_model_b_pair_subgroups(
            candidates,
            observations,
            min_size=min_size,
            direction=direction,
            opportunity_type=
                opportunity_type,
        )
    )

    if not pair_results.empty:
        pair_results.insert(
            0,
            "depth",
            2,
        )

    result_frames = [
        frame
        for frame in (
            single_results,
            pair_results,
        )
        if not frame.empty
    ]

    if not result_frames:
        return pd.DataFrame()

    return (
        pd.concat(
            result_frames,
            ignore_index=True,
        )
        .sort_values(
            "quality",
            ascending=False,
        )
        .reset_index(drop=True)
    )