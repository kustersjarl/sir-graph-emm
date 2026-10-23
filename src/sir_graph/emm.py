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
# Description generation with global or local discretisation
# ============================================================


@dataclass(frozen=True)
class Description:
    """
    One evaluable subgroup description of depth 1 or 2.
    """

    description: str
    depth: int
    mask: NDArray[np.bool_]


def equal_frequency_cut_points(
    values: NDArray,
    n_bins: int,
) -> list[float]:
    """
    Cut points of ``n_bins`` equal-frequency bins.

    Cut points are rounded up to whole values (ages are integers),
    deduplicated, and kept only if they actually split ``values``.
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or n_bins < 2:
        return []

    quantiles = np.quantile(
        values,
        np.arange(1, n_bins) / n_bins,
    )

    cut_points = np.unique(np.ceil(quantiles))

    return [
        float(c)
        for c in cut_points
        if np.any(values < c) and np.any(values >= c)
    ]


def _age_conditions(
    ages: NDArray,
    age_column: str,
    cut_points: list[float],
) -> list[Candidate]:
    """
    Two half-bounded conditions per cut point: age >= c and age < c.
    """

    conditions: list[Candidate] = []

    for c in cut_points:
        label = f"{c:g}"

        conditions.append(
            Candidate(
                description=f"{age_column} >= {label}",
                attribute=age_column,
                mask=ages >= c,
            )
        )

        conditions.append(
            Candidate(
                description=f"{age_column} < {label}",
                attribute=age_column,
                mask=ages < c,
            )
        )

    return conditions


def generate_descriptions(
    df: pd.DataFrame,
    *,
    categorical_columns: list[str],
    age_column: str = "age",
    max_depth: int = 2,
    discretisation: str = "local",
    n_bins: int = 10,
    age_thresholds=range(20, 81, 5),
) -> list[Description]:
    """
    Enumerate all subgroup descriptions up to depth 2.

    discretisation="global":
        Age cut points are the fixed ``age_thresholds``, computed once
        for the whole population. Equivalent to the original search.

    discretisation="local":
        Age cut points are ``n_bins`` equal-frequency bins computed on
        the subgroup that is being refined. At depth 1 this subgroup is
        the whole population. When age refines a categorical condition
        (e.g. smoking == 1), its cut points are recomputed on the ages
        inside that condition.

    In both modes, two conditions on the same attribute are never
    combined, and descriptions selecting exactly the same people are
    kept only once (first occurrence wins).
    """

    if max_depth not in (1, 2):
        raise ValueError("current implementation supports max_depth 1 or 2")

    if discretisation not in ("global", "local"):
        raise ValueError("discretisation must be 'global' or 'local'")

    if age_column not in df.columns:
        raise KeyError(f"{age_column!r} not found in dataframe")

    ages = df[age_column].to_numpy(dtype=np.float64)

    categorical = generate_candidates(
        df,
        categorical_columns=categorical_columns,
        age_column=age_column,
        age_thresholds=[],
    )

    if discretisation == "global":
        root_age = _age_conditions(
            ages,
            age_column,
            [float(c) for c in age_thresholds],
        )
    else:
        root_age = _age_conditions(
            ages,
            age_column,
            equal_frequency_cut_points(ages, n_bins),
        )

    level_1 = categorical + root_age

    descriptions: list[Description] = []
    seen: set[bytes] = set()

    def add(description: str, depth: int, mask: NDArray[np.bool_]) -> None:
        key = _mask_key(mask)
        if key in seen:
            return
        seen.add(key)
        descriptions.append(
            Description(description=description, depth=depth, mask=mask)
        )

    for condition in level_1:
        add(condition.description, 1, condition.mask)

    if max_depth == 1:
        return descriptions

    def pair(a: Candidate, b: Candidate) -> None:
        add(f"{a.description} AND {b.description}", 2, a.mask & b.mask)

    if discretisation == "global":
        for a, b in combinations(level_1, 2):
            if a.attribute != b.attribute:
                pair(a, b)
        return descriptions

    # Local: categorical x categorical
    for a, b in combinations(categorical, 2):
        if a.attribute != b.attribute:
            pair(a, b)

    # Local: a categorical subgroup refined by age, with cut points
    # computed on the ages inside that subgroup.
    for a in categorical:
        local_cut_points = equal_frequency_cut_points(ages[a.mask], n_bins)
        for b in _age_conditions(ages, age_column, local_cut_points):
            pair(a, b)

    # Local: a root-level age condition refined by a categorical one
    # (no discretisation needed for the refinement).
    for a in root_age:
        for b in categorical:
            pair(a, b)

    return descriptions


def search_descriptions(
    descriptions: list[Description],
    scorer,
) -> pd.DataFrame:
    """
    Score every description with ``scorer(mask) -> dict | None`` and
    return the results sorted by quality (highest first).
    """

    results = []

    for d in descriptions:
        score = scorer(d.mask)
        if score is None:
            continue
        results.append(
            {
                "depth": d.depth,
                "description": d.description,
                **score,
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("quality", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


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
# MODEL A
# Exposure-adjusted susceptibility
# ============================================================


def score_model_a_subgroup(
    mask: NDArray[np.bool_],
    observations: ModelAObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    exposure_type: str = "weighted",
) -> dict | None:
    """
    Score a person-level subgroup using Model A.

    Model A asks whether members of the subgroup
    become infected more easily than the rest,
    after accounting for infectious exposure.

    exposure_type="raw":
        Exposure is the number of infectious
        neighbours.

    exposure_type="weighted":
        Exposure is weighted by source
        infectiousness and edge multiplier.

    For subgroup G:

        lambda_G =
            fitted susceptibility hazard

        lambda_rest =
            fitted susceptibility hazard
            outside G

        rate_ratio =
            lambda_G / lambda_rest

        model effect =
            log(rate_ratio)

    Quality:

        sqrt(|G| / N) * effect

    Subgroup support is based on unique people.
    """

    mask = np.asarray(
        mask,
        dtype=np.bool_,
    )

    if mask.ndim != 1:
        raise ValueError(
            "mask must be a 1-D "
            "person-level mask"
        )

    n_people = len(mask)
    n_group = int(
        mask.sum()
    )

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

    infected = np.asarray(
        observations.infected,
        dtype=np.bool_,
    )

    if node_index.ndim != 1:
        raise ValueError(
            "observations.node_index "
            "must be 1-D"
        )

    if len(
        node_index
    ) == 0:
        return None

    if (
        np.any(
            node_index < 0
        )
        or np.any(
            node_index >= n_people
        )
    ):
        raise ValueError(
            "observation node indices "
            "are incompatible with the "
            "person-level subgroup mask"
        )

    # --------------------------------------------------------
    # Choose exposure definition
    # --------------------------------------------------------

    if exposure_type == "raw":

        escape_exposure = np.asarray(
            observations
            .raw_escape_exposure,
            dtype=np.float64,
        )

        infection_exposure = np.asarray(
            observations
            .raw_infection_exposure,
            dtype=np.float64,
        )

    elif exposure_type == "weighted":

        escape_exposure = np.asarray(
            observations
            .weighted_escape_exposure,
            dtype=np.float64,
        )

        infection_exposure = np.asarray(
            observations
            .weighted_infection_exposure,
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "exposure_type must be "
            "'raw' or 'weighted'"
        )

    # --------------------------------------------------------
    # Validate observation arrays
    # --------------------------------------------------------

    if not (
        infected.shape
        == node_index.shape
        == escape_exposure.shape
        == infection_exposure.shape
    ):
        raise ValueError(
            "Model A observation arrays "
            "must have matching shapes"
        )

    if np.any(
        ~np.isfinite(
            escape_exposure
        )
    ):
        raise ValueError(
            "escape exposure must contain "
            "only finite values"
        )

    if np.any(
        ~np.isfinite(
            infection_exposure
        )
    ):
        raise ValueError(
            "infection exposure must contain "
            "only finite values"
        )

    if np.any(
        escape_exposure < 0
    ):
        raise ValueError(
            "escape exposure cannot be negative"
        )

    if np.any(
        infection_exposure < 0
    ):
        raise ValueError(
            "infection exposure cannot be negative"
        )

    # --------------------------------------------------------
    # Convert person-level subgroup membership
    # to run-person membership
    # --------------------------------------------------------

    observation_mask = mask[
        node_index
    ]

    group_observations = int(
        observation_mask.sum()
    )

    rest_observations = int(
        (
            ~observation_mask
        ).sum()
    )

    group_infections = int(
        infected[
            observation_mask
        ].sum()
    )

    rest_infections = int(
        infected[
            ~observation_mask
        ].sum()
    )

    # --------------------------------------------------------
    # Fit susceptibility hazards
    # --------------------------------------------------------

    group_hazard = (
        fit_model_a_hazard(
            infected[
                observation_mask
            ],
            escape_exposure[
                observation_mask
            ],
            infection_exposure[
                observation_mask
            ],
        )
    )

    rest_hazard = (
        fit_model_a_hazard(
            infected[
                ~observation_mask
            ],
            escape_exposure[
                ~observation_mask
            ],
            infection_exposure[
                ~observation_mask
            ],
        )
    )

    # Cannot compute a finite log hazard ratio
    # if either side has no finite positive MLE.
    if (
        group_hazard is None
        or rest_hazard is None
        or group_hazard <= 0
        or rest_hazard <= 0
    ):
        return None

    # --------------------------------------------------------
    # Model A effect
    # --------------------------------------------------------

    rate_ratio = (
        group_hazard
        / rest_hazard
    )

    log_rate_ratio = float(
        np.log(
            rate_ratio
        )
    )

    if direction == "high":

        effect = (
            log_rate_ratio
        )

    elif direction == "low":

        effect = (
            -log_rate_ratio
        )

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
        np.sqrt(
            support
        )
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

        "infections":
            group_infections,

        "rest_infections":
            rest_infections,

        "exposure_type":
            exposure_type,

        "group_escape_exposure":
            float(
                escape_exposure[
                    observation_mask
                ].sum()
            ),

        "rest_escape_exposure":
            float(
                escape_exposure[
                    ~observation_mask
                ].sum()
            ),

        "group_hazard":
            group_hazard,

        "rest_hazard":
            rest_hazard,

        "rate_ratio":
            rate_ratio,

        "log_rate_ratio":
            log_rate_ratio,

        "quality":
            quality,
    }


def search_model_a_single_subgroups(
    candidates: list[Candidate],
    observations: ModelAObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    exposure_type: str = "weighted",
) -> pd.DataFrame:
    """
    Evaluate all depth-1 subgroup descriptions
    using Model A.
    """

    results = []

    for candidate in candidates:

        score = score_model_a_subgroup(
            candidate.mask,
            observations,
            min_size=min_size,
            direction=direction,
            exposure_type=
                exposure_type,
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
        pd.DataFrame(
            results
        )
        .sort_values(
            "quality",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def search_model_a_pair_subgroups(
    candidates: list[Candidate],
    observations: ModelAObservations,
    *,
    min_size: int = 20,
    direction: str = "high",
    exposure_type: str = "weighted",
) -> pd.DataFrame:
    """
    Evaluate depth-2 conjunctions using Model A.

    Equivalent member sets are evaluated only once.
    """

    results = []

    seen_masks = {
        _mask_key(
            candidate.mask
        )
        for candidate
        in candidates
    }

    for a, b in combinations(
        candidates,
        2,
    ):

        if (
            a.attribute
            == b.attribute
        ):
            continue

        mask = (
            a.mask
            & b.mask
        )

        mask_key = (
            _mask_key(
                mask
            )
        )

        if (
            mask_key
            in seen_masks
        ):
            continue

        seen_masks.add(
            mask_key
        )

        score = score_model_a_subgroup(
            mask,
            observations,
            min_size=min_size,
            direction=direction,
            exposure_type=
                exposure_type,
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
        pd.DataFrame(
            results
        )
        .sort_values(
            "quality",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def search_model_a_subgroups(
    df: pd.DataFrame,
    observations: ModelAObservations,
    *,
    categorical_columns: list[str],
    age_column: str = "age",
    age_thresholds=range(
        20,
        81,
        5,
    ),
    min_size: int = 20,
    max_depth: int = 2,
    direction: str = "high",
    exposure_type: str = "weighted",
    discretisation: str = "global",
    n_bins: int = 10,
    descriptions: list[Description] | None = None,
) -> pd.DataFrame:
    """
    Run EMM using Model A.

    Model A searches for subgroups with
    exceptional susceptibility after adjusting
    for actual infectious exposure.

    exposure_type="raw":
        Count infectious neighbours.

    exposure_type="weighted":
        Weight infectious neighbours by
        source infectiousness and edge multiplier.

    discretisation="global" (default) keeps the original search.
    discretisation="local" recomputes equal-frequency age cut points
    within each refined subgroup (see generate_descriptions).
    Precomputed ``descriptions`` may be passed to skip generation.

    Supports depths 1 and 2.
    """

    if descriptions is not None or discretisation == "local":
        if descriptions is None:
            descriptions = generate_descriptions(
                df,
                categorical_columns=categorical_columns,
                age_column=age_column,
                max_depth=max_depth,
                discretisation=discretisation,
                n_bins=n_bins,
                age_thresholds=age_thresholds,
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

    if len(
        df
    ) == 0:
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

    if exposure_type not in (
        "raw",
        "weighted",
    ):
        raise ValueError(
            "exposure_type must be "
            "'raw' or 'weighted'"
        )

    candidates = generate_candidates(
        df,
        categorical_columns=
            categorical_columns,
        age_column=
            age_column,
        age_thresholds=
            age_thresholds,
    )

    single_results = (
        search_model_a_single_subgroups(
            candidates,
            observations,
            min_size=min_size,
            direction=direction,
            exposure_type=
                exposure_type,
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
        search_model_a_pair_subgroups(
            candidates,
            observations,
            min_size=min_size,
            direction=direction,
            exposure_type=
                exposure_type,
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
        .reset_index(
            drop=True
        )
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
    discretisation: str = "global",
    n_bins: int = 10,
    descriptions: list[Description] | None = None,
) -> pd.DataFrame:
    """
    Run EMM using Model B.

    discretisation="global" (default) keeps the original search.
    discretisation="local" recomputes equal-frequency age cut points
    within each refined subgroup (see generate_descriptions).
    Precomputed ``descriptions`` may be passed to skip generation.

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

    if descriptions is not None or discretisation == "local":
        if descriptions is None:
            descriptions = generate_descriptions(
                df,
                categorical_columns=categorical_columns,
                age_column=age_column,
                max_depth=max_depth,
                discretisation=discretisation,
                n_bins=n_bins,
                age_thresholds=age_thresholds,
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