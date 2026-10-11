from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Candidate:
    """
    One possible subgroup condition.

    Example:
        smoking == 1
        age >= 60
    """

    description: str
    attribute: str
    mask: NDArray[np.bool_]


def score_subgroup(
    mask: NDArray[np.bool_],
    target: NDArray[np.float64],
    *,
    min_size: int = 20,
    direction: str = "high",
) -> dict | None:
    """
    Score one subgroup.

    Quality:
        sqrt(|G| / N) * effect

    For direction="high":
        effect = mean(G) - mean(not G)

    For direction="low":
        effect = mean(not G) - mean(G)

    For direction="both":
        effect = abs(mean(G) - mean(not G))
    """
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=np.float64)

    if mask.shape != target.shape:
        raise ValueError(
            f"mask and target must have same shape, "
            f"got {mask.shape} and {target.shape}"
        )

    # Ignore rows where the target is NaN/inf
    valid = np.isfinite(target)

    mask = mask[valid]
    target = target[valid]

    n = len(target)
    n_group = int(mask.sum())

    if n_group < min_size:
        return None

    if n - n_group < min_size:
        return None

    mean_group = float(target[mask].mean())
    mean_rest = float(target[~mask].mean())

    raw_difference = mean_group - mean_rest

    if direction == "high":
        effect = raw_difference
    elif direction == "low":
        effect = -raw_difference
    elif direction == "both":
        effect = abs(raw_difference)
    else:
        raise ValueError(
            "direction must be 'high', 'low', or 'both'"
        )

    support = n_group / n
    quality = np.sqrt(support) * effect

    return {
        "size": n_group,
        "support": support,
        "mean_residual": mean_group,
        "rest_mean_residual": mean_rest,
        "difference": raw_difference,
        "quality": quality,
    }


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
            raise KeyError(f"{column!r} not found in dataframe")

        for value in df[column].dropna().unique():
            candidates.append(
                Candidate(
                    description=f"{column} == {value}",
                    attribute=column,
                    mask=(df[column] == value).to_numpy(),
                )
            )

    if age_column not in df.columns:
        raise KeyError(f"{age_column!r} not found in dataframe")

    for threshold in age_thresholds:
        candidates.append(
            Candidate(
                description=f"{age_column} >= {threshold}",
                attribute=age_column,
                mask=(df[age_column] >= threshold).to_numpy(),
            )
        )

        candidates.append(
            Candidate(
                description=f"{age_column} < {threshold}",
                attribute=age_column,
                mask=(df[age_column] < threshold).to_numpy(),
            )
        )

    return candidates


def search_single_subgroups(
    candidates: list[Candidate],
    target: NDArray[np.float64],
    *,
    min_size: int = 20,
    direction: str = "high",
) -> pd.DataFrame:
    """
    Evaluate all single-condition subgroups.
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
                "description": candidate.description,
                **score,
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("quality", ascending=False)
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
    Evaluate all two-condition conjunctions.

    Conditions using the same attribute are currently not combined.
    For example:
        age >= 50 AND age < 70

    is intentionally excluded from this first implementation.
    """
    results = []

    for a, b in combinations(candidates, 2):

        # Do not combine two conditions on the same attribute yet
        if a.attribute == b.attribute:
            continue

        mask = a.mask & b.mask

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
                    f"{a.description} AND {b.description}"
                ),
                **score,
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("quality", ascending=False)
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
    Convenience function for running the complete EMM search.

    Currently supports subgroup descriptions of depth 1 or 2.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"{target_column!r} not found in dataframe"
        )

    if max_depth not in (1, 2):
        raise ValueError(
            "current implementation supports max_depth 1 or 2"
        )

    candidates = generate_candidates(
        df,
        categorical_columns=categorical_columns,
        age_column=age_column,
        age_thresholds=age_thresholds,
    )

    target = df[target_column].to_numpy(
        dtype=np.float64
    )

    single_results = search_single_subgroups(
        candidates,
        target,
        min_size=min_size,
        direction=direction,
    )

    single_results.insert(
        0,
        "depth",
        1,
    )

    if max_depth == 1:
        return single_results

    pair_results = search_pair_subgroups(
        candidates,
        target,
        min_size=min_size,
        direction=direction,
    )

    pair_results.insert(
        0,
        "depth",
        2,
    )

    return (
        pd.concat(
            [single_results, pair_results],
            ignore_index=True,
        )
        .sort_values("quality", ascending=False)
        .reset_index(drop=True)
    )