import numpy as np
import pandas as pd

from sir_graph.emm import (
    generate_candidates,
    score_subgroup,
    search_pair_subgroups,
    search_single_subgroups,
    search_subgroups,
)


def make_test_data() -> pd.DataFrame:
    """
    Synthetic data with a deliberately exceptional subgroup:
    age >= 60 AND smoking == 1
    """

    n = 100

    df = pd.DataFrame(
        {
            "age": np.concatenate(
                [
                    np.full(25, 65),
                    np.full(25, 65),
                    np.full(25, 35),
                    np.full(25, 35),
                ]
            ),
            "smoking": np.concatenate(
                [
                    np.ones(25),
                    np.zeros(25),
                    np.ones(25),
                    np.zeros(25),
                ]
            ).astype(int),
            "sex": ["Female", "Male"] * 50,
            "occupation": ["Office-based"] * n,
            "alcohol": np.zeros(n, dtype=int),
            "obesity": np.zeros(n, dtype=int),
        }
    )

    # Only age >= 60 AND smoker gets large residual
    df["emm_residual"] = 0.0

    planted = (
        (df["age"] >= 60)
        & (df["smoking"] == 1)
    )

    df.loc[
        planted,
        "emm_residual"
    ] = 1.0

    return df


def test_score_subgroup_positive_exception() -> None:
    target = np.array(
        [1.0, 1.0, 0.0, 0.0]
    )

    mask = np.array(
        [True, True, False, False]
    )

    result = score_subgroup(
        mask,
        target,
        min_size=1,
    )

    assert result is not None
    assert result["mean_residual"] == 1.0
    assert result["rest_mean_residual"] == 0.0
    assert result["quality"] > 0


def test_score_subgroup_rejects_small_group() -> None:
    target = np.arange(10, dtype=float)

    mask = np.zeros(10, dtype=bool)
    mask[0] = True

    result = score_subgroup(
        mask,
        target,
        min_size=2,
    )

    assert result is None


def test_generate_candidates() -> None:
    df = make_test_data()

    candidates = generate_candidates(
        df,
        categorical_columns=[
            "sex",
            "occupation",
            "smoking",
            "alcohol",
            "obesity",
        ],
        age_thresholds=[60],
    )

    descriptions = {
        c.description
        for c in candidates
    }

    assert "smoking == 1" in descriptions
    assert "age >= 60" in descriptions
    assert "age < 60" in descriptions


def test_pair_search_recovers_planted_group() -> None:
    df = make_test_data()

    candidates = generate_candidates(
        df,
        categorical_columns=[
            "sex",
            "occupation",
            "smoking",
            "alcohol",
            "obesity",
        ],
        age_thresholds=[60],
    )

    results = search_pair_subgroups(
        candidates,
        df["emm_residual"].to_numpy(),
        min_size=5,
    )

    assert (
        results.iloc[0]["description"]
        == "smoking == 1 AND age >= 60"
    )


def test_complete_search_recovers_planted_group() -> None:
    df = make_test_data()

    results = search_subgroups(
        df,
        target_column="emm_residual",
        categorical_columns=[
            "sex",
            "occupation",
            "smoking",
            "alcohol",
            "obesity",
        ],
        age_thresholds=[60],
        min_size=5,
        max_depth=2,
    )

    assert (
        results.iloc[0]["description"]
        == "smoking == 1 AND age >= 60"
    )

    assert results.iloc[0]["depth"] == 2