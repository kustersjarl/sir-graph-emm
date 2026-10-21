import numpy as np
import pandas as pd
import pytest

from sir_graph.emm import search_model_a_subgroups
from sir_graph.experiments import CATEGORICAL, search_model_a_fast
from sir_graph.scenarios import (
    build_world,
    harm,
    major_runs,
    make_run_plan,
    observe,
    run_plan,
    superspreaders,
)


@pytest.fixture(scope="module")
def people():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "node_id": np.arange(n),
            "age": rng.integers(1, 80, n),
            "sex": rng.choice(["Male", "Female"], n),
            "occupation": rng.choice(["Student", "Healthcare", "Sales and services"], n),
            "smoking": rng.integers(0, 2, n),
            "alcohol": rng.integers(0, 2, n),
            "obesity": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "infection_rate": rng.uniform(0.25, 0.45, n),
        }
    )


@pytest.mark.parametrize(
    "scenario",
    [
        harm("obesity == 1", 3.0, "smoking == 1 AND obesity == 0", 1.5, mean_degree=8.0, p=0.1),
        # heterogeneous infectiousness: weighted exposures are not integers
        superspreaders("occupation == Healthcare AND smoking == 1", mean_degree=8.0, p=0.1),
    ],
    ids=["harm", "uneven_infectiousness"],
)
def test_fast_model_a_matches_original(people, scenario):
    world = build_world(people, scenario, seed=3)
    runs = major_runs(run_plan(world, make_run_plan(len(people), n_runs=15, n_seeds=5, seed=4)))
    obs = observe(world, runs).model_a

    original = search_model_a_subgroups(people, obs, categorical_columns=CATEGORICAL)
    fast = search_model_a_fast(people, obs, categorical_columns=CATEGORICAL)

    assert len(fast) == len(original) > 0
    merged = original.merge(fast, on="description", suffixes=("", "_fast"))
    assert len(merged) == len(original)
    for column in ("size", "observations", "infections", "rest_infections"):
        assert (merged[column] == merged[f"{column}_fast"]).all()
    for column in ("group_escape_exposure", "group_hazard", "rest_hazard", "log_rate_ratio", "quality"):
        assert np.allclose(merged[column], merged[f"{column}_fast"], rtol=1e-8, atol=1e-12)
    # same ranking apart from near-ties
    assert original.description.head(20).tolist() == fast.description.head(20).tolist()
