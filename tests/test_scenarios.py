import numpy as np
import pandas as pd
import pytest

from sir_graph.exposure import run_metrics
from sir_graph.scenarios import (
    Description,
    Plant,
    Scenario,
    assign_degrees,
    build_world,
    calibrate_beta,
    calibrate_plant,
    diverse_top,
    exposure_decoy,
    harm,
    hidden_susceptibility,
    jaccard,
    leave_one_out,
    make_run_plan,
    node_table,
    observe,
    null_scenario,
    plant_diagnostics,
    recovery_report,
    replicate,
    run_plan,
    sample_degree,
    superspreaders,
)


@pytest.fixture(scope="module")
def people():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "node_id": np.arange(n),
            "age": rng.integers(1, 75, n),
            "sex": rng.choice(["Male", "Female"], n),
            "occupation": rng.choice(["Student", "Healthcare", "Sales and services"], n),
            "smoking": rng.integers(0, 2, n),
            "alcohol": rng.integers(0, 2, n),
            "obesity": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "infection_rate": rng.uniform(0.25, 0.45, n),
        }
    )


SMALL = dict(mean_degree=8.0)


def test_description_parse_roundtrip_and_mask(people):
    d = Description.parse("age >= 40 AND occupation == Sales and services")
    assert str(d) == "age >= 40 AND occupation == Sales and services"
    expected = (people.age >= 40) & (people.occupation == "Sales and services")
    assert np.array_equal(d.mask(people), expected.to_numpy())


def test_description_matches_ignores_order():
    d = Description.parse("age >= 60 AND smoking == 1")
    assert d.matches("smoking == 1 AND age >= 60")
    assert not d.matches("smoking == 1")
    assert not d.matches("smoking == 1 AND age >= 65")


def test_description_rejects_bad_input():
    with pytest.raises(ValueError):
        Description.parse("age is large")
    with pytest.raises(ValueError):
        Description(())


def test_sample_degree_is_valid():
    rng = np.random.default_rng(1)
    d = sample_degree(400, 10.0, 5.0, rng=rng)
    assert d.sum() % 2 == 0 and d.min() >= 1 and d.max() <= 100
    assert abs(d.mean() - 10.0) < 1.0


def test_assign_degrees_preserves_multiset_and_orders_groups():
    rng = np.random.default_rng(2)
    base = sample_degree(500, 12.0, 5.0, rng=rng)
    high = np.zeros(500, dtype=bool)
    high[:60] = True
    low = np.zeros(500, dtype=bool)
    low[60:150] = True
    out = assign_degrees(base, high=high, low=low, rng=rng)

    assert np.array_equal(np.sort(out), np.sort(base))
    assert out[high].min() >= out[~high & ~low].max() - 1  # high block sits at the top
    assert out[low].max() <= out[~high & ~low].min() + 1
    with pytest.raises(ValueError):
        assign_degrees(base, high=high, low=high, rng=rng)


def test_plants_must_be_disjoint_and_nonempty(people):
    a = Plant("a", Description.parse("age >= 30"))
    b = Plant("b", Description.parse("age >= 50"))
    with pytest.raises(ValueError, match="overlap"):
        build_world(people, Scenario("x", (a, b), **SMALL), seed=0)
    c = Plant("c", Description.parse("age >= 500"))
    with pytest.raises(ValueError, match="matches nobody"):
        build_world(people, Scenario("x", (c,), **SMALL), seed=0)
    with pytest.raises(ValueError, match="unique"):
        Scenario("x", (a, a))


def test_decoy_gets_high_degree_but_normal_parameters(people):
    world = build_world(people, exposure_decoy("alcohol == 1 AND age < 30", **SMALL), seed=3)
    mask = world.masks["decoy"]
    degree = world.graph.degree()
    assert degree[mask].mean() > 1.5 * degree[~mask].mean()
    s, f = world.params.susceptibility, world.params.infectiousness
    assert np.allclose(s, s[0]) and np.allclose(f, 1.0)  # homogeneous base, no parameter effect


def test_hidden_susceptibility_world(people):
    world = build_world(people, hidden_susceptibility("smoking == 1 AND alcohol == 1", 2.0, **SMALL), seed=3)
    mask = world.masks["hidden"]
    s = world.params.susceptibility
    assert np.allclose(s[mask], 2.0 * s[~mask][0])
    assert world.graph.degree()[mask].mean() < world.graph.degree()[~mask].mean()


def test_superspreader_traits_have_exact_mean_and_large_variance(people):
    world = build_world(people, superspreaders("smoking == 1", 0.25, **SMALL), seed=4)
    mask = world.masks["superspreaders"]
    f = world.params.infectiousness
    assert f[mask].mean() == pytest.approx(1.0)
    assert f[mask].var() > 2.0
    assert np.allclose(f[~mask], 1.0)


def test_data_susceptibility_mode_uses_column(people):
    world = build_world(people, null_scenario(base_susceptibility="data", **SMALL), seed=0)
    assert np.allclose(world.params.susceptibility, people.infection_rate)


def test_world_is_deterministic(people):
    sc = superspreaders("smoking == 1", 0.25, **SMALL)
    a, b = build_world(people, sc, seed=5), build_world(people, sc, seed=5)
    assert a.graph == b.graph
    assert np.array_equal(a.params.infectiousness, b.params.infectiousness)


def test_without_removes_effect_but_keeps_network(people):
    sc = harm("obesity == 1", 3.0, "smoking == 1 AND obesity == 0", 1.5, **SMALL)
    world = build_world(people, sc, seed=6)
    counterfactual = world.without("small_strong")
    small, large = world.masks["small_strong"], world.masks["large_mild"]

    assert counterfactual.graph == world.graph
    s0, s1 = world.params.susceptibility, counterfactual.params.susceptibility
    assert np.allclose(s1[small], s1[~small & ~large][0])
    assert np.allclose(s1[large], s0[large])  # the other plant is untouched
    assert not counterfactual.scenario.plant("small_strong").has_parameter_effect


def test_run_plan_is_reproducible(people):
    world = build_world(people, null_scenario(**SMALL), seed=7)
    plan = make_run_plan(300, n_runs=4, n_seeds=3, seed=8)
    again = make_run_plan(300, n_runs=4, n_seeds=3, seed=8)
    one, two = run_plan(world, plan), run_plan(world, again)
    assert all(np.array_equal(a.infection_time, b.infection_time) for a, b in zip(one, two))


def test_node_table_and_diagnostics(people):
    world = build_world(people, exposure_decoy("alcohol == 1 AND age < 30", **SMALL), seed=9)
    runs = run_plan(world, make_run_plan(300, n_runs=30, n_seeds=3, seed=10))
    metrics = run_metrics(world.graph, runs)
    table = node_table(world, observe(world, runs))

    assert len(table) == 300 and "plant_decoy" in table
    assert table.n_infected.sum() == metrics.natural.sum()
    assert table.infection_freq.dropna().between(0, 1).all()
    assert (table.at_risk_runs <= 30).all()
    # analysis observations agree with the per-run metrics
    assert np.array_equal(table.exposure_total, np.where(metrics.is_seed, 0, metrics.exposure).sum(axis=0))
    assert np.array_equal(table.offspring_total, np.where(metrics.natural, metrics.offspring, 0).sum(axis=0))
    assert np.array_equal(table.opportunity_total, np.where(metrics.natural, metrics.opportunity, 0).sum(axis=0))
    # homogeneous infectiousness, so weighting changes nothing for exposure
    assert np.allclose(table.weighted_exposure_total, table.exposure_total)

    diag = plant_diagnostics(world, metrics)
    assert diag.loc["decoy", "n_in"] == world.masks["decoy"].sum()
    assert diag.loc["decoy", "mean_degree_in"] > diag.loc["decoy", "mean_degree_out"]


def test_calibrate_plant_rejects_bad_bracket(people):
    world = build_world(people, hidden_susceptibility("smoking == 1 AND alcohol == 1", 2.0, **SMALL), seed=11)
    plan = make_run_plan(300, n_runs=15, n_seeds=3, seed=12)
    with pytest.raises(ValueError, match="does not enclose"):
        calibrate_plant(world, "hidden", plan, bracket=(2.0, 2.1))


def test_calibrate_plant_returns_world_inside_bracket(people):
    world = build_world(people, hidden_susceptibility("smoking == 1 AND alcohol == 1", 2.0, **SMALL), seed=11)
    plan = make_run_plan(300, n_runs=15, n_seeds=3, seed=12)
    calibrated, info = calibrate_plant(world, "hidden", plan, bracket=(1.0, 12.0), tol=0.3, max_iter=3)
    assert 1.0 <= info["multiplier"] <= 12.0 and len(info["history"]) >= 3
    assert calibrated.scenario.plant("hidden").susceptibility == info["multiplier"]
    assert calibrated.graph == world.graph


def test_leave_one_out_columns(people):
    sc = harm("obesity == 1", 3.0, "smoking == 1 AND obesity == 0", 1.5, **SMALL)
    world = build_world(people, sc, seed=13)
    table = leave_one_out(world, make_run_plan(300, n_runs=20, n_seeds=3, seed=14))
    assert set(table.index) == {"small_strong", "large_mild"}
    assert {"size", "contribution", "se", "per_member"} <= set(table.columns)


def test_decoy_has_no_parameter_effect_so_no_loo_row(people):
    world = build_world(people, exposure_decoy("alcohol == 1 AND age < 30", **SMALL), seed=9)
    assert leave_one_out(world, make_run_plan(300, n_runs=5, n_seeds=3, seed=1)).empty


def test_jaccard():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 0], dtype=bool)
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert np.isnan(jaccard(~a & a, ~a & a))


def test_recovery_report_uses_returned_descriptions(people):
    world = build_world(people, exposure_decoy("alcohol == 1 AND age < 30", **SMALL), seed=9)
    results = pd.DataFrame({"description": ["smoking == 1", "age < 30 AND alcohol == 1", "sex == Male"]})
    report = recovery_report(results, world, top_k=3)
    assert report.loc["decoy", "rank"] == 2
    assert report.loc["decoy", "best_of_top3_jaccard"] == pytest.approx(1.0)
    assert report.loc["decoy", "top1_jaccard"] < 1.0  # the wrong top result is not rewarded

    missing = recovery_report(pd.DataFrame({"description": ["smoking == 1"]}), world, top_k=1)
    assert np.isnan(missing.loc["decoy", "rank"])


def test_replicate_collects_one_row_per_replicate(people):
    def analyze(world, runs, metrics):
        return {"mean_final_size": float(np.mean([r.final_size for r in runs]))}

    out = replicate(
        people, null_scenario(**SMALL), analyze, n_replicates=3, n_runs=5, base_seed=20
    )
    assert out.replicate.tolist() == [0, 1, 2] and out.mean_final_size.notna().all()


def test_null_scenario_has_empty_ground_truth_tables(people):
    world = build_world(people, null_scenario(**SMALL), seed=15)
    plan = make_run_plan(300, n_runs=5, n_seeds=3, seed=16)
    metrics = run_metrics(world.graph, run_plan(world, plan))
    assert plant_diagnostics(world, metrics).empty
    assert leave_one_out(world, plan).empty
    report = recovery_report(pd.DataFrame({"description": ["smoking == 1"]}), world, top_k=1)
    assert report.empty


def test_recovery_report_flags_recovered(people):
    world = build_world(people, exposure_decoy("alcohol == 1 AND age < 30", **SMALL), seed=9)
    report = recovery_report(pd.DataFrame({"description": ["alcohol == 1 AND age < 30"]}), world, top_k=1)
    assert bool(report.loc["decoy", "recovered"])
    report = recovery_report(pd.DataFrame({"description": ["smoking == 1"]}), world, top_k=1)
    assert not bool(report.loc["decoy", "recovered"])


def test_diverse_top_skips_near_duplicates(people):
    results = pd.DataFrame(
        {
            "description": ["age < 30", "age < 31", "sex == Male", "smoking == 1"],
            "quality": [1.0, 0.99, 0.5, -0.1],
        }
    )
    top = diverse_top(results, people, q=2, gamma=0.1)
    assert top.description.tolist() == ["age < 30", "sex == Male"]
    assert len(diverse_top(results, people, q=10)) == 3  # non-positive quality is never picked


def test_not_equal_condition(people):
    mask = Description.parse("occupation != Student").mask(people)
    assert np.array_equal(mask, (people.occupation != "Student").to_numpy())


def test_calibrate_beta_hits_attack_rate(people):
    world = build_world(people, null_scenario(**SMALL), seed=17)
    plan = make_run_plan(300, n_runs=20, n_seeds=5, seed=18)
    p, info = calibrate_beta(world, plan, attack_rate=0.4, tol=0.02)
    assert 0.01 <= p <= 0.5 and abs(info["gap"]) <= 0.05


def test_leave_one_out_skips_minor_outbreaks(people):
    sc = harm("obesity == 1", 3.0, "smoking == 1 AND obesity == 0", 1.5, **SMALL)
    world = build_world(people, sc, seed=13)
    table = leave_one_out(world, make_run_plan(300, n_runs=20, n_seeds=3, seed=14))
    assert (table.runs <= 20).all() and (table.runs > 0).all()
