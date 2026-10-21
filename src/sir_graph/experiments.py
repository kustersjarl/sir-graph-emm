from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import brentq
from sir_graph.analysis import ModelAObservations, fit_degree_baseline
from sir_graph.dynamics import SimulationResult
from sir_graph.emm import (
    _mask_key,
    generate_candidates,
    score_model_a_subgroup,
    score_model_b_subgroup,
    search_model_b_subgroups,
    search_subgroups,
)
from sir_graph.scenarios import (
    RECOVERY_JACCARD,
    Description,
    Observations,
    World,
    build_world,
    calibrate_beta,
    diverse_top,
    harm,
    jaccard,
    leave_one_out,
    load_people,
    major_runs,
    make_run_plan,
    node_table,
    null_scenario,
    observe,
    run_plan,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic_data.csv"
OUT = ROOT / "data" / "experiments"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CATEGORICAL = ["sex", "occupation", "smoking", "alcohol", "obesity"]
TOP_K = 10

# effect column of each method's search output
EFFECT = {
    "B0_infection": "difference",
    "B0+_infection": "difference",
    "B0_offspring": "difference",
    "B0+_offspring": "difference",
    "model_a": "log_rate_ratio",
    "model_b": "log_rate_ratio",
}

@dataclass(frozen=True, eq=False)
class ModelAIndex:
    escape: NDArray[np.float64]  # weighted escape exposure per person
    observations: NDArray[np.int64]  # person-run rows per person
    infections: NDArray[np.int64]  # natural infections per person
    values: NDArray[np.float64]  # distinct weighted exposures on the infection step
    counts: sparse.csr_matrix  # (values, people): infections of each person at each value


def model_a_index(obs: ModelAObservations, n_people: int) -> ModelAIndex:
    node = obs.node_index
    infected = obs.infected
    values, value_index = np.unique(obs.weighted_infection_exposure[infected], return_inverse=True)
    counts = sparse.csr_matrix(
        (np.ones(value_index.size), (value_index, node[infected])),
        shape=(values.size, n_people),
    )
    return ModelAIndex(
        escape=np.bincount(node, weights=obs.weighted_escape_exposure, minlength=n_people),
        observations=np.bincount(node, minlength=n_people),
        infections=np.bincount(node[infected], minlength=n_people),
        values=values,
        counts=counts,
    )


def _fit_hazard(escape: float, counts: NDArray[np.float64], values: NDArray[np.float64]) -> float | None:
    if counts.sum() == 0:
        return 0.0
    if escape <= 0:
        return None
    present = counts > 0
    c, v = counts[present], values[present]

    def score(hazard: float) -> float:
        denom = np.expm1(hazard * v)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            term = np.divide(v, denom, out=np.zeros_like(v), where=denom != 0)
        return float(-escape + (c * term).sum())

    upper = 1.0
    while score(upper) > 0:
        upper *= 2.0
        if upper > 1e12:
            return None
    return float(brentq(score, 1e-12, upper, xtol=1e-12, rtol=1e-12))


def score_model_a_fast(mask: NDArray[np.bool_], index: ModelAIndex, *, min_size: int = 20) -> dict | None:
    n_people = mask.size
    n_group = int(mask.sum())
    if n_group < min_size or n_people - n_group < min_size:
        return None
    weights = mask.astype(np.float64)
    group_counts = index.counts @ weights
    rest_counts = index.counts @ (1.0 - weights)
    group_escape = float(index.escape[mask].sum())
    rest_escape = float(index.escape[~mask].sum())
    group_hazard = _fit_hazard(group_escape, group_counts, index.values)
    rest_hazard = _fit_hazard(rest_escape, rest_counts, index.values)
    if group_hazard is None or rest_hazard is None or group_hazard <= 0 or rest_hazard <= 0:
        return None
    log_rate_ratio = float(np.log(group_hazard / rest_hazard))
    support = n_group / n_people
    return {
        "size": n_group,
        "support": support,
        "observations": int(index.observations[mask].sum()),
        "rest_observations": int(index.observations[~mask].sum()),
        "infections": int(index.infections[mask].sum()),
        "rest_infections": int(index.infections[~mask].sum()),
        "exposure_type": "weighted",
        "group_escape_exposure": group_escape,
        "rest_escape_exposure": rest_escape,
        "group_hazard": group_hazard,
        "rest_hazard": rest_hazard,
        "rate_ratio": group_hazard / rest_hazard,
        "log_rate_ratio": log_rate_ratio,
        "quality": np.sqrt(support) * log_rate_ratio,
    }


def search_model_a_fast(
    people: pd.DataFrame,
    obs: ModelAObservations,
    *,
    categorical_columns: list[str],
    min_size: int = 20,
) -> pd.DataFrame:
    index = model_a_index(obs, len(people))
    candidates = generate_candidates(people, categorical_columns=categorical_columns)

    def scored(pairs) -> pd.DataFrame:
        rows = []
        for description, mask in pairs:
            score = score_model_a_fast(mask, index, min_size=min_size)
            if score is not None:
                rows.append({"description": description, **score})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("quality", ascending=False).reset_index(drop=True)

    def conjunctions():
        seen = {_mask_key(c.mask) for c in candidates}
        for a, b in combinations(candidates, 2):
            if a.attribute == b.attribute:
                continue
            mask = a.mask & b.mask
            key = _mask_key(mask)
            if key in seen:
                continue
            seen.add(key)
            yield f"{a.description} AND {b.description}", mask

    single = scored((c.description, c.mask) for c in candidates)
    pairs = scored(conjunctions())
    frames = []
    for depth, frame in ((1, single), (2, pairs)):
        if not frame.empty:
            frame.insert(0, "depth", depth)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("quality", ascending=False).reset_index(drop=True)


def search_all(world: World, runs: list[SimulationResult]) -> tuple[dict[str, pd.DataFrame], Observations]:
    obs = observe(world, runs)
    table = node_table(world, obs)
    degree = table["degree"].to_numpy()
    table["infection_resid"] = fit_degree_baseline(degree, table["infection_freq"]).residuals
    table["offspring_resid"] = fit_degree_baseline(degree, table["mean_secondary"]).residuals

    def naive(target: str) -> pd.DataFrame:
        return search_subgroups(table, target_column=target, categorical_columns=CATEGORICAL)

    results = {
        "B0_infection": naive("infection_freq"),
        "B0+_infection": naive("infection_resid"),
        "B0_offspring": naive("mean_secondary"),
        "B0+_offspring": naive("offspring_resid"),
        "model_a": search_model_a_fast(world.people, obs.model_a, categorical_columns=CATEGORICAL),
        "model_b": search_model_b_subgroups(world.people, obs.model_b, categorical_columns=CATEGORICAL),
    }
    return results, obs


def rerank(results: pd.DataFrame, method: str, alpha: float) -> pd.DataFrame:
    quality = results["support"] ** alpha * results[EFFECT[method]]
    return results.assign(quality=quality).sort_values("quality", ascending=False, ignore_index=True)

def _beta_path(tag: str) -> Path:
    return OUT / f"beta_{tag}.json"


def _thresholds_path(tag: str, n_runs: int) -> Path:
    return OUT / f"thresholds_{tag}_R{n_runs}.csv"


def beta(people: pd.DataFrame, tag: str, *, n_runs: int = 50, seed: int = 999) -> float:
    """Calibrated transmission probability p for this population, computed once per tag."""
    path = _beta_path(tag)
    
    if path.exists():
        return float(json.loads(path.read_text())["p"])
    
    start = time.time()
    world = build_world(people, null_scenario(), seed=seed)
    plan = make_run_plan(len(people), n_runs=n_runs, n_seeds=world.scenario.n_seeds, seed=seed)
    p, info = calibrate_beta(world, plan)
    path.write_text(json.dumps({"p": p, "gap": info["gap"], "runs": n_runs, "history": info["history"]}, indent=2))
    print(f"calibrated p = {p:.5f} (attack rate gap {info['gap']:+.4f}, {time.time() - start:.0f}s)")
    return p


def load_thresholds(tag: str, n_runs: int) -> pd.DataFrame | None:
    path = _thresholds_path(tag, n_runs)
    return pd.read_csv(path) if path.exists() else None


def _threshold(thresholds: pd.DataFrame | None, method: str, alpha: float) -> float:
    if thresholds is None:
        return float("nan")
    match = thresholds[(thresholds["method"] == method) & np.isclose(thresholds["alpha"], alpha)]
    return float(match["threshold"].iloc[0]) if len(match) else float("nan")


def _plant_rank(ranked: pd.DataFrame, description: Description) -> float:
    hits = [i + 1 for i, text in enumerate(ranked["description"]) if description.matches(text)]
    return float(hits[0]) if hits else float("nan")


def plant_rows(
    world: World,
    results: dict[str, pd.DataFrame],
    thresholds: pd.DataFrame | None,
) -> list[dict]:
    rows = []
    
    for method, found in results.items():
        
        for alpha in ALPHAS:
            ranked = rerank(found, method, alpha)
            top = diverse_top(ranked, world.people, q=TOP_K)
            top_masks = [Description.parse(d).mask(world.people) for d in top["description"]]
            threshold = _threshold(thresholds, method, alpha)
            
            for plant in world.scenario.plants:
                truth = world.masks[plant.name]
                rank = _plant_rank(ranked, plant.description)
                quality = ranked["quality"].iloc[int(rank) - 1] if not np.isnan(rank) else float("nan")
                best = max((jaccard(mask, truth) for mask in top_masks), default=0.0)
                rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "plant": plant.name,
                        "size": int(truth.sum()),
                        "rank": rank,
                        "quality": quality,
                        "threshold": threshold,
                        "significant": quality > threshold if not np.isnan(threshold) else np.nan,
                        f"best_top{TOP_K}_jaccard": best,
                        "recovered": best >= RECOVERY_JACCARD,
                    }
                )
    return rows


def effect_rows(world: World, obs: Observations) -> list[dict]:
    rows = []
    
    for plant in world.scenario.plants:
        mask = world.masks[plant.name]
        a = score_model_a_subgroup(mask, obs.model_a, direction="both")
        b = score_model_b_subgroup(mask, obs.model_b, direction="both")
        true_a, true_b = np.log(plant.susceptibility), np.log(plant.infectiousness)
        est_a = a["log_rate_ratio"] if a else float("nan")
        est_b = b["log_rate_ratio"] if b else float("nan")
        rows.append(
            {
                "plant": plant.name,
                "size": int(mask.sum()),
                "true_log_susceptibility": true_a,
                "model_a_log_rr": est_a,
                "model_a_error": est_a - true_a,
                "true_log_infectiousness": true_b,
                "model_b_log_rr": est_b,
                "model_b_error": est_b - true_b,
            }
        )
    return rows

def _map(fn: Callable, tasks: Iterable[tuple], jobs: int) -> list:
    tasks = list(tasks)
    if jobs <= 1:
        return [fn(*task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        return list(pool.map(fn, *zip(*tasks)))


def _null_network(
    people: pd.DataFrame, p: float, m: int, n_networks: int, plans_per_network: int, n_runs: int, base_seed: int
) -> list[dict]:
    """All null batches on network m,the network is built once."""
    rows = []
    scenario = null_scenario(p=p)
    world = build_world(people, scenario, seed=base_seed + m)
    for b in range(plans_per_network):
        start = time.time()
        plan = make_run_plan(len(people), n_runs=n_runs, n_seeds=scenario.n_seeds, seed=base_seed + 100 * m + b)
        all_runs = run_plan(world, plan)
        runs = major_runs(all_runs)
        results, _ = search_all(world, runs)
        for method, found in results.items():
            for alpha in ALPHAS:
                top = rerank(found, method, alpha).iloc[0]
                rows.append(
                    {
                        "network": m,
                        "batch": b,
                        "method": method,
                        "alpha": alpha,
                        "max_quality": top["quality"],
                        "top_description": top["description"],
                        "top_size": top["size"],
                        "minor_outbreaks": len(all_runs) - len(runs),
                    }
                )
        print(f"null network {m + 1}/{n_networks} batch {b + 1}/{plans_per_network} ({time.time() - start:.0f}s)", flush=True)
    return rows


def calibration(
    people: pd.DataFrame,
    p: float,
    *,
    n_networks: int,
    plans_per_network: int,
    n_runs: int,
    jobs: int = 1,
    base_seed: int = 1000,
) -> pd.DataFrame:
    tasks = ((people, p, m, n_networks, plans_per_network, n_runs, base_seed) for m in range(n_networks))
    return pd.DataFrame([row for rows in _map(_null_network, tasks, jobs) for row in rows])


def thresholds(null: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    return (
        null.groupby(["method", "alpha"])["max_quality"]
        .agg(batches="size", mean="mean", sd="std", threshold=lambda q: q.quantile(level))
        .reset_index()
    )


def _harm_network(
    people: pd.DataFrame, p: float, m: int, n_networks: int, n_runs: int, thresholds: pd.DataFrame | None, base_seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    start = time.time()
    world = build_world(people, harm(p=p), seed=base_seed + m)
    plan = make_run_plan(len(people), n_runs=n_runs, n_seeds=world.scenario.n_seeds, seed=base_seed + m)
    all_runs = run_plan(world, plan)
    runs = major_runs(all_runs)
    results, obs = search_all(world, runs)

    truth = leave_one_out(world, plan)
    truth_rows = [{"network": m, **row} for row in truth.reset_index().to_dict("records")]
    order = truth["contribution"].rank(ascending=False)
    rank_rows = []
    for row in plant_rows(world, results, thresholds):
        row["true_contribution"] = truth.loc[row["plant"], "contribution"]
        row["true_order"] = order[row["plant"]]
        rank_rows.append({"network": m, "minor_outbreaks": len(all_runs) - len(runs), **row})
    effect_list = [{"network": m, **row} for row in effect_rows(world, obs)]
    print(f"harm network {m + 1}/{n_networks} ({time.time() - start:.0f}s)", flush=True)
    return rank_rows, effect_list, truth_rows


def harm_sweep(
    people: pd.DataFrame,
    p: float,
    *,
    n_networks: int,
    n_runs: int,
    thresholds: pd.DataFrame | None,
    jobs: int = 1,
    base_seed: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = ((people, p, m, n_networks, n_runs, thresholds, base_seed) for m in range(n_networks))
    parts = _map(_harm_network, tasks, jobs)
    return tuple(pd.DataFrame([row for part in parts for row in part[i]]) for i in range(3))


def harm_summary(ranks: pd.DataFrame) -> pd.DataFrame:
    keys = ["network", "method", "alpha"]
    wide = ranks.pivot_table(index=keys, columns="plant", values=["rank", "true_order"], dropna=False)
    found = wide["rank"].fillna(np.inf)  # a plant that is never found ranks last
    small_first = found["small_strong"] < found["large_mild"]
    truly_small_first = wide[("true_order", "small_strong")] < wide[("true_order", "large_mild")]
    out = pd.DataFrame(
        {
            "median_rank_small": wide[("rank", "small_strong")],
            "median_rank_large": wide[("rank", "large_mild")],
            "ranks_like_truth": small_first == truly_small_first,
        }
    )
    return out.groupby(level=["method", "alpha"]).agg(
        networks=("ranks_like_truth", "size"),
        median_rank_small=("median_rank_small", "median"),
        median_rank_large=("median_rank_large", "median"),
        ranks_like_truth=("ranks_like_truth", "mean"),
    ).reset_index()


def _write(frames: dict[str, pd.DataFrame], tag: str) -> None:
    for name, frame in frames.items():
        if not frame.empty:
            frame.to_csv(OUT / f"{name}_{tag}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["beta", "calibration", "harm"])
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--networks", type=int, default=10)
    parser.add_argument("--plans", type=int, default=8)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)), help="networks simulated in parallel"
    )
    args = parser.parse_args()

    tag = args.tag or f"n{args.nodes}"
    OUT.mkdir(parents=True, exist_ok=True)
    people = load_people(str(DATA), n_nodes=args.nodes, seed=0)
    p = beta(people, tag)
    if args.experiment == "beta":
        return

    if args.experiment == "calibration":
        null = calibration(
            people, p, n_networks=args.networks, plans_per_network=args.plans, n_runs=args.runs, jobs=args.jobs
        )
        null.to_csv(OUT / f"calibration_{tag}_R{args.runs}.csv", index=False)
        summary = thresholds(null)
        summary.to_csv(_thresholds_path(tag, args.runs), index=False)

    else:
        ranks, effects, truth = harm_sweep(
            people,
            p,
            n_networks=args.networks,
            n_runs=args.runs,
            thresholds=load_thresholds(tag, args.runs),
            jobs=args.jobs,
        )
        summary = harm_summary(ranks)
        _write({"harm_ranks": ranks, "harm_effects": effects, "harm_truth": truth, "harm_summary": summary}, tag)
        print()
        print(effects.groupby("plant").median(numeric_only=True).drop(columns="network").round(3).to_string())

    print()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
