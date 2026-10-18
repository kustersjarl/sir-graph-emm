from __future__ import annotations
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sir_graph.dynamics import SimulationResult, simulate
from sir_graph.exposure import RunMetrics, run_metrics
from sir_graph.graph import Graph
from sir_graph.havel_hakimi import havel_hakimi, randomize_degree_preserving
from sir_graph.parameters import NodeParams
from sir_graph.states import choose_seeds

"""
people = load_people("data/synthetic_data.csv", n_nodes=1000, seed=0)
world = build_world(people, exposure_decoy(), seed=1)
plan = make_run_plan(world.graph.n_nodes, n_runs=300, n_seeds=5, seed=2)
runs = run_plan(world, plan)
metrics = run_metrics(world.graph, runs)
table = node_table(world, metrics)      # one row per person
plant_diagnostics(world, metrics)       # did the plant do what we designed?
"""

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

CONDITION_RE = re.compile(r"^(\w+) (==|>=|<) (.+)$")

@dataclass(frozen=True)
class Condition:
    attribute: str
    op: Literal["==", ">=", "<"]
    value: int | float | str

    def mask(self, df: pd.DataFrame) -> BoolArray:
        col = df[self.attribute]
        if self.op == "==":
            return (col == self.value).to_numpy(dtype=bool)
        if self.op == ">=":
            return (col >= self.value).to_numpy(dtype=bool)
        return (col < self.value).to_numpy(dtype=bool)

    def __str__(self) -> str:
        return f"{self.attribute} {self.op} {self.value}"


@dataclass(frozen=True)
class Description:
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("at least one condition")

    @classmethod
    def parse(cls, text: str) -> Description:
        conditions = []
        
        for part in text.split(" AND "):
            match = CONDITION_RE.match(part.strip())
            if match is None:
                raise ValueError(f"cannot parse condition {part!r}")
            attribute, op, raw = match.groups()
            value: int | float | str
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            conditions.append(Condition(attribute, op, value))
        return cls(tuple(conditions))

    def mask(self, df: pd.DataFrame) -> BoolArray:
        out = np.ones(len(df), dtype=bool)
        for condition in self.conditions:
            out &= condition.mask(df)
        return out

    def matches(self, text: str) -> bool:
        return {str(c) for c in self.conditions} == set(text.split(" AND "))

    def __str__(self) -> str:
        return " AND ".join(str(c) for c in self.conditions)


@dataclass(frozen=True)
class Plant:
    """
    Susceptibility and infectiousness are multipliers.
    Degree can be assigned randomly or from the high/low end of the sequence.
    Dispersion gives group members different infectiousness values.
    """
    name: str
    description: Description
    susceptibility: float = 1.0
    infectiousness: float = 1.0
    infectiousness_dispersion: float | None = None
    degree: Literal["random", "high", "low"] = "random"

    def __post_init__(self) -> None:
        if self.susceptibility <= 0 or self.infectiousness <= 0:
            raise ValueError("multipliers must be positive")
        if (self.infectiousness_dispersion is not None and self.infectiousness_dispersion <= 0):
            raise ValueError("infectiousness_dispersion (gamma shape) must be positive")

    @property
    def has_parameter_effect(self) -> bool:
        return (
            self.susceptibility != 1.0
            or self.infectiousness != 1.0
            or self.infectiousness_dispersion is not None
        )

    def without_parameter_effect(self) -> Plant:
        return replace(
            self,
            susceptibility=1.0,
            infectiousness=1.0,
            infectiousness_dispersion=None,
        )


@dataclass(frozen=True)
class Scenario:
    """
    Susceptibility can be set to the population mean or taken from the data.
    A degree dispersion of None uses a Poisson distribution.
    """
    name: str
    plants: tuple[Plant, ...] = ()
    base_susceptibility: Literal["homogeneous", "data"] = "homogeneous"
    mean_degree: float = 15.0
    degree_dispersion: float | None = 5.0
    p: float = 0.12
    recovery: float = 0.4
    n_seeds: int = 5
    susceptibility_column: str = "infection_rate"

    def __post_init__(self) -> None:
        names = [plant.name for plant in self.plants]
        if len(set(names)) != len(names):
            raise ValueError(f"plant names must be unique, got {names}")

    def plant(self, name: str) -> Plant:
        for plant in self.plants:
            if plant.name == name:
                return plant
        raise KeyError(name)

    def replace_plant(self, name: str, **changes) -> Scenario:
        plant = self.plant(name)
        return replace(
            self,
            plants=tuple(
                replace(plant, **changes) if p.name == name else p for p in self.plants
            ),
        )


def load_people(path: str, *, n_nodes: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    people = df.sample(n=n_nodes, random_state=seed).reset_index(drop=True)
    people.insert(0, "node_id", np.arange(n_nodes))
    return people


def _is_graphical(degrees: IntArray) -> bool:
    """Return whether these degrees can form a simple graph."""
    d = np.sort(degrees)[::-1]
    n = d.size
    
    if d.sum() % 2 or d.min() < 0 or d.max() >= n:
        return False
    
    cum = np.cumsum(d)
    
    for k in range(1, n + 1):
        if cum[k - 1] > k * (k - 1) + np.minimum(d[k:], k).sum():
            return False
    return True


def sample_degree(
    n_nodes: int,
    mean: float,
    dispersion: float | None,
    *,
    rng: np.random.Generator,
    min_degree: int = 1,
) -> IntArray:
    """Sample degrees, make the sum even, and check that they form a graph."""
    if dispersion is None:
        degrees = rng.poisson(mean, size=n_nodes)
    else:
        degrees = rng.negative_binomial(dispersion, dispersion / (dispersion + mean), size=n_nodes)
        
    degrees = np.clip(degrees, min_degree, max(min_degree, n_nodes // 4)).astype(np.int64)
    
    if degrees.sum() % 2:
        candidates = np.flatnonzero(degrees < n_nodes // 4)
        degrees[rng.choice(candidates)] += 1
        
    if not _is_graphical(degrees):
        raise ValueError(
            "sampled degree sequence is not graphical, lower the mean or dispersion"
        )
    return degrees


def assign_degrees(
    degrees: IntArray,
    *,
    high: BoolArray | None = None,
    low: BoolArray | None = None,
    rng: np.random.Generator,
) -> IntArray:
    """
    Give high nodes the largest degrees and low nodes the smallest.
    Shuffle within each group so the degree sequence itself stays unchanged.
    """
    n = degrees.size
    high = np.zeros(n, dtype=bool) if high is None else high
    low = np.zeros(n, dtype=bool) if low is None else low
    if np.any(high & low):
        raise ValueError("a node cannot be both in a high-degree and a low-degree plant")

    order = np.sort(degrees)
    k_high, k_low = int(high.sum()), int(low.sum())
    
    out = np.empty(n, dtype=np.int64)
    out[np.flatnonzero(high)] = rng.permutation(order[n - k_high :])
    out[np.flatnonzero(low)] = rng.permutation(order[:k_low])
    out[np.flatnonzero(~high & ~low)] = rng.permutation(order[k_low : n - k_high])
    
    return out


@dataclass(frozen=True, eq=False)
class World:
    scenario: Scenario
    people: pd.DataFrame
    graph: Graph
    params: NodeParams
    masks: dict[str, BoolArray]
    seed: int

    def replace_plant(self, name: str, **changes) -> World:
        """same people and graph, but rebuild the parameters"""
        scenario = self.scenario.replace_plant(name, **changes)
        return replace(
            self,
            scenario=scenario,
            params=build_params(self.people, scenario, self.masks, self.seed),
        )

    def without(self, name: str) -> World:
        """Remove this parameter effect without changing the graph."""
        plant = self.scenario.plant(name).without_parameter_effect()
        return self.replace_plant(
            name,
            susceptibility=plant.susceptibility,
            infectiousness=plant.infectiousness,
            infectiousness_dispersion=plant.infectiousness_dispersion,
        )


def plant_masks(people: pd.DataFrame, scenario: Scenario) -> dict[str, BoolArray]:
    masks = {plant.name: plant.description.mask(people) for plant in scenario.plants}
    
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(
                f"plant {name!r} matches nobody: {scenario.plant(name).description}"
            )
    names = list(masks)
    
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = int(np.sum(masks[a] & masks[b]))
            if overlap:
                raise ValueError(f"plants {a!r} and {b!r} overlap in {overlap} people")
    return masks


def build_params(
    people: pd.DataFrame,
    scenario: Scenario,
    masks: dict[str, BoolArray],
    seed: int,
) -> NodeParams:
    n = len(people)
    
    if scenario.base_susceptibility == "homogeneous":
        s = np.full(n, people[scenario.susceptibility_column].mean())
    else:
        s = people[scenario.susceptibility_column].to_numpy(dtype=np.float64).copy()
    f = np.ones(n)
    
    for i, plant in enumerate(scenario.plants):
        mask = masks[plant.name]
        s[mask] *= plant.susceptibility
        if plant.infectiousness_dispersion is None:
            f[mask] *= plant.infectiousness
        else:
            k = plant.infectiousness_dispersion
            rng = np.random.default_rng([seed, 2, i])  # one stream per plant
            draws = rng.gamma(k, 1.0 / k, size=int(mask.sum()))
            draws /= draws.mean()  # heavy tails make the sample mean noisy; pin it to 1
            f[mask] *= plant.infectiousness * draws
    return NodeParams(s, f, np.full(n, scenario.recovery))


def build_world(
    people: pd.DataFrame,
    scenario: Scenario,
    *,
    seed: int,
    swaps_per_edge: int = 10,
) -> World:
    """Build the graph and parameters"""
    n = len(people)
    masks = plant_masks(people, scenario)

    degree_rng = np.random.default_rng([seed, 0])
    swap_rng = np.random.default_rng([seed, 1])

    base = sample_degree(n, scenario.mean_degree, scenario.degree_dispersion, rng=degree_rng)
    high = np.zeros(n, dtype=bool)
    low = np.zeros(n, dtype=bool)
    
    for plant in scenario.plants:
        if plant.degree == "high":
            high |= masks[plant.name]
        elif plant.degree == "low":
            low |= masks[plant.name]
    degrees = assign_degrees(base, high=high, low=low, rng=degree_rng)

    graph = havel_hakimi(degrees)
    graph = randomize_degree_preserving(graph, swaps_per_edge * graph.n_edges, rng=swap_rng)

    return World(
        scenario=scenario,
        people=people,
        graph=graph,
        params=build_params(people, scenario, masks, seed),
        masks=masks,
        seed=seed,
    )


@dataclass(frozen=True, eq=False)
class RunPlan:
    seed_sets: list[IntArray]
    run_seeds: IntArray

    @property
    def n_runs(self) -> int:
        return len(self.seed_sets)


def make_run_plan(n_nodes: int, *, n_runs: int, n_seeds: int, seed: int) -> RunPlan:
    rng = np.random.default_rng([seed, 3])
    seed_sets = [choose_seeds(n_nodes, n_seeds, rng=rng) for _ in range(n_runs)]
    run_seeds = rng.integers(0, 2**31 - 1, size=n_runs, dtype=np.int64)
    return RunPlan(seed_sets, run_seeds)


def run_plan(world: World, plan: RunPlan) -> list[SimulationResult]:
    return [
        simulate(world.graph, seeds, world.scenario.p, world.params, rng=int(run_seed))
        for seeds, run_seed in zip(plan.seed_sets, plan.run_seeds, strict=False)
    ]

def _rate(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0 else float("nan")


def _safe_div(a: NDArray, b: NDArray) -> NDArray[np.float64]:
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def node_table(world: World, metrics: RunMetrics) -> pd.DataFrame:
    """
    Combine each person's data with their simulation results.
    The rate columns describe infections, exposure contacts, and offspring.
    """
    nat, seed = metrics.natural, metrics.is_seed
    n_infected = nat.sum(axis=0)
    at_risk = (~seed).sum(axis=0)
    offspring = np.where(nat, metrics.offspring, 0).sum(axis=0)
    opportunity = np.where(nat, metrics.opportunity, 0).sum(axis=0)
    exposure = metrics.exposure.sum(axis=0)

    df = world.people.copy()
    df["degree"] = world.graph.degree()
    
    for name, mask in world.masks.items():
        df[f"plant_{name}"] = mask
        
    df["n_infected"] = n_infected
    df["at_risk_runs"] = at_risk
    df["infection_freq"] = _safe_div(n_infected.astype(float), at_risk.astype(float))
    df["exposure_total"] = exposure
    df["exposure_rate"] = _safe_div(n_infected.astype(float), exposure.astype(float))
    df["offspring_total"] = offspring
    df["opportunity_total"] = opportunity
    df["offspring_per_opportunity"] = _safe_div(offspring.astype(float), opportunity.astype(float))
    df["mean_secondary"] = _safe_div(offspring.astype(float), n_infected.astype(float))
    return df


def _side_stats(world: World, metrics: RunMetrics, mask: BoolArray) -> dict[str, float]:
    nat, seed = metrics.natural[:, mask], metrics.is_seed[:, mask]
    offspring = metrics.offspring[:, mask][nat]
    mean_off = float(offspring.mean()) if offspring.size else float("nan")
    fano = (
        float(offspring.var() / mean_off)
        if offspring.size and mean_off > 0
        else float("nan")
    )
    return {
        "n": int(mask.sum()),
        "mean_degree": float(world.graph.degree()[mask].mean()),
        "infection_rate": _rate(nat.sum(), (~seed).sum()),
        "exposure_rate": _rate(nat.sum(), metrics.exposure[:, mask].sum()),
        "mean_offspring": mean_off,
        "offspring_fano": fano,
        "offspring_per_opportunity": _rate(offspring.sum(), metrics.opportunity[:, mask][nat].sum()),
    }


def plant_diagnostics(world: World, metrics: RunMetrics) -> pd.DataFrame:
    """Compare each planted group with everyone outside it."""
    rows = []
    for name, mask in world.masks.items():
        inside = _side_stats(world, metrics, mask)
        outside = _side_stats(world, metrics, ~mask)
        row: dict[str, float | str] = {"plant": name}
        for key in inside:
            row[f"{key}_in"] = inside[key]
            row[f"{key}_out"] = outside[key]
        rows.append(row)
    if not rows:  # null scenario
        return pd.DataFrame(index=pd.Index([], name="plant"))
    return pd.DataFrame(rows).set_index("plant")


def _group_vs_population(
    runs: Sequence[SimulationResult],
    mask: BoolArray,
    target: Literal["infection_rate", "mean_offspring"],
) -> tuple[float, float]:
    """Return the group's statistic and the population's across runs"""
    times = np.stack([run.infection_time for run in runs])
    nat, seed = times > 0, times == 0
    
    if target == "infection_rate":
        return (
            _rate(nat[:, mask].sum(), (~seed)[:, mask].sum()),
            _rate(nat.sum(), (~seed).sum()),
        )
        
    offspring = np.stack([run.offspring for run in runs])
    return (
        _rate(offspring[:, mask][nat[:, mask]].sum(), nat[:, mask].sum()),
        _rate(offspring[nat].sum(), nat.sum()),
    )


def calibrate_plant(
    world: World,
    plant_name: str,
    plan: RunPlan,
    *,
    trait: Literal["susceptibility", "infectiousness"] = "susceptibility",
    target: Literal["infection_rate", "mean_offspring"] = "infection_rate",
    bracket: tuple[float, float] = (1.0, 8.0),
    tol: float = 0.005,
    max_iter: int = 12,
) -> tuple[World, dict]:
    """
    Adjust a plant's multiplier to bring its statistic closer to the population's.
    Each trial uses the same run plan. Test the result with a new plan afterwards, since these runs were used for calibration.
    """
    mask = world.masks[plant_name]

    def gap(multiplier: float) -> float:
        trial = world.replace_plant(plant_name, **{trait: multiplier})
        group, everyone = _group_vs_population(run_plan(trial, plan), mask, target)
        return group - everyone

    lo, hi = bracket
    g_lo, g_hi = gap(lo), gap(hi)
    history = [(lo, g_lo), (hi, g_hi)]
    if g_lo > 0 or g_hi < 0:
        raise ValueError(
            f"bracket {bracket} does not enclose a match "
            f"(gap {g_lo:+.4f} .. {g_hi:+.4f}); "
            "widen it or change the plant's degree placement"
        )
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        g = gap(mid)
        history.append((mid, g))
        if abs(g) <= tol:
            break
        lo, hi = (mid, hi) if g < 0 else (lo, mid)

    best, best_gap = min(history, key=lambda item: abs(item[1]))
    calibrated = world.replace_plant(plant_name, **{trait: best})
    return calibrated, {"multiplier": best, "gap": best_gap, "history": history}


def leave_one_out(world: World, plan: RunPlan) -> pd.DataFrame:
    """Measure how much each plant changes the final outbreak size"""
    full = np.array([run.final_size for run in run_plan(world, plan)], dtype=float)
    rows = []
    
    for plant in world.scenario.plants:
        if not plant.has_parameter_effect:
            continue
        without = np.array([run.final_size for run in run_plan(world.without(plant.name), plan)])
        diff = full - without
        size = int(world.masks[plant.name].sum())
        rows.append(
            {
                "plant": plant.name,
                "size": size,
                "contribution": diff.mean(),
                "se": diff.std(ddof=1) / np.sqrt(diff.size),
                "per_member": diff.mean() / size,
            }
        )
    columns = ["plant", "size", "contribution", "se", "per_member"]
    return pd.DataFrame(rows, columns=columns).set_index("plant")

def jaccard(a: BoolArray, b: BoolArray) -> float:
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else float("nan")

def recovery_report(
    results: pd.DataFrame,
    world: World,
    *,
    top_k: int = 10,
    description_column: str = "description",
) -> pd.DataFrame:
    """Show each plant's search rank and overlap with the top results."""
    rows = []
    top = results.head(top_k)
    top_masks = [Description.parse(d).mask(world.people) for d in top[description_column]]
    
    for plant in world.scenario.plants:
        truth = world.masks[plant.name]
        ranks = [
            i + 1
            for i, text in enumerate(results[description_column])
            if plant.description.matches(text)
        ]
        scores = [jaccard(m, truth) for m in top_masks]
        rows.append(
            {
                "plant": plant.name,
                "planted": str(plant.description),
                "rank": ranks[0] if ranks else float("nan"),
                "top1_jaccard": scores[0] if scores else float("nan"),
                f"best_of_top{top_k}_jaccard": max(scores) if scores else float("nan"),
            }
        )
        
    columns = [
        "plant",
        "planted",
        "rank",
        "top1_jaccard",
        f"best_of_top{top_k}_jaccard",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("plant")


def replicate(
    people: pd.DataFrame,
    scenario: Scenario,
    analyze: Callable[[World, list[SimulationResult], RunMetrics], dict],
    *,
    n_replicates: int,
    n_runs: int,
    base_seed: int = 0,
    fresh_network: bool = True,
    prepare: Callable[[World, RunPlan], World] | None = None,
) -> pd.DataFrame:
    rows = []
    world: World | None = None
    
    for r in range(n_replicates):
        if world is None or fresh_network:
            world = build_world(people, scenario, seed=base_seed + r)
        plan = make_run_plan(
            world.graph.n_nodes,
            n_runs=n_runs,
            n_seeds=scenario.n_seeds,
            seed=base_seed + r,
        )
        current = prepare(world, plan) if prepare is not None else world
        runs = run_plan(current, plan)
        rows.append({"replicate": r, **analyze(current, runs, run_metrics(current.graph, runs))})
    return pd.DataFrame(rows)


# THE EXPERIMENTAL SCENARIOS


def null_scenario(**overrides) -> Scenario:
    """calibration run, nothing planted"""
    return Scenario(name="calibration", **overrides)

def exposure_decoy(description: str = "alcohol == 1 AND age < 30", **overrides) -> Scenario:
    """extra contacts"""
    plant = Plant("decoy", Description.parse(description), degree="high")
    return Scenario(name="exposure_decoy", plants=(plant,), **overrides)


def hidden_susceptibility(
    description: str = "alcohol == 1 AND smoking == 1",
    multiplier: float = 2.0,
    **overrides,
) -> Scenario:
    """low-degree group with raised susceptibility"""
    plant = Plant("hidden", Description.parse(description), susceptibility=multiplier, degree="low")
    return Scenario(name="hidden_susceptibility", plants=(plant,), **overrides)


def superspreaders(
    description: str = "occupation == Healthcare AND smoking == 1",
    shape: float = 0.2,
    **overrides,
) -> Scenario:
    """Create a group with uneven infectiousness but the same average"""
    plant = Plant("superspreaders",Description.parse(description),infectiousness_dispersion=shape)
    return Scenario(name="superspreaders", plants=(plant,), **overrides)


def harm(
    small: str = "obesity == 1 AND age >= 40",
    small_effect: float = 3.0,
    large: str = "smoking == 1 AND age < 40",
    large_effect: float = 1.3,
    trait: Literal["susceptibility", "infectiousness"] = "susceptibility",
    **overrides,
) -> Scenario:
    """a small group with a strong effect and a large group with a mild one"""
    plants = (
        Plant("small_strong", Description.parse(small), **{trait: small_effect}),
        Plant("large_mild", Description.parse(large), **{trait: large_effect}),
    )
    return Scenario(name="harm", plants=plants, **overrides)
