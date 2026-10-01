import numpy as np
import pytest
from sir_graph.parameters import NodeParams, transmission_probability

from sir_graph.states import State, choose_seeds

# states


def test_state_values_are_count_columns() -> None:
    assert (State.S, State.I, State.R) == (0, 1, 2)


def test_choose_seeds_distinct_sorted_in_range() -> None:
    seeds = choose_seeds(100, 10, rng=0)
    assert seeds.dtype == np.int64
    assert len(np.unique(seeds)) == 10
    assert np.all(np.diff(seeds) > 0)
    assert np.all((seeds >= 0) & (seeds < 100))


def test_choose_seeds_reproducible() -> None:
    assert np.array_equal(choose_seeds(100, 5, rng=3), choose_seeds(100, 5, rng=3))


def test_choose_seeds_respects_eligible() -> None:
    eligible = np.zeros(50, dtype=bool)
    eligible[[4, 9, 30]] = True
    seeds = choose_seeds(50, 3, rng=0, eligible=eligible)
    assert set(seeds.tolist()) == {4, 9, 30}


@pytest.mark.parametrize(("n", "k"), [(5, 6), (5, -1)])
def test_choose_seeds_invalid_count(n: int, k: int) -> None:
    with pytest.raises(ValueError):
        choose_seeds(n, k, rng=0)


def test_choose_seeds_bad_mask_shape() -> None:
    with pytest.raises(ValueError):
        choose_seeds(5, 1, rng=0, eligible=np.ones(4, dtype=bool))


# parameters


def test_homogeneous() -> None:
    params = NodeParams.homogeneous(4, recovery=0.3)
    assert params.n_nodes == 4
    assert np.all(params.susceptibility == 1.0)
    assert np.all(params.infectiousness == 1.0)
    assert np.all(params.recovery == 0.3)


def test_arrays_are_read_only_copies() -> None:
    s = np.ones(3)
    params = NodeParams(s, np.ones(3), np.ones(3))
    s[0] = 99.0  # mutating the input must not leak in
    assert params.susceptibility[0] == 1.0
    with pytest.raises(ValueError):
        params.susceptibility[0] = 2.0


@pytest.mark.parametrize(
    ("s", "f", "q"),
    [
        ([1.0, -1.0], [1.0, 1.0], [1.0, 1.0]),  # negative susceptibility
        ([1.0, 1.0], [1.0, np.inf], [1.0, 1.0]),  # non-finite infectiousness
        ([1.0, 1.0], [1.0, 1.0], [1.0, 0.0]),  # q = 0: never recovers
        ([1.0, 1.0], [1.0, 1.0], [1.0, 1.5]),  # q > 1
        ([1.0, 1.0], [1.0], [1.0, 1.0]),  # length mismatch
    ],
)
def test_validation(s: list[float], f: list[float], q: list[float]) -> None:
    with pytest.raises(ValueError):
        NodeParams(np.array(s), np.array(f), np.array(q))


def test_scaled_only_changes_masked_nodes() -> None:
    base = NodeParams.homogeneous(4, recovery=0.5)
    mask = np.array([True, False, True, False])
    planted = base.scaled(mask, susceptibility=3.0, infectiousness=0.5, recovery=0.2)
    assert np.array_equal(planted.susceptibility, [3.0, 1.0, 3.0, 1.0])
    assert np.array_equal(planted.infectiousness, [0.5, 1.0, 0.5, 1.0])
    assert np.array_equal(planted.recovery, [0.2, 0.5, 0.2, 0.5])
    assert np.all(base.susceptibility == 1.0)  # original untouched


def test_scaled_rejects_bad_mask() -> None:
    with pytest.raises(ValueError):
        NodeParams.homogeneous(4).scaled(np.array([True, False]))


def test_transmission_probability_product_and_clipping() -> None:
    params = NodeParams(
        susceptibility=np.array([1.0, 2.0, 10.0]),
        infectiousness=np.array([0.5, 1.0, 1.0]),
        recovery=np.ones(3),
    )
    src = np.array([0, 1, 1])
    dst = np.array([1, 0, 2])
    prob = transmission_probability(0.4, params, src, dst)
    assert np.allclose(prob, [0.4, 0.4, 1.0])
