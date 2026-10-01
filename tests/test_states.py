import numpy as np
import pytest

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
