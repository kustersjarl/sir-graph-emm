import numpy as np
import pytest

from sir_graph.parameters import NodeParams, transmission_probability

#  parameters


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


def test_transmission_probability_is_exponential_in_product() -> None:
    params = NodeParams(
        susceptibility=np.array([1.0, 2.0, 10.0]),
        infectiousness=np.array([0.5, 1.0, 1.0]),
        recovery=np.ones(3),
    )
    src = np.array([0, 1, 1])
    dst = np.array([1, 0, 2])
    prob = transmission_probability(0.4, params, src, dst)
    # hazards 0.4*0.5*2 = 0.4 ; 0.4*1*1 = 0.4 ; 0.4*1*10 = 4
    assert np.allclose(prob, 1 - np.exp(-np.array([0.4, 0.4, 4.0])))


def test_transmission_probability_edge_multiplier() -> None:
    params = NodeParams.homogeneous(2)

    src = np.array(
        [0, 0],
        dtype=np.int64,
    )

    dst = np.array(
        [1, 1],
        dtype=np.int64,
    )

    multiplier = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    prob = transmission_probability(
        0.2,
        params,
        src,
        dst,
        edge_multiplier=multiplier,
    )

    assert np.allclose(
        prob,
        1 - np.exp(-np.array([0.2, 0.4])),
    )