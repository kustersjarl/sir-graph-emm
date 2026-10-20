import numpy as np
import pytest

from sir_graph.analysis import (
    ModelAObservations,
)
from sir_graph.emm import (
    score_model_a_subgroup,
)


def test_score_model_a_subgroup() -> None:
    # Person 0:
    #
    # escape = 1
    # infection exposure = 1
    #
    # lambda_G = log(2)
    #
    # Person 1:
    #
    # escape = 2
    # infection exposure = 1
    #
    # lambda_rest = log(1.5)

    observations = ModelAObservations(
        run_index=np.array(
            [
                0,
                0,
            ],
            dtype=np.int64,
        ),
        node_index=np.array(
            [
                0,
                1,
            ],
            dtype=np.int64,
        ),
        infected=np.array(
            [
                True,
                True,
            ],
            dtype=np.bool_,
        ),
        raw_escape_exposure=np.array(
            [
                1,
                2,
            ],
            dtype=np.int64,
        ),
        raw_infection_exposure=np.array(
            [
                1,
                1,
            ],
            dtype=np.int64,
        ),
        weighted_escape_exposure=np.array(
            [
                1.0,
                2.0,
            ],
            dtype=np.float64,
        ),
        weighted_infection_exposure=np.array(
            [
                1.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    )

    mask = np.array(
        [
            True,
            False,
        ],
        dtype=np.bool_,
    )

    score = score_model_a_subgroup(
        mask,
        observations,
        min_size=1,
        direction="high",
        exposure_type="weighted",
    )

    assert score is not None

    expected_group_hazard = (
        np.log(
            2.0
        )
    )

    expected_rest_hazard = (
        np.log(
            1.5
        )
    )

    expected_rr = (
        expected_group_hazard
        / expected_rest_hazard
    )

    assert score[
        "group_hazard"
    ] == pytest.approx(
        expected_group_hazard
    )

    assert score[
        "rest_hazard"
    ] == pytest.approx(
        expected_rest_hazard
    )

    assert score[
        "rate_ratio"
    ] == pytest.approx(
        expected_rr
    )

    assert score[
        "log_rate_ratio"
    ] == pytest.approx(
        np.log(
            expected_rr
        )
    )

    assert score[
        "quality"
    ] == pytest.approx(
        np.sqrt(
            0.5
        )
        * np.log(
            expected_rr
        )
    )