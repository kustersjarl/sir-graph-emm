from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NodeAttributes:
    age_group: NDArray[np.str_]
    sex: NDArray[np.str_]
    occupation: NDArray[np.str_]
    household_size: NDArray[np.int64]
    mobility: NDArray[np.str_]
    vaccinated: NDArray[np.bool_]

    @property
    def n_nodes(self) -> int:
        return len(self.age_group)

    @classmethod
    def random(
        cls,
        n_nodes: int,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> "NodeAttributes":

        rng = np.random.default_rng(rng)

        age_group = rng.choice(
            ["child", "young_adult", "adult", "elderly"],
            size=n_nodes,
            p=[0.20, 0.25, 0.40, 0.15],
        )

        sex = rng.choice(
            ["female", "male"],
            size=n_nodes,
        )

        occupation = rng.choice(
            ["student", "office", "healthcare", "service", "retired"],
            size=n_nodes,
        )

        household_size = rng.integers(
            1,
            7,
            size=n_nodes,
        ).astype(np.int64)

        mobility = rng.choice(
            ["low", "medium", "high"],
            size=n_nodes,
            p=[0.25, 0.50, 0.25],
        )

        vaccinated = rng.choice(
            [False, True],
            size=n_nodes,
            p=[0.30, 0.70],
        )

        return cls(
            age_group=age_group,
            sex=sex,
            occupation=occupation,
            household_size=household_size,
            mobility=mobility,
            vaccinated=vaccinated,
        )