from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NodeAttributes:
    person_id: NDArray[np.int64]
    age: NDArray[np.int64]
    sex: NDArray[np.str_]
    occupation: NDArray[np.str_]
    smoking: NDArray[np.bool_]
    alcohol: NDArray[np.bool_]
    obesity: NDArray[np.bool_]

    @property
    def n_nodes(self) -> int:
        return len(self.person_id)