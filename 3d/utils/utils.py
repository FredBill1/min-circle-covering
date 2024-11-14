from itertools import product, repeat

import numpy as np
from numpy.typing import NDArray


def unit_hypercube(dim: int) -> NDArray[np.float64]:
    return np.array(list(product(*repeat((0, 1), dim))), dtype=np.float64)
