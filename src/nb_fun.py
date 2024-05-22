"""Numba function to be used in other py modules with custom classes"""

import numpy as np
from numba import njit
from typing import Tuple

@njit
def _sort_njit(xs: np.ndarray, ps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(xs)
    return xs[indices], ps[indices]
