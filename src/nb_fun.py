"""Numba function to be used in other py modules with custom classes"""

import numpy as np
from numba import njit, jit
from typing import Tuple

import pdb
NUM_PRECISION_DECIMALS: int = 10


@njit
def _sort_njit(xs: np.ndarray, ps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(xs)
    return xs[indices], ps[indices]


@njit(debug=True)
def _qf_njit(xs: np.ndarray, ps: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile function evaluation for array of values."""
    ret: np.ndarray = np.zeros(u.size)
    ret[np.isclose(u, 1)] = xs[-1]
    ret[np.isclose(u, 0)] = -np.inf

    # handle position where 0 < u < 1
    ret[np.invert(np.isclose(u, 1) + np.isclose(u, 0))] =\
        xs[
            np.searchsorted(
                np.round(np.cumsum(ps), NUM_PRECISION_DECIMALS),
                u[np.invert(np.isclose(u, 0) + np.isclose(u, 1))])
            ]
    return ret
