"""Numba function to be used in other py modules with custom classes"""

import numpy as np
from numba import njit, jit
from typing import Tuple, List

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


@njit
def aggregate_conv_results(distr: Tuple[np.ndarray, np.ndarray], accuracy: float=1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate results of convolution.
    Sum up probabilities of same values.
    """

    val_sorted_indices: np.ndarray = np.argsort(distr[0])  # n log n
    val_sorted: np.ndarray = distr[0][val_sorted_indices]
    probs_sorted: np.ndarray = distr[1][val_sorted_indices]

    ret_dist_v: List = []
    ret_dist_p: List = []
    current: int = 0
    # i: int = 1
    # ret_dist_v.append(val_sorted[current])
    # ret_dist_p.append(probs_sorted[current])

    for i in range(1, distr[0].size):
        # if np.abs(val_sorted[i] - val_sorted[i - 1]) < accuracy:
        if np.abs(val_sorted[i] - val_sorted[current]) < accuracy:
            probs_sorted[current] += probs_sorted[i]
        else:
            ret_dist_v.append(val_sorted[current])
            ret_dist_p.append(probs_sorted[current])
            current = i

            # ret_dist_v.append(val_sorted[i])
            # ret_dist_p.append(probs_sorted[i])
            # current = i

    ret_dist_v.append(val_sorted[current])
    ret_dist_p.append(probs_sorted[current])

    return np.asarray(ret_dist_v), np.asarray(ret_dist_p)


@njit
def conv_njit(
        a: Tuple[np.ndarray, np.ndarray],
        b: Tuple[np.ndarray, np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    Numba JIT compiled version.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


@njit
def cdf_njit(xk: np.ndarray, pk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Assume that xk is sorted (ascending order).
    """
    ret: np.ndarray = np.zeros(x.size)
    for i in range(x.size):
        ret[i] = np.sum(pk[xk <= x[i]])
    return ret
