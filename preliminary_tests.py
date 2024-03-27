"""Preliminary tests for master thesis."""

# import numpy as np


import scipy.stats as sp
import numpy as np
from typing import Tuple, Dict, Sequence, Callable
from numba import njit


STATES = {1, 2}
ACTIONS = {1, 2, 3, 4}

# need some initial collection of distrs for the total reward =: nu^0
# need some return distributons
# need samples from return distr for each triple (s, a, s') (N * |S| * |A| * |S|) many)
# need Bellman operator
# need some policy (fixed), define state - action dynamics
# need some distributions such that everything is known and can be compared
# define random bellman operator as partial function


class MDP:
    """Markov decision process."""

    def __init__(self, states: Sequence, actions: Sequence, rewards: Dict):
        """Initialize MDP."""
        pass


class RV_Discrete:
    """Discrete atomic random variable."""

    def __init__(self, xk, pk) -> None:
        """Initialize discrete random variable."""
        self.xk: np.darray = xk
        self.pk: np.ndarray = pk


def conv(a: Tuple[np.ndarray, np.ndarray],
         b: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    0th entry values, 1st entry probabilities.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


@njit
def conv_jit(a: Tuple[np.ndarray, np.ndarray],
             b: Tuple[np.ndarray, np.ndarray]) \
             -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    Numba JIT compiled version.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


def apply_projection(func: Callable) -> Callable:
    """Apply binning as returned from func.

    Assumes that kwargs["probs"] is in kwargs.
    """
    def apply_projection_inner(func, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        bins: np.ndarray
        no_of_bins: int
        bins, no_of_bins = func(*args, **kwargs)
        # do binning, stupidly
        new_values: np.ndarray = np.zeros(no_of_bins)
        new_probs: np.ndarray = np.zeros(no_of_bins)

        for i in range(no_of_bins):
            new_values[i] = np.count_nonzero(bins == i)
            new_probs[i] = np.sum(kargs["probs"][bins == i])
        return new_values, new_probs

    return apply_projection_inner


@apply_projection
@njit
def project_eqi(values: np.ndarray, probs: np.ndarray, no_of_bins: int,
                state: int) -> Tuple[np.ndarray, int]:
    """Project equisdistantly. Return bins."""
    v_min: np.float64
    v_max: np.float64
    v_min, v_max = np.min(values), np.max(values)

    bins: np.ndarray = np.digitize(values,
                                   np.linspace(v_min, v_max, no_of_bins))
    return bins, no_of_bins


# @njit
# @apply_projection
# def project(values: np.ndarray, probs: np.ndarray, iteration: int,
            # bin_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    # """General projection function."""
    # v_min, v_max = np.max(values), np.min(values)
# 
    # bins: np.ndarray = bin_func(values, probs, iteration)
    # return bins, bins
# 
    # # TODO continue here
# 
    # proj_values: np.ndarray = np.linspace(v_min, v_max, iteration)
    # proj_probs: np.ndarray = np.zeros(iteration)
    # return proj_values, proj_probs


def main():
    """Call main function."""
    # rt = sp.norm(loc=0, scale=1)
    bernoulli_scaled = sp.rv_discrete(values=([-1, 1], [0.5, 0.5]))
    print(bernoulli_scaled.rvs(size=10))


if __name__ == "__main__":
    a_val = np.array([1, 2, 3])
    a_probs = np.array([0.3, 0.4, 0.3])
    b_val = np.array([-5, 5])
    b_probs = np.array([0.5, 0.5])

    a = (a_val, a_probs)
    b = (b_val, b_probs)

    c = conv(a, b)

    # main()
