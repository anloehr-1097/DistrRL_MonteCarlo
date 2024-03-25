"""Preliminary tests for master thesis."""

# import numpy as np


import scipy.stats as sp
import numpy as np
from typing import Tuple, Dict, Sequence, Callable, List
import random
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
        self.xk: np.ndarray = xk
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
    def apply_projection_inner(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        bins: np.ndarray
        no_of_bins: int
        bin_values: np.ndarray

        bins, bin_values, no_of_bins = func(*args, **kwargs)
        # do binning, stupidly
        # new_values: np.ndarray = np.zeros(no_of_bins)
        new_probs: np.ndarray = np.zeros(no_of_bins)

        for i in range(no_of_bins):
            new_probs[i] = np.sum(kwargs["probs"][bins == i])

        # TODO: ensure that probs sum to 1, other behviour
        rem: np.float64 = 1 - np.sum(new_probs)
        new_probs[random.randint(0, no_of_bins - 1)] += rem

        return bin_values, new_probs


    return apply_projection_inner


@apply_projection
@njit
def project_eqi(values: np.ndarray, probs: np.ndarray, no_of_bins: int,
                state: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Project equisdistantly. Return bins."""
    v_min: np.float64
    v_max: np.float64
    v_min, v_max = np.min(values), np.max(values)

    bin_values: np.ndarray = np.linspace(v_min, v_max, no_of_bins)
    assert bin_values.size == no_of_bins, "wrong number of bins."
    bins: np.ndarray = np.digitize(values, bin_values)
    return bins, bin_values, no_of_bins


@njit
@apply_projection
def project(values: np.ndarray, probs: np.ndarray, iteration: int,
            bin_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    # TODO delete this
    """General projection function."""
    v_min, v_max = np.max(values), np.min(values)

    bins: np.ndarray = bin_func(values, probs, iteration)
    return bins, bins


    proj_values: np.ndarray = np.linspace(v_min, v_max, iteration)
    proj_probs: np.ndarray = np.zeros(iteration)
    return proj_values, proj_probs


def simulate_update(time_steps: int, num_samples: int, return_distr) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Simulate update of return distribution."""
    samples: np.ndarray = np.asarray([return_distr.rvs()
                                      for _ in range(num_samples)])
    approx_list: List = []
    g_0: Tuple[np.ndarray, np.ndarray] = \
        (np.random.random(2), np.array([0.5, 0.5]))

    emp_distr: Tuple[np.ndarray, np.ndarray] = (samples,
                                                np.ones(num_samples) / num_samples)
    for t in range(time_steps):
        g_t: Tuple[np.ndarray, np.ndarray] = conv_jit(emp_distr, approx_list[-1])
        g_t = project_eqi(values=g_t[0], probs=g_t[1], no_of_bins=(t+1)*10, state=1)
        approx_list.append(g_t)

    return approx_list[-1]


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
    vals = np.linspace(0, 100, 1000)
    ps = np.ones(1000) * 1/1000
    # apply_projection(project_eqi)(values=vals, probs=ps, no_of_bins=10, state=1)
    new_vals, new_probs = project_eqi(values=vals, probs=ps, no_of_bins=10, state=1)

    print(new_vals, new_probs)
    print(np.sum(new_probs))
    print("completed binning.")
    # Main()
