"""Preliminary tests for master thesis."""

# import numpy as np


import random
from typing import Callable, Dict, List, Sequence, Tuple, Collection

import numpy as np
import scipy.stats as sp
from numba import njit, jit
import matplotlib.pyplot as plt
import time


STATES = {1, 2}
ACTIONS = {1, 2, 3, 4}
DEBUG = True
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


def conv(
    a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    0th entry values, 1st entry probabilities.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs



def time_it(debug: bool) -> Callable:
    """Time function.

    Print time for func call if debug is True.
    """

    def time_it_dec(func: Callable) -> Callable:
        if debug:
        
            def time_it_inner(*args, **kwargs): 
                start: float = time.time()
                ret_val = func(*args, **kwargs)
                end: float = time.time()
                if "time_step" in **kwargs:
                    print(f"Time taken for time step {kwargs['time_step']}: {end-start}")
                else:
                    print(f"Time taken: {end-start}")
                return ret_val
            return time_it_inner

        else: return func

    return time_it_dec


@njit
def conv_jit(
    a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    Numba JIT compiled version.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


@njit
def aggregate_conv_results(distr: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate results of convolution.

    Sum up probabilities of same values.
    """


    val_sorted_indices: np.ndarray = np.argsort(distr[0])  # n log n
    val_sorted: np.ndarray = distr[0][val_sorted_indices]
    probs_sorted: np.ndarray = distr[1][val_sorted_indices]

    ret_dist_v: List = []
    ret_dist_p: List = []
    current: int = 0
    i: int = 1
    ret_dist_v.append(val_sorted[current])
    ret_dist_p.append(probs_sorted[current])



    for i in range(1, distr[0].size):
        if np.abs(val_sorted[i] - val_sorted[i - 1]) < 1e-10:
            probs_sorted[current] += probs_sorted[i]
        else:
            ret_dist_v.append(val_sorted[i])
            ret_dist_p.append(probs_sorted[i])
            current = i
        
        
    # values: np.ndarray = np.unique(distr[0])
    # probs: np.ndarray = np.zeros(values.size)

    # for i, val in enumerate(values):
        # probs[i] = np.sum(distr[1][distr[0] == val])
    # return values, probs

    return np.asarray(ret_dist_v), np.asarray(ret_dist_p)


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

        # aufsummierren, durh  Summe  teilen

        return bin_values, new_probs

    return apply_projection_inner


@apply_projection
@njit
def project_eqi(
    values: np.ndarray, probs: np.ndarray, no_of_bins: int, state: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Project equisdistantly. Return bins."""
    v_min: np.float64
    v_max: np.float64
    v_min, v_max = np.min(values), np.max(values)
    breadth: np.float64 = v_max - v_min
    breadth /= no_of_bins

    bin_values: np.ndarray = np.linspace(v_min-breadth, v_max+breadth, no_of_bins)
    assert bin_values.size == no_of_bins, "wrong number of bins."
    bins: np.ndarray = np.digitize(values, bin_values)
    return bins, bin_values, no_of_bins

def simulate_update(
        time_steps: int, num_samples: int,
        reward_distr: Tuple[np.ndarray, np.ndarray],
        proj_func: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate update of aggregate return distribution over time_steps periods.

    Return aggregate return distribution.
    """

    approx_list: List = []
    # start with some random intial aggregate return distribution
    num_elements_g0: int = np.random.randint(1, 100)
    rand_values: np.ndarray = np.random.random(num_elements_g0)

    g_0: Tuple[np.ndarray, np.ndarray] = (rand_values, rand_values / np.sum(rand_values))
    print(f"g_0: {g_0}")
    approx_list.append(g_0)


    for t in range(time_steps):
        # g_t: Tuple[np.ndarray, np.ndarray] = conv_jit(emp_distr, approx_list[-1])
        # g_t = proj_func(values=g_t[0], probs=g_t[1], no_of_bins=(t + 1) * 10, state=1)
        # approx_list.append(g_t)
        g_t: Tuple[np.ndarray, np.ndarray] = \
            simulate_one_step(reward_distr, approx_list[-1],
                              time_step=t, proj_func=proj_func)

        approx_list.append(g_t)

    return approx_list[-1]


@time_it(DEBUG)
def simulate_one_step(dist1: Tuple[np.ndarray, np.ndarray],
                      dist2: Tuple[np.ndarray, np.ndarray],
                      time_step: int, proj_func: Callable) -> Tuple[np.ndarray, np.ndarray]:

    g_t: Tuple[np.ndarray, np.ndarray] = aggregate_conv_results(conv_jit(dist1, dist2))
    g_t = proj_func(values=g_t[0], probs=g_t[1], no_of_bins=(time_step + 1) * 10, state=1)
    return g_t


def plot_atomic_distr(distr: Tuple[np.ndarray, np.ndarray]) -> None:
    """Plot atomic distribution."""
    num_atom: int = distr[0].size
    x_min: np.float64 = np.min(distr[0])
    x_max: np.float64 = np.max(distr[0])

    # bins: np.ndarray = np.digitize(distr[0], np.linspace(x_min, x_max, num_atom // 5))
    new_vals: np.ndarray
    new_probs: np.ndarray
    new_vals, new_probs = project_eqi(values=distr[0], probs=distr[1], no_of_bins=num_atom// 40, state=1)

    # print(np.sum(new_probs))
    plt.bar(new_vals, new_probs)
    # plt.show()
    return None
  

def main():
    """Call main function."""
    # using controlled experiment from paper
    # rt = sp.norm(loc=0, scale=1)
    S: Collection= {1,2,3}
    A: Collection = {1}
    N: int = 100
    T: int = 10

    Rewards: Dict = {
        (1, 2) : sp.norm(loc=-3, scale=1).rvs(N),
        (2, 3) : sp.norm(loc=5, scale=2).rvs(N),
        (3, 1) : sp.norm(loc=0, scale=0.5).rvs(N),
    }
    gamma: np.float64 = 0.7
    initial_return: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        1: (Rewards[], np.ones(N) / N),
        


    }

    # policy
    def pi(s: int, a: int, s_prime: int) -> np.float64:
        match (s, a, s_prime):
            case (1, 1, 2): return 1.0
            case (2, 1, 3): return 1.0
            case (3, 1, 1): return 1.0
            case _: return 0.0


    # re
    for i in range(T):
        for s in S:
            for s_prime in S:
                # sample from return distribution
                # update return distribution
                # update approx return distribution
                pass


            


    


if __name__ == "__main__":
    a_val = np.array([1, 2, 3])
    a_probs = np.array([0.3, 0.4, 0.3])
    b_val = np.array([-5, 5])
    b_probs = np.array([0.5, 0.5])

    a = (a_val, a_probs)
    b = (b_val, b_probs)

    c = conv(a, b)
    vals = np.linspace(0, 100, 1000)
    ps = np.ones(1000) * 1 / 1000
    # apply_projection(project_eqi)(values=vals, probs=ps, no_of_bins=10, state=1)
    new_vals, new_probs = project_eqi(values=vals, probs=ps, no_of_bins=10, state=1)

    print(new_vals, new_probs)
    print(np.sum(new_probs))
    print("completed binning.")
    # main()
