"""Preliminary tests for master thesis.

TODO: replace every indexing scheme with numerical indexing just based on the
    pointer of the element
    inkeeping with that, hold probs as a 2d array with rows summing to one,
    which might make code faster, check that
"""
#
# from __future__ import annotations
# from collections import OrderedDict
# import logging
# import time
# from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
# from dataclasses import dataclass
# from enum import Enum
# import itertools
# import functools
# import numpy as np
# # import scipy.stats as sp
# from scipy.stats import rv_continuous
# from scipy.stats.distributions import rv_frozen
# from numba import njit
#
# from .nb_fun import _sort_njit, _qf_njit
# from .utils import assert_probs_distr
# from .random_variables import DiscreteRV, ContinuousRV, RV

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# TERMINAL: int = -1




# iteration -> evaluated size functions as parameters
# quant_projection_algo: Callable[[int, List[Tuple[State, Action, State]], List[State]], Tuple[ProjectionParameter, ProjectionParameter]] = \
#     functools.partial(
#         algo_size_fun, inner_size_fun=poly_size_fun,
#         outer_size_fun=poly_size_fun,
#         previous_return_estimate=None,
#         previous_reward_estimate=None
#     )




def categorical_projection(rv: DiscreteRV,
                           particles: np.ndarray)\
                           -> Tuple[np.ndarray, np.ndarray]:
    """Apply categorical projection as described in book to a distribution."""

    distr = rv.distr()
    # sort array
    hypo_insert_pos: np.ndarray = np.searchsorted(particles, distr[0])
    if DEBUG:
        print(f"Hypo insert pos: {hypo_insert_pos}")

    # determine closest neighbors
    left_neigh_index: np.ndarray = hypo_insert_pos - 1
    right_neigh_index: np.ndarray = hypo_insert_pos

    new_probs: np.ndarray = np.zeros(particles.size)

    assert left_neigh_index.size == right_neigh_index.size, \
        "Size mismatch in neighbors."
    # left_neigh: np.ndarray = particles[hypo_insert_pos - 1]
    # right_neigh: np.ndarray = particles[hypo_insert_pos]
    # determine weights to assign to neighbors
    # left_weight: np.ndarray = (right_neigh - distr[0]) / (right_neigh - left_neigh)
    # right_weight: np.ndarray = 1 - left_weight

    for i in range(left_neigh_index.size):
        if (left_neigh_index[i] == -1) and (distr[0][i] < particles[0]):
            # to the left of first particle -> left particle gets all of the mass
            new_probs[0] += distr[1][i]

        elif (right_neigh_index[i] == particles.size) and (distr[0][i] > particles[-1]):
            # to the right of last particle -> right particle gets all of the mass
            new_probs[-1] += distr[1][i]

        else:
            # assign mass to neighbors according to distance to neighbors
            left_mass_i: np.float64 = \
                1 - np.abs(particles[left_neigh_index[i]] - distr[0][i])

            right_mass_i: np.float64 = 1 - left_mass_i
            new_probs[left_neigh_index[i]] += left_mass_i * distr[1][i]
            new_probs[right_neigh_index[i]] += right_mass_i * distr[1][i]

    if DEBUG:
        print(f"Particles: {particles}")
        print(f"Probs: {new_probs}")
    return (particles, new_probs)




# TODO: possible enable @njit
# @njit
# def quantile_projection(rv: RV,
#                         no_quantiles: int) -> RV:
#     """Apply quantile projection as described in book to a distribution.
#
#     Assume that atoms in distribution are sorted in ascending order.
#     Assume that unique values in RV distribution, i.e. equal values are aggregated.
#     """
#     vals: np.ndarray = rv.xk
#     probs: np.ndarray = rv.pk
#
#     if probs.size < no_quantiles: return rv
#
#     # aggregate probs
#     cum_probs: np.ndarray = np.cumsum(probs)
#     locs: np.ndarray = (2 * (np.arange(1, no_quantiles + 1)) - 1) / (2 * no_quantiles)
#     quantiles_at_locs: np.ndarray = np.searchsorted(locs, cum_probs) + 1  # TODO double check this
#     values_at_quantiles: np.ndarray = vals[quantiles_at_locs]
#     # quantiles: np.ndarray = np.cumsum(np.ones(no_quantiles) / no_quantiles)
#     # assert np.isclose(quantiles[-1], 1), "Quantiles do not sum to 1."
#     # # determine indices for quantiles
#     # quantile_locs: np.ndarray = np.searchsorted(cum_probs, quantiles)
#     # quantile_locs = np.clip(quantile_locs, 0, len(vals) - 1)
#     # return quantile_locs, (np.ones(no_of_bins) / no_of_bins)
#     # return vals[quantile_locs], (np.ones(no_of_bins) / no_of_bins)
#     return RV(values_at_quantiles, np.ones(no_quantiles) / no_quantiles)
#


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
                if "time_step" in dict(**kwargs).keys():
                    print(
                        f"Time taken for time step {kwargs['time_step']}: {end-start}"
                    )
                else:
                    print(f"Time taken: {end-start}")
                return ret_val

            return time_it_inner

        else:
            return func

    return time_it_dec


def main():
    """Call main function."""
    from .sample_envs import cyclical_env
    # mdp = cyclical_env.mdp
    # policy = mdp.generate_random_policy()
    ddp(
        cyclical_env.mdp,
        QuantileProjection(),
        QuantileProjection(),
        quant_projection_algo,
        cyclical_env.return_distr_fun_est,
        2)
    return None



if __name__ == "__main__":

    main()
