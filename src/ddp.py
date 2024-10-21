from __future__ import annotations
import logging
from typing import List, Optional, Type
import itertools
import numpy as np
from .random_variables import DiscreteRV, scale
from .config import DEBUG
from .nb_fun import conv_njit,  aggregate_conv_results
from .drl_primitives import (
    MDP,
    ReturnDistributionFunction,
    RewardDistributionCollection,
    ParamAlgo
)
from .projections import ProjectionParameter, Projection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


###################################
# Algorithm 5.1  - classical DBO  #
###################################
def dbo(mdp: MDP, ret_distr_function: ReturnDistributionFunction,
        reward_distr_coll: Optional[RewardDistributionCollection]=None) -> None:
    """Single application of the DBO.

    If rewards_distr is None, then use the rewards from the MDP.
    This corresponds to the original DBO. If rewards_distr is given, then
    use the given reward distribution. This corresponds to the extended DBO.
    """

    reward_distr_coll = reward_distr_coll if reward_distr_coll else mdp.rewards
    if mdp.current_policy is None:
        raise ValueError("No policy set for MDP.")

    eta_next = dict()
    for state, distr in ret_distr_function.distr.items():
        new_vals: List = []
        new_probs: List = []

        for action in mdp.actions:
            for next_state in mdp.states:
                transition_prob = mdp.transition_probs[(state, action)][next_state.index]
                prob = mdp.current_policy[state][action.index] * transition_prob
                if prob == 0:
                    continue
                reward_distr = reward_distr_coll[(state, action, next_state)]
                assert isinstance(reward_distr, DiscreteRV), "For application of the DBO, rewards need to be discrete."
                if next_state.is_terminal:
                    new_vals.append(reward_distr.xk)
                    new_probs.append(reward_distr.pk * prob)
                else:
                    distr_update = conv_njit(
                        scale(ret_distr_function[next_state], mdp.gamma).distr(),
                        reward_distr.distr()
                    )
                    # scale(ret_distr_function[next_state], mdp.gamma),
                    # reward_distr)
                    new_vals.append(distr_update[0])
                    new_probs.append(distr_update[1] * prob)

        eta_next[state] = aggregate_conv_results(
            (np.concatenate(new_vals), np.concatenate(new_probs))
        )
        eta_next[state] = DiscreteRV(*eta_next[state])

    for state, distr in eta_next.items():
        ret_distr_function[state] = distr


def ddp(
    mdp: MDP, inner_projection: Type[Projection],
    outer_projection: Type[Projection],
    param_algorithm: ParamAlgo,
    return_distr_function: ReturnDistributionFunction,
    reward_distr_coll: Optional[RewardDistributionCollection],
        iteration_num: int) -> ReturnDistributionFunction:
    """1 Step of Distributional dynamic programming in iteration iteration_num.

    Carry out one step of distributional dynamic programming.
    """

    inner_params: ProjectionParameter
    outer_params: ProjectionParameter
    # apply inner projection
    inner_params, outer_params = param_algorithm(
        iteration_num,
        return_distr_function,
        # mdp.rewards,
        reward_distr_coll if reward_distr_coll else None,
        mdp,
        list(itertools.product(mdp.states, mdp.actions, mdp.states)),
        mdp.states,
    )

    if DEBUG: logger.info(f" Inner & outer params: {inner_params, outer_params}")

    rewards_distr_coll = RewardDistributionCollection(
        list(mdp.rewards.rewards.keys()),
        [inner_projection(mdp.rewards[(s, a, s_bar)], inner_params[(s, a, s_bar)]) for
         (s, a, s_bar) in
         itertools.product(mdp.states, mdp.actions, mdp.states)
         if mdp.transition_probs[(s, a)][s_bar.index] > 0
         ]
    )
    # apply step of dbo

    dbo(mdp, return_distr_function, rewards_distr_coll)
    # apply outer projection
    return_distr_iterate: ReturnDistributionFunction = \
        ReturnDistributionFunction(
            return_distr_function.states,
            [outer_projection(return_distr_function[s], outer_params[s]) for
                s in return_distr_function.states]
        )
    return return_distr_iterate
