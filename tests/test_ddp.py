"""Tests for Distributional Dynamic Programming algo."""

import unittest
import itertools
from typing import Tuple, List, Optional, Type
from src.drl_primitives import (
    Action,
    PPComponent,
    ProjectionParameter,
    ReturnDistributionFunction,
    RewardDistributionCollection,
    State,
    MDP
)

from src.projections import (
    algo_size_fun,
    QuantileProjection,
    quant_projection_algo,
)

from src.ddp import ddp
from src.sample_envs import cyclical_env
import logging

DEBUG: bool = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestDDP(unittest.TestCase):

    def setUp(self):
        self.inner_index_set: List[Tuple[State, Action, State]] = \
            list(
                itertools.product(
                    cyclical_env.mdp.states,
                    cyclical_env.mdp.actions,
                    cyclical_env.mdp.states
                )
            )
        self.outer_index_set: List[State] = list(cyclical_env.mdp.states)

    def test_ddp_one_step(self):

        ret_dist_est: ReturnDistributionFunction = \
            ddp(cyclical_env.mdp,
                QuantileProjection,
                QuantileProjection,
                quant_projection_algo,  # expect output sizes (1,1)
                cyclical_env.return_distr_fun_est,
                reward_distr_coll=None,
                iteration_num=1
                )
        if DEBUG:
            logger.info("DDP one step completed.")

        self.assertTrue(
            (ret_dist_est[cyclical_env.mdp.states[0]].size == 1) and
            (ret_dist_est[cyclical_env.mdp.states[1]].size == 1) and
            (ret_dist_est[cyclical_env.mdp.states[2]].size == 1)
        )

    def test_ddp_one_step_another_size_fun(self):

        def in_size_fun(x: int) -> PPComponent:
            return x**2

        def out_size_fun(x: int) -> PPComponent:
            return x**3

        def quant_proj_2(
            iteration_num: int,
            ret_distr_fun: ReturnDistributionFunction,
            rew_distr_coll: Optional[RewardDistributionCollection],
            mdp: MDP,
            inner_index_set: List[Tuple[State, Action, State]],
            outer_index_set: List[State]) \
                -> Tuple[ProjectionParameter, ProjectionParameter]:
            return algo_size_fun(
                iteration_num,
                inner_index_set=self.inner_index_set,
                outer_index_set=self.outer_index_set,
                inner_size_fun=in_size_fun,
                outer_size_fun=out_size_fun
            )
        # quant_proj_2: ParamAlgo = \
        #     functools.partial(
        #     algo_size_fun,
        #     inner_index_set=self.inner_index_set,
        #     outer_index_set=self.outer_index_set,
        #     inner_size_fun=in_size_fun,
        #     outer_size_fun=out_size_fun
        # )
        ret_dist_est: ReturnDistributionFunction = \
            ddp(
                cyclical_env.mdp,
                QuantileProjection(),
                QuantileProjection(),
                quant_proj_2,
                cyclical_env.return_distr_fun_est,
                reward_distr_coll=None,
                iteration_num=2
            )

        self.assertTrue(
            (ret_dist_est[cyclical_env.mdp.states[0]].size == 8) and
            (ret_dist_est[cyclical_env.mdp.states[1]].size == 8) and
            (ret_dist_est[cyclical_env.mdp.states[2]].size == 8)
        )
