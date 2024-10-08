"""Tests for Distributional Dynamic Programming algo."""

import unittest
import functools
import itertools
from typing import Tuple, Callable, List
import numpy as np
from src.preliminary_tests import (
    DiscreteRV,
    Action,
    PPComponent,
    ProjectionParameter,
    ReturnDistributionFunction,
    State,
    algo_size_fun,
    ddp,
    QuantileProjection,
    quant_projection_algo,
)
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
                QuantileProjection(),
                QuantileProjection(),
                quant_projection_algo,  # expect output sizes (1,1)
                cyclical_env.return_distr_fun_est,
                1)
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

        quant_proj_2: Callable[
            ...,
            Tuple[ProjectionParameter, ProjectionParameter]] = \
            functools.partial(
            algo_size_fun,
            inner_index_set=self.inner_index_set,
            outer_index_set=self.outer_index_set,
            inner_size_fun=in_size_fun,
            outer_size_fun=out_size_fun
        )
        ret_dist_est: ReturnDistributionFunction = \
            ddp(
                cyclical_env.mdp,
                QuantileProjection(),
                QuantileProjection(),
                quant_proj_2,
                cyclical_env.return_distr_fun_est,
                iteration_num=2
            )

        self.assertTrue(
            (ret_dist_est[cyclical_env.mdp.states[0]].size == 8) and
            (ret_dist_est[cyclical_env.mdp.states[1]].size == 8) and
            (ret_dist_est[cyclical_env.mdp.states[2]].size == 8)
        )
