"""Tests for Distributional Dynamic Programming algo."""

import unittest
import functools
import itertools
from typing import Tuple, Callable, List
import numpy as np
from src.preliminary_tests import (
    RV,
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


class TestQuantileProjection(unittest.TestCase):

    def setUp(self):
        self.states = [State(i, "{i}", i) for i in range(3)]
        self.actions: List[Action] = [Action(1, "1", 1)]
        self.inner_index_set: List[Tuple[State, Action, State]] = list(
            itertools.product(self.states, self.actions, self.states)
        )
        self.outer_index_set: List[State] = self.states

        self.inner_proj: QuantileProjection = QuantileProjection()
        self.outer_proj: QuantileProjection = QuantileProjection()
        rv1_xk = np.arange(1, 11)
        rv2_xk = np.arange(5, 15)
        rv3_xk = np.arange(11, 21)
        pk = np.ones(10) / 10

        self.return_distr_fun_est: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [RV(xk=x, pk=pk) for x in [rv1_xk, rv2_xk, rv3_xk]]
            )

    def param_algo(self, x: int) -> \
            Tuple[ProjectionParameter, ProjectionParameter]:
        ipp: ProjectionParameter = ProjectionParameter(
            {(s, a, s_): x**2 for s, a, s_ in self.inner_index_set}
        )
        opp: ProjectionParameter = ProjectionParameter(
            {s: x**3 for s in self.outer_index_set}
        )
        return ipp, opp

    def test_projections(self):
        inner_param: ProjectionParameter
        outer_param: ProjectionParameter
        inner_param, outer_param = self.param_algo(2)

        rv1_proj = self.inner_proj(
            self.return_distr_fun_est[self.states[0]], inner_param[
                (self.states[0], self.actions[0], self.states[0])
                ]
        )
        rv2_proj = self.inner_proj(self.return_distr_fun_est[self.states[1]],
                                   inner_param[
                                       self.states[0],
                                       self.actions[0],
                                       self.states[0]
                                   ]
                                   )
        rv3_proj = self.outer_proj(self.return_distr_fun_est[self.states[2]],
                                   outer_param[self.states[0]])

        if DEBUG:
            logger.info(f"Inner projection output size: {rv1_proj.size}")
            logger.info(f"Outer projection output size: {rv3_proj.size}")

        self.assertTrue(rv1_proj.size == 4)
        self.assertTrue(rv2_proj.size == 4)
        self.assertTrue(rv3_proj.size == 8)


class TestRandomProjection(unittest.TestCase):

    def setUp(self):
        self.states = [State(i, "{i}", i) for i in range(3)]
        rv1_xk = np.arange(1, 11)
        rv2_xk = np.arange(5, 15)
        rv3_xk = np.arange(11, 21)
        pk = np.ones(10) / 10
        self.return_distr_fun_est: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [RV(xk=x, pk=pk) for x in [rv1_xk, rv2_xk, rv3_xk]]
            )


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
