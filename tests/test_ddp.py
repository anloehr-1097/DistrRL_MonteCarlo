"""Tests for Distributional Dynamic Programming algo."""

import unittest
import functools
from typing import Tuple, Callable
import numpy as np
from src.preliminary_tests import (
    RV,
    ProjectionParameter,
    ReturnDistributionFunction,
    State,
    RewardDistributionCollection,
    algo_size_fun,
    ddp,
    QuantileProjection,
    quant_projection_algo,
    Projection
)
from src.sample_envs import cyclical_env
import logging

DEBUG: bool = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestQuantileProjection(unittest.TestCase):

    def param_algo(self, x: int) -> Tuple[ProjectionParameter, ProjectionParameter]:
        return ProjectionParameter(x**2), ProjectionParameter(x**3)

    def setUp(self):
        self.inner_proj: QuantileProjection = QuantileProjection()
        self.outer_proj: QuantileProjection = QuantileProjection()
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

    def test_projections(self):
        inner_param: ProjectionParameter
        outer_param: ProjectionParameter
        inner_param, outer_param = self.param_algo(2)
        self.inner_proj.set_params(inner_param)
        self.outer_proj.set_params(outer_param)

        rv1_proj = self.inner_proj(self.return_distr_fun_est[self.states[0]])
        rv2_proj = self.inner_proj(self.return_distr_fun_est[self.states[1]])
        rv3_proj = self.outer_proj(self.return_distr_fun_est[self.states[2]])

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

    def test_projection(self):
        pass


class TestDDP(unittest.TestCase):

    def setUp(self):
        pass

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

        def in_size_fun(x: int) -> ProjectionParameter:
            return ProjectionParameter(x**2)

        def out_size_fun(x: int) -> ProjectionParameter:
            return ProjectionParameter(x**3)

        quant_proj_2: Callable[..., Tuple[ProjectionParameter, ProjectionParameter]] = functools.partial(
            algo_size_fun,
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
