"""Test all kinds of projections with discrete as well as continuous
random variables.
"""

import unittest
import logging
import itertools
from typing import List, Tuple

import numpy as np
import scipy.stats as sp

from src.preliminary_tests import (
    ContinuousRV,
    State,
    Action,
    QuantileProjection,
    RandomProjection,
    RV,
    ReturnDistributionFunction,
    ProjectionParameter,
)

DEBUG: bool = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestQuantileProjection(unittest.TestCase):

    def setUp(self):
        self.states = [State(0, "{0}", 0)]
        self.actions: List[Action] = [Action(0, "0", 0)]
        self.inner_index_set: List[Tuple[State, Action, State]] = list(
            itertools.product(self.states, self.actions, self.states)
        )
        self.outer_index_set: List[State] = self.states

        self.inner_proj: QuantileProjection = QuantileProjection()
        self.outer_proj: QuantileProjection = QuantileProjection()
        rv1_xk_discrete = np.arange(1, 11)
        pk = np.ones(10) / 10

        self.return_distr_fun_est_discrete: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [RV(xk=rv1_xk_discrete, pk=pk)]
            )
        self.return_distr_fun_est_cont: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [ContinuousRV(sp.norm(loc=0, scale=1))]
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

    def test_projection(self):
        inner_param: ProjectionParameter
        outer_param: ProjectionParameter
        inner_param, outer_param = self.param_algo(2)

        rv1_proj = self.inner_proj(
            self.return_distr_fun_est_discrete[self.states[0]], inner_param[
                (self.states[0], self.actions[0], self.states[0])
                ]
            )
        rv2_proj = self.inner_proj(self.return_distr_fun_est_cont[self.states[0]],
                                   inner_param[
                                       self.states[0],
                                       self.actions[0],
                                       self.states[0]
                                       ]
                                   )
        rv3_proj = self.outer_proj(self.return_distr_fun_est_discrete[self.states[0]],
                                   outer_param[self.states[0]])

        if DEBUG:
            logger.info(f"Inner projection output size: {rv1_proj.size}")
            logger.info(f"Outer projection output size: {rv3_proj.size}")

        self.assertTrue(rv1_proj.size == 4)
        self.assertTrue(rv2_proj.size == 4)
        self.assertTrue(rv3_proj.size == 8)


class TestRandomProjection(unittest.TestCase):
    pass


class TestGridValueProjection(unittest.TestCase):
    pass
