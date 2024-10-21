"""Test all kinds of projections with discrete as well as continuous
random variables.
"""

import unittest
import logging
import itertools
from typing import List, Tuple, Type

import numpy as np
import scipy.stats as sp

from src.random_variables import ContinuousRV, DiscreteRV
from src.drl_primitives import State, Action, ReturnDistributionFunction
from src.projections import (
    PPComponent,
    QuantileProjection,
    RandomProjection,
    ProjectionParameter,
    GridValueProjection
)

DEBUG: bool = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestQuantileProjection(unittest.TestCase):
    """TODO maybe get rid of Projection Parameter.
    May circumvent its usage in these tests.
    """

    def setUp(self):
        self.states = [State(0, "{0}", 0)]
        self.actions: List[Action] = [Action(0, "0", 0)]
        self.inner_index_set: List[Tuple[State, Action, State]] = list(
            itertools.product(self.states, self.actions, self.states)
        )
        self.outer_index_set: List[State] = self.states

        self.inner_proj: Type[QuantileProjection] = QuantileProjection
        self.outer_proj: Type[QuantileProjection] = QuantileProjection
        rv1_xk_discrete = np.arange(1, 11)
        pk = np.ones(10) / 10

        self.return_distr_fun_est_discrete: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [DiscreteRV(xk=rv1_xk_discrete, pk=pk)]  # type: ignore
            )
        self.return_distr_fun_est_cont: ReturnDistributionFunction = \
            ReturnDistributionFunction(
                self.states,
                [ContinuousRV(sp.norm(loc=0, scale=1))]  # type: ignore
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

    def test_quantile_projection(self):
        logger.info("Test quantile projection")
        inner_param: ProjectionParameter
        outer_param: ProjectionParameter
        inner_param, outer_param = self.param_algo(2)

        rv1_proj = self.inner_proj.project(
            self.return_distr_fun_est_discrete[self.states[0]], inner_param[
                (self.states[0], self.actions[0], self.states[0])
            ]
        )
        rv2_proj = self.inner_proj.project(
            self.return_distr_fun_est_cont[self.states[0]],
            inner_param[(
                self.states[0],
                self.actions[0],
                self.states[0])
            ]
        )
        rv3_proj = self.outer_proj.project(self.return_distr_fun_est_discrete[
                                           self.states[0]],
                                           outer_param[self.states[0]])

        rv4_proj = self.inner_proj.project(ContinuousRV(sp.uniform(0, 1)),
                                           inner_param[
                                               (self.states[0],
                                                self.actions[0],
                                                self.states[0])
                                               ]
                                           )

        if DEBUG:
            logger.info(f"Inner projection output size: {rv1_proj.size}")
            logger.info(f"Inner projection output size: {rv2_proj.size}")
            logger.info(f"Inner projection output size: {rv4_proj.size}")
            logger.info(f"Outer projection output size: {rv3_proj.size}")
            logger.info(f"Projcection results of unif[0,1] onto 4 quantiles: \
                {rv4_proj.xk, rv4_proj.pk}")

        self.assertTrue(rv1_proj.size == 4)
        self.assertTrue(rv2_proj.size == 4)
        self.assertTrue(rv3_proj.size == 8)
        self.assertTrue((rv4_proj.xk == np.array(
            [(1/8), (3/8), (5/8), (7/8)])).all())
        self.assertTrue((rv4_proj.pk == np.ones(4) / 4).all())


class TestRandomProjection(unittest.TestCase):

    def setUp(self):
        self.states = [State(0, "{0}", 0)]
        self.actions: List[Action] = [Action(0, "0", 0)]
        self.inner_index_set: List[Tuple[State, Action, State]] = list(
            itertools.product(self.states, self.actions, self.states)
        )
        self.outer_index_set: List[State] = self.states

        self.inner_proj: Type[RandomProjection] = RandomProjection
        self.outer_proj: Type[RandomProjection] = RandomProjection
        rv1_xk_discrete: np.ndarray = np.arange(1, 11)
        pk: np.ndarray = np.ones(10) / 10
        self.rv1_discrete = DiscreteRV(xk=rv1_xk_discrete, pk=pk)
        self.rv2_cont = ContinuousRV(sp.norm(loc=0, scale=1))

    def test_random_projection(self):
        logger.info("Test random projection")
        inner_ppcom: PPComponent = 5
        outer_ppcom: PPComponent = 25

        rv1_proj_inner: DiscreteRV = self.inner_proj.project(
            self.rv1_discrete, inner_ppcom)
        rv1_proj_outer: DiscreteRV = self.outer_proj.project(
            self.rv1_discrete, outer_ppcom)
        rv2_proj_inner: DiscreteRV = self.inner_proj.project(
            self.rv2_cont, inner_ppcom)
        rv2_proj_outer: DiscreteRV = self.outer_proj.project(
            self.rv2_cont, outer_ppcom)

        self.assertIsNotNone(rv1_proj_inner, "RV1 disc inner proj is None")
        self.assertIsNotNone(rv1_proj_outer, "RV1 disc outer proj is None")
        self.assertIsNotNone(rv2_proj_inner, "RV2 cont inner proj is None")
        self.assertIsNotNone(rv2_proj_outer, "RV2 cont outer proj is None")


class TestGridValueProjection(unittest.TestCase):
    def setUp(self):
        """Set up the test environment.

        rv1: Unif([0.1, 1]) discrete with steps of width 0.1.
        grid values:
            ys: 0.1 0.2 0.3 0.4 0.9
            xs: 0.15 0.25 0.35 0.7
        """

        rv1_xk_discrete: np.ndarray = np.linspace(0.1, 1, 10)
        pk: np.ndarray = np.ones(10) / 10
        self.rv1_discrete: DiscreteRV = DiscreteRV(xk=rv1_xk_discrete, pk=pk)
        self.rv2_cont: ContinuousRV = ContinuousRV(sp.uniform(0, 1))

        # TODO deal with numerical inaccuracies leading to CDF(0.7)
        # eval being wrong in discrete rv test case
        self.grid_values: np.ndarray = \
            np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.15, 0.25, 0.35, 0.7])
        self.grid_value_proj: Type[GridValueProjection] = GridValueProjection

    def test_grid_value_projection(self):
        """Test the grid value projection."""
        logger.info("Test grid value projection")
        rv1_proj: DiscreteRV = self.grid_value_proj.project(self.rv1_discrete,
                                                            self.grid_values)
        rv2_proj: DiscreteRV = self.grid_value_proj.project(self.rv2_cont,
                                                            self.grid_values)

        exp_probs_cont: np.ndarray = np.asarray([0.15, 0.1, 0.1, 0.35, 0.3])
        exp_probs_disc: np.ndarray = np.asarray([0.1, 0.1, 0.1, 0.4, 0.3])

        self.assertTrue(np.all(np.isclose(rv1_proj.xk, self.grid_values[:5])),
                        f"rv1_proj.xk = {rv1_proj.xk} \n \
                        Expected: {self.grid_values[:5]}")

        self.assertTrue(np.all(np.isclose(rv1_proj.pk, exp_probs_disc)),
                        f"rv1_proj.pk = {rv1_proj.pk} \n \
                        Expected: {exp_probs_disc}")

        self.assertTrue(np.all(np.isclose(rv2_proj.xk, self.grid_values[:5])),
                        f"rv2_proj.xk = {rv2_proj.xk} \n \
                        Expected: {self.grid_values[:5]}")

        self.assertTrue(np.all(np.isclose(rv2_proj.pk, exp_probs_cont)),
                        f"rv2_proj.pk = {rv2_proj.pk} \n \
                        Expected: {exp_probs_cont}")
