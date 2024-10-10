import unittest
import logging
from collections import OrderedDict
from typing import Tuple
import numpy as np
import scipy.stats as sp
from src.preliminary_tests import (
    MDP,
    ContinuousRV,
    quantile_find,
    bisect,
    support_find,
    algo_cdf_2,
)
from src.sample_envs import cyclical_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCDFParamAlgos(unittest.TestCase):
    def setUp(self):
        self.unif: ContinuousRV = ContinuousRV(sp.uniform(0, 2))
        self.norm: ContinuousRV = ContinuousRV(sp.norm(loc=0, scale=20))
        self.unif_sup: Tuple[float, float] = (0, 2)
        self.norm_sup: Tuple[float, float] = (-126, 126)

    def test_bisect_unif(self):
        logger.info("Test bisection on unif[0, 2]")
        us: np.ndarray = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
        # TODO fix this type 
        qs: np.ndarray = bisect(self.unif, us, 0.0, 2.0, precision=0.001)  # type: ignore
        expected_qs: np.ndarray = np.array([0, 0.2, 0.4, 1, 1.6, 2])
        self.assertTrue(np.allclose(expected_qs, qs, atol=0.01),
                        f"Expected: {expected_qs}, Got: {qs}")

    def test_bisect_norm(self):
        logger.info("Test bisection on norm[0, 20]")
        us: np.ndarray = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
        # TODO fix this type
        qs: np.ndarray = bisect(self.norm, us, -130.0, 130.0, precision=0.001)  # type: ignore
        self.assertTrue(np.allclose(us, self.norm.cdf(qs), atol=0.01),)

    def test_support_find(self):
        logger.info("Test support find")
        sup_unif: Tuple[float, float] = support_find(self.unif, 0.001)
        sup_norm: Tuple[float, float] = support_find(self.norm, 0.001)

        self.assertTrue(np.isclose(self.norm.cdf(sup_norm[0]), 0.0, atol=0.001) and
                        np.isclose(self.norm.cdf(sup_norm[1]), 1.0, atol=0.001),
                        f"Support find failed: Expected (-126, 126), Got: {sup_norm}")

        self.assertTrue(np.isclose(self.unif.cdf(sup_unif[0]), 0.0, atol=0.001) and
                        np.isclose(self.unif.cdf(sup_unif[1]), 1.0, atol=0.001),
                        f"Support find failed: Expected (0, 2), Got: {sup_unif}")

    def test_quantile_find(self):
        logger.info("Test quantile find")
        q_table = OrderedDict({0: self.unif_sup[0], 1: self.unif_sup[1], 0.5: 1})
        for i in range(1, 3):
            q_table = quantile_find(self.unif, i, q_table)
        self.assertTrue(np.isclose(q_table[0.25], 0.5, atol=0.01) and
                        np.isclose(q_table[0.75], 1.5, atol=0.01) and
                        np.isclose(q_table[0.5], 1.0, atol=0.01) and
                        np.isclose(q_table[0.0], 0.0, atol=0.01) and
                        np.isclose(q_table[1.0], 2.0, atol=0.01),
                        "Quantile find failed")
        expected: OrderedDict = OrderedDict({0: self.unif_sup[0], 1: self.unif_sup[1], 0.5: 1, 0.25: 0.5, 0.75: 1.5})
        self.assertEqual(q_table, expected, f"Quantile find failed. Expected: {expected}, Got: {q_table}")

    def test_algo_2_cdf(self):
        mdp: MDP = cyclical_env.mdp
        # TODO 
        pass
