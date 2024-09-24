"""Tests for the random variable abstractions."""

import numpy as np
from typing import Tuple
import unittest
from src.preliminary_tests import RV
import logging

DEBUG: bool = True 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestFiniteRV(unittest.TestCase):
    def setUp(self):
        vals = np.arange(1, 11, 1)
        probs = np.ones(10) * (1/10)
        self.rv = RV(vals, probs)

        if DEBUG:
            logger.info(f"Random variable: {self.rv.distr()}")
        return None

    def test_cdf(self):
        cdf_eval = self.rv.cdf(3)
        self.assertTrue(
            np.isclose(cdf_eval, 0.3),
            f"CDF evaluation failed.\nExpected: 0.3\nGot: {cdf_eval}")

    def test_qf(self):
        qf_eval_0 = self.rv.qf(0)
        qf_eval_1 = self.rv.qf(0.35)
        qf_eval_2 = self.rv.qf(0.8)
        qf_eval_3 = self.rv.qf(1.0)
        qf_eval = np.array([qf_eval_0, qf_eval_1, qf_eval_2, qf_eval_3])
        if DEBUG: logger.info(f"Quantile function evaluation: {qf_eval}")
        expected = np.array([1, 4, 8, 10])
        
        self.assertTrue(
            np.isclose(qf_eval, expected).all(),
            f"Quantile function evaluation failed.\nExpected: {expected}\nGot: {qf_eval}")
