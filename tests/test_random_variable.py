"""Tests for the random variable abstractions."""

import numpy as np
import unittest
from src.preliminary_tests import RV, aggregate_conv_results
import logging
import pdb

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

    def test_qf_single(self):
        qf_eval_0 = self.rv.qf_single(0)
        qf_eval_1 = self.rv.qf_single(0.35)
        qf_eval_2 = self.rv.qf_single(0.8)
        qf_eval_3 = self.rv.qf_single(1.0)
        qf_eval = np.array([qf_eval_0, qf_eval_1, qf_eval_2, qf_eval_3])
        if DEBUG: logger.info(f"Quantile function evaluation: {qf_eval}")
        expected = np.array([-np.inf, 4, 8, 10])
        self.assertTrue(
            np.isclose(qf_eval, expected).all(),
            f"Quantile function evaluation failed.\nExpected:\
            {expected}\nGot: {qf_eval}")

    def test_qf(self):
        qs = np.array([0.0, 0.35, 0.8, 1.0])
        qf_eval = self.rv.qf(qs)

        qs_single = 0.172
        qf_eval_single = self.rv.qf(qs_single)

        if DEBUG: logger.info(
            f"Quantile function evaluations: {qf_eval, qf_eval_single}"
        )
        expected = np.array([-np.inf, 4, 8, 10])
        expected_single = 2
        self.assertTrue(
            np.isclose(qf_eval, expected).all() &
            np.isclose(qf_eval_single, expected_single),
            f"Quantile function evaluation failed.\nExpected:\
            {expected, expected_single}\nGot: {qf_eval, qf_eval_single}")

    def test_creation_duplicates(self):
        xk = np.array([1,2,3,3,3,4,5])
        pk = np.ones(xk.size)
        pk = pk / pk.sum()
        rv = RV(xk, pk)
        self.assertTrue(np.unique(rv.xk).size == rv.xk.size)


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.xk = np.array([1, 2, 2, 3, 4, 5, 5.1, 8, 9, 10])
        self.pk = np.ones(self.xk.size)
        self.pk = self.pk / self.pk.sum()
        self.rv = RV(self.xk, self.pk)

    def test_aggregation(self):
        xk_agg, pk_agg = aggregate_conv_results(self.rv.distr())

        if DEBUG: logger.info(f"Aggregated distribution: {xk_agg, pk_agg}")
        expected_xk = np.array([1, 2, 3, 4, 5, 5.1, 8, 9, 10])
        expected_pk = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.assertTrue(
            np.isclose(xk_agg, expected_xk).all() &
            np.isclose(pk_agg, expected_pk).all(),
            f"Aggregation failed.\nExpected: {expected_xk, expected_pk}\nGot:\
            {xk_agg, pk_agg}"
        )