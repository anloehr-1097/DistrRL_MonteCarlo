"""Tests for the random variable abstractions."""

import numpy as np
from typing import Tuple
import unittest
from src.preliminary_tests import RV

DEBUG: bool = False


class TestFiniteRV(unittest.TestCase):
    def setUp(self):
        self.rv = RV(
            xk=np.arange(0, 10, 10),
            pk=(np.ones(10) / 10),
        )
        return None

    def test_cdf(self):
        cdf_eval = self.rv.cdf(3)
        self.assertTrue(np.isclose(cdf_eval, 0.3),
                        "CDF evaluation failed.")


