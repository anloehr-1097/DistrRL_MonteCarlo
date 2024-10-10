import unittest
import numpy as np
from src.random_variables import DiscreteRV
from src.drl_primitives import wasserstein_beta


class TestWassersteinBetaDistance(unittest.TestCase):
    def setUp(self):
        self.xi: np.ndarray = np.linspace(1, 4, 4)
        self.yi: np.ndarray = np.linspace(2.5, 5.5, 4)
        self.ps: np.ndarray = np.ones(self.yi.size) / self.yi.size

        self.rv1 = DiscreteRV(self.xi, self.ps)
        self.rv2 = DiscreteRV(self.yi, self.ps)

    def test_wasserstein_beta_distance(self):
        beta = 1
        w1 = wasserstein_beta(self.rv1, self.rv2, beta)
        self.assertEqual(w1, 1.5, "Error in Wasserstein calc. {w1} != 1.5")
