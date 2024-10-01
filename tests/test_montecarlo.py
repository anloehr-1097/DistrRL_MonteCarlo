import unittest

from src.preliminary_tests import monte_carlo_eval
from src.sample_envs import cyclical_env, bernoulli_env


class TestMC(unittest.TestCase):
    def setUp(self):
        return None

    def test_no_runtime_error(self):
        rdf_cyclical_est = monte_carlo_eval(cyclical_env.mdp, 10, 40)
        rdf_bernoulli_est = monte_carlo_eval(bernoulli_env.mdp, 10, 40)
        self.assertIsNotNone(rdf_cyclical_est, "Cyclical env return distribution eval failed.")
        self.assertIsNotNone(rdf_bernoulli_est, "Bernoulli env return distribution eval failed.")
