import unittest
from src.sample_envs import bernoulli_env

# TODO make sure to design test that ensures that Bernoulli MDP with policy is solved approx. correctly.

class TestBernoulli(unittest.TestCase):
    def setUp(self):
        self.env = bernoulli_env

    def test_ddp_semantics(self):
        pass
