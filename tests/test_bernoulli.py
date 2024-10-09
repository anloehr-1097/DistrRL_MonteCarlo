import unittest
import logging
import scipy.stats as sp
from src.sample_envs import bernoulli_env
from src.preliminary_tests import (
    ContinuousRV,
    ReturnDistributionFunction,
    ddp,
    QuantileProjection,
    quant_projection_algo,
    wasserstein_beta
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# solved approx. correctly.


class TestBernoulli(unittest.TestCase):
    """Test DDP with Bernoulli Env. Expect approx Unif[0,2] distr."""

    def setUp(self):
        self.env = bernoulli_env

    def test_ddp_semantics_quant_projection(self):
        logging.info("test_ddp_semantics_quant_projection")
        approx: ReturnDistributionFunction = self.env.return_distr_fun_est

        for i in range(1,10):
            approx = ddp(mdp=self.env.mdp,
                         inner_projection=QuantileProjection(),
                         outer_projection=QuantileProjection(),
                         param_algorithm=quant_projection_algo,
                         return_distr_function=self.env.return_distr_fun_est,
                         iteration_num=i)

        # compare to unif [0,2]
        unif02 = ContinuousRV(sp.uniform(0, 2))
        w1 = wasserstein_beta(approx[self.env.mdp.states[0]], unif02, 1)
        logging.info(f"Wasserstein distance between approx and Unif[0,2]: {w1}")
        self.assertTrue(w1 <= 0.1, "Check DDP semantics.")
