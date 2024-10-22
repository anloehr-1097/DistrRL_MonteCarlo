import unittest
import logging
import scipy.stats as sp
from typing import Type

from src.random_variables import ContinuousRV
from src.projections import (
    GridValueProjection,
    QuantileProjection,
)
from src.param_algorithms import (
    param_algo_with_cdf_algo,
    q_proj_poly_poly
    # quant_projection_algo,
)
from src.drl_primitives import (
    ReturnDistributionFunction,
    wasserstein_beta
)
from src.ddp import ddp
from src.sample_envs import bernoulli_env

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

        for i in range(1, 10):
            approx = ddp(mdp=self.env.mdp,
                         inner_projection=QuantileProjection,
                         outer_projection=QuantileProjection,
                         # param_algorithm=quant_projection_algo,
                         param_algorithm=q_proj_poly_poly,
                         return_distr_function=self.env.return_distr_fun_est,
                         reward_distr_coll=None,
                         iteration_num=i)

        # compare to unif [0,2]
        unif02 = ContinuousRV(sp.uniform(0, 2))
        w1 = wasserstein_beta(approx[self.env.mdp.states[0]], unif02, 1)
        logging.info(f"Wasserstein distance between q-q approx and Unif[0,2]: {w1}")
        self.assertTrue(w1 <= 0.1, "Check DDP semantics.")

    def test_ddp_semantics_cdf_projection(self):
        logging.info("test_ddp_semantics_cdf_projection")
        approx: ReturnDistributionFunction = self.env.return_distr_fun_est
        gv_proj: Type[GridValueProjection] = GridValueProjection
        q_proj: Type[QuantileProjection] = QuantileProjection

        for i in range(1, 10):
            approx = ddp(mdp=self.env.mdp,
                         inner_projection=gv_proj,
                         outer_projection=q_proj,
                         param_algorithm=param_algo_with_cdf_algo,
                         return_distr_function=self.env.return_distr_fun_est,
                         reward_distr_coll=self.env.mdp.rewards,
                         iteration_num=i)

        # compare to unif [0,2]
        unif02 = ContinuousRV(sp.uniform(0, 2))
        w1 = wasserstein_beta(approx[self.env.mdp.states[0]], unif02, 1)
        logging.info(f"Wasserstein distance between cdf-q approx and Unif[0,2]: {w1}")
        self.assertTrue(w1 <= 0.1, "Check DDP semantics.")
