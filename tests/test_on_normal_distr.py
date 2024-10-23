import unittest
from typing import Callable
import copy
import logging
from numba.core.utils import functools
import numpy as np
from src.ddp import ddp, dbo
from src.drl_primitives import (
    PPComponent,
    ReturnDistributionFunction,
    RewardDistributionCollection,
    MDP,
    ParamAlgo,
    extended_metric,
    wasserstein_beta
)
from src.param_algorithms import (
    SizeFun,
    combine_to_param_algo,
    transform_to_param_algo,
    param_algo_from_size_fun
)
from src.projections import QuantileProjection, RandomProjection
from src.random_variables import DiscreteRV
from src.sample_envs import cyclical_env, cyclycal_real_return_distr_fun

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestDDPOnCyclicalEnv(unittest.TestCase):
    def setUp(self):
        self.mdp: MDP = cyclical_env.mdp
        self.ret_distr_est: ReturnDistributionFunction = ReturnDistributionFunction(
            states=cyclical_env.mdp.states,
            distributions=[
                DiscreteRV(xk=np.zeros(1), pk=np.ones(1)) for _ in cyclical_env.mdp.states
            ]
        )
        self.rewards: RewardDistributionCollection = RewardDistributionCollection(
            state_action_state_triples=list(cyclical_env.mdp.rewards.rewards.keys()),
            distributions=[
                cyclical_env.mdp.rewards[_].empirical(50)
                for _ in cyclical_env.mdp.rewards.rewards.keys()
            ]
        )

    def test_dbo(self):
        ret_distr_est: ReturnDistributionFunction = copy.deepcopy(self.ret_distr_est)
        for _ in range(2):
            dbo(self.mdp, ret_distr_est, self.rewards)
        logger.info(f"Num atoms after 2 applications of DBO: \
            {ret_distr_est[self.mdp.states[0]].size}")
        logger.info("DBO application completed for 2 steps")

    def test_ddp(self):
        num_iterations: int = 9 
        ret_distr_est: ReturnDistributionFunction = copy.deepcopy(self.ret_distr_est)
        inner_size_fun: Callable[[int], PPComponent] = functools.partial(SizeFun.POLY, 2)
        outer_size_fun: Callable[[int], PPComponent] = functools.partial(SizeFun.POLY, 3)

        param_algo: ParamAlgo = combine_to_param_algo(
            transform_to_param_algo(  # type: ignore
                param_algo_from_size_fun,
                size_fun=inner_size_fun
            ),
            transform_to_param_algo(  # type: ignore
                param_algo_from_size_fun,
                size_fun=outer_size_fun,
                distr_coll=None
            )
        )
        for i in range(20):
            ret_distr_est = ddp(
                self.mdp,
                QuantileProjection,
                RandomProjection,
                param_algo,
                ret_distr_est,
                self.rewards,
                i+1
            )

        logger.info("10 iterations of DDP completed.")
        logger.info(f"Size per component of last iterat: {ret_distr_est[self.mdp.states[0]].size}")
        logger.info(f"Expected size: {outer_size_fun(10)}")
        logger.info(f"Wasserstein distance to real return distr function: \
            {extended_metric(wasserstein_beta, ret_distr_est.distr, cyclycal_real_return_distr_fun.distr)}")
        self.assertTrue(ret_distr_est[self.mdp.states[0]].size <= outer_size_fun(10))
        
