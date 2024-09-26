"""Tests for Distributional Dynamic Programming algo."""

import unittest
import numpy as np
from src.preliminary_tests import (
    ddp,
    QuantileProjection, quant_projection_algo
)
from src.sample_envs import cyclical_env
import logging

DEBUG: bool = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestDDP(unittest.TestCase):
    def test_ddp_one_step(self):

        ddp(cyclical_env.mdp,
            QuantileProjection(),
            QuantileProjection(),
            quant_projection_algo,
            cyclical_env.return_distr_fun_est,
            1)
        if DEBUG:
            logger.info("DDP one step completed.")

        self.assertNoLogs(logger, level=logging.ERROR)
