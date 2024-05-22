import scipy.stats as sp
import numpy as np
from .preliminary_tests import RV_Discrete

def emp_normal(mean: float=0, variance: float=1, size: int=100) -> np.ndarray:
    """Generate a sample from a normal distribution."""
    xs = np.random.normal(mean, variance, size)
    ps = np.ones(size) / np.float64(size)
    return RV_Discrete(xs, ps)


