# import scipy.stats as sp
import numpy as np
from .random_variables import RV


def emp_normal(mean: float=0, std: float=1, size: int=100) -> RV:
    """Generate a sample from a normal distribution."""
    xs = np.random.normal(mean, std, size)
    ps = np.ones(size) / np.float64(size)
    return RV(xs, ps)
