import logging
from typing import Callable, Tuple, Union
import numpy as np
from numba import njit
# from src.preliminary_tests import DiscreteRV

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def assert_probs_distr(probs: np.ndarray) -> None:
    assert np.isclose(np.sum(probs), 1), "Probs do not sum to 1."


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    if np.allclose(np.sum(probs), 1, atol=1e-11):
        return probs
    logger.warning(f"Probs do sum to {np.sum(probs)}.\
                   Normalizing to sum to 1.")
    return probs / np.sum(probs)


def dkw_bounds(
    cdf: Union[Callable, Tuple[np.ndarray, np.ndarray]],
    xval: np.ndarray,
    n: int,
        prob: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cdf along with DKW bounds to given prob level."""
    # p = 2 * exp(-2 * n * epsilon^2)
    epsilon = np.sqrt(np.log(prob / 2) / (- 2 * n))
    if callable(cdf):
        yval = cdf(xval)  # assume cdf consumes np.ndarray
    else:
        yval = cdf[1]

    upper_bound = np.minimum(yval + epsilon, 1)
    lower_bound = np.maximum(yval - epsilon, 0)
    return yval, lower_bound, upper_bound
