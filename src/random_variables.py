from __future__ import annotations
from typing import Tuple, Union

import numpy as np
from scipy.stats.distributions import rv_frozen
from .config import NUMBA_SUPPORT, DEBUG, NUM_PRECISION_DECIMALS
from .nb_fun import _sort_njit, _qf_njit, aggregate_conv_results


class RV:
    """Random variable base class used for subclassing."""

    def __init__(self) -> None:
        """Initialize discrete random variable."""
        pass

    def sample(self, num_samples: int=1) -> np.ndarray:
        return np.array([])

    def __call__(self) -> np.ndarray:
        """Sample from distribution."""
        return self.sample()

    def cdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF."""
        raise NotImplementedError

    def qf(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate QF vectorized."""
        raise NotImplementedError


class ContinuousRV(RV):
    """This is a wrapper for scipy continuous random variables.

    The aim is to make continuous rv work with this implementation.

    Needs to be supported:
    - Projections of all kinds
    - Monte Carlo Methods, i.e. sampling
    """

    def __init__(self, scipy_rv_cont: rv_frozen) -> None:
        self.sp_rv_cont = scipy_rv_cont
        self.size = np.inf

    def sample(self, num_samples: int=1) -> np.ndarray:
        return np.asarray(self.sp_rv_cont.rvs(size=num_samples))

    def cdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF at x.
        Args:
        x: np.ndarray of size 1xN
        """
        return self.sp_rv_cont.cdf(x)

    def qf(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate QF vectorized."""
        return self.sp_rv_cont.ppf(u)

    def empirical(self, num_samples: int=500) -> DiscreteRV:
        """Return empirical distribution to continuous RV."""

        samples: np.ndarray = self.sample(num_samples)
        unique, counts = np.unique(samples, return_counts=True)
        return DiscreteRV(unique, counts / samples.size)


class DiscreteRV(RV):
    """Discrete atomic random variable.

    It is vital that the RV is not modified from outside but only with
    the provided methods.
    This is to ensure that the distribution's atoms are sorted when is_sorted is True.
    """

    def __init__(self, xk, pk) -> None:
        """Initialize discrete random variable."""
        assert isinstance(xk, np.ndarray) and isinstance(pk, np.ndarray), \
            "Not numpy arrays upon creation of RV_Discrete."
        assert xk.size == pk.size, "Size mismatch in xk and pk."

        self.xk: np.ndarray
        self.pk: np.ndarray
        self.xk, self.pk = aggregate_conv_results((xk, pk))  # making sure xk has unique values
        self.is_sorted: bool = False
        self.size = self.xk.size

    def support(self) -> Tuple[float, float]:
        """Return support of distribution."""
        return self.xk[0], self.xk[-1]

    def distr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return distribution as Tuple of numpy arrays."""
        return self.xk, self.pk

    def get_cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        self._sort_njit() if NUMBA_SUPPORT else self._sort()
        return self.xk, np.cumsum(self.pk)

    def _sort(self) -> None:
        """Sort values and probs."""
        indices = np.argsort(self.xk)
        self.xk = self.xk[indices]
        self.pk = self.pk[indices]
        self.is_sorted = True

    def _sort_njit(self) -> None:
        self.xk, self.pk = _sort_njit(self.xk, self.pk)
        self.is_sorted = True

    def sample(self, num_samples: int=1) -> np.ndarray:
        """Sample from distribution.

        So far only allow 1D sampling.
        """
        return np.random.choice(self.xk, p=self.pk, size=num_samples)

    def __call__(self) -> np.ndarray:
        """Sample from distribution."""
        return self.sample()

    def _cdf_single(self, x: float, accuracy: float = 1e-10) -> float:
        """Only to be called from cdf method since no sorting."""
        return np.sum(self.pk[self.xk <= x+accuracy])

    def cdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF."""
        if not self.is_sorted:
            self._sort_njit() if NUMBA_SUPPORT else self._sort()

        if isinstance(x, np.ndarray):
            cdf_evals: np.ndarray = np.zeros(x.size)
            for i in range(x.size):
                cdf_evals[i] = self._cdf_single(x[i])
            return cdf_evals

        # any other numeric literal (int, float)
        return np.asarray(self._cdf_single(x))

    def qf_single(self, u: float) -> float:
        """Evaluate quantile function."""
        if np.isclose(u, 1): return self.xk[-1]
        elif np.isclose(u, 0): return -np.inf
        else:
            if not self.is_sorted:
                self._sort_njit() if NUMBA_SUPPORT else self._sort()
            return self.xk[(np.searchsorted(np.round(np.cumsum(self.pk), NUM_PRECISION_DECIMALS), u))]

    def qf(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate QF vectorized."""
        if isinstance(u, float):
            return self.qf_single(u)

        # if NUMBA_SUPPORT:
        #     return _qf_njit(self.xk, self.pk, u)  # u: np.ndarray
        else:
            return np.vectorize(self.qf_single)(u)


def scale(distr: DiscreteRV, gamma: np.float64) -> DiscreteRV:
    """Scale distribution by factor."""
    new_val = distr.xk * gamma
    new_prob = distr.pk
    return DiscreteRV(new_val, new_prob)


def conv(a: DiscreteRV, b: DiscreteRV) -> DiscreteRV:
    """Convolution of two distributions.
    0th entry values, 1st entry probabilities.
    """
    new_val: np.ndarray = np.add(a.xk, b.xk[:, None]).flatten()
    probs: np.ndarray = np.multiply(a.pk, b.pk[:, None]).flatten()
    return DiscreteRV(new_val, probs)

