from __future__ import annotations
import logging
from typing import Tuple, Union, Optional
from numba import njit
import numpy as np
from scipy.stats.distributions import rv_frozen
from scipy.stats import rv_discrete
# from scipy.stats._distn_infrastructure import rv_sample
from .config import NUMBA_SUPPORT, NUM_PRECISION_DECIMALS, SCIPY_ACTIVE
from .nb_fun import _sort_njit, _qf_njit, aggregate_conv_results, cdf_njit
from .utils import normalize_probs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def empirical(self, num_samples: int=50) -> DiscreteRV:
        raise NotImplementedError

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
        self.sp_rv = scipy_rv_cont
        self.size = np.inf

    def sample(self, num_samples: int=1) -> np.ndarray:
        return np.asarray(self.sp_rv.rvs(size=num_samples))

    def cdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF at x.
        Args:
        x: np.ndarray of size 1xN
        """
        return self.sp_rv.cdf(x)

    def qf(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate QF vectorized."""
        return self.sp_rv.ppf(u)

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
        # self.xk, self.pk = self.make_unique_atoms(xk, pk)
        self.is_sorted: bool = True
        self.size: int = self.xk.size
        # remove later
        self.pk = normalize_probs(self.pk)
        assert np.allclose(np.sum(self.pk), 1.0, atol=1e-10), "Probs do not sum to 1."

        self.sp_rv: Optional[rv_discrete] = rv_discrete(values=(self.xk, self.pk)) if SCIPY_ACTIVE else None

    def support(self) -> Tuple[float, float]:
        """Return support of distribution."""
        if self.sp_rv is not None:
            return self.sp_rv.a, self.sp_rv.b
        return self.xk[0], self.xk[-1]

    def distr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return distribution as Tuple of numpy arrays."""
        return self.xk, self.pk


    def make_unique_atoms(self, xk: np.ndarray, pk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make sure that xk has unique values."""

        unq_atoms, inv_indices = np.unique(xk, return_inverse=True)
        new_probs = np.zeros(unq_atoms.size)
        for i in range(xk.size):
            new_probs[i] = np.sum(pk[inv_indices == i])

        assert np.allclose(np.sum(new_probs), 1.0, atol=1e-10), \
            "Probs do not sum to 1."
        return unq_atoms, new_probs


    def get_cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.sp_rv is not None:
            return self.sp_rv.xk, self.sp_rv.cdf(self.sp_rv.xk)  # type: ignore
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
        if self.sp_rv is not None:
            return self.sp_rv.rvs(size=num_samples)  # type: ignore
        return np.random.choice(self.xk, p=self.pk, size=num_samples)

    def __call__(self) -> np.ndarray:
        """Sample from distribution."""
        return self.sample()

    def _cdf_single(self, x: float, accuracy: float = 1e-10) -> float:
        """Only to be called from cdf method since no sorting."""
        return np.sum(self.pk[self.xk <= x+accuracy])

    def cdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF."""
        if self.sp_rv is not None:
            return self.sp_rv.cdf(x)

        if isinstance(x, np.ndarray):
            xks: np.ndarray = np.tile(self.xk, (x.size, 1))
            cond: np.ndarray = xks <= x[:, np.newaxis]
            pks: np.ndarray = np.tile(self.pk, (x.size, 1))
            return np.sum(pks * cond, axis=1)

        # if not self.is_sorted:
        #     self._sort_njit() if NUMBA_SUPPORT else self._sort()
        #
        # if isinstance(x, np.ndarray):
        #     # return np.vectorize(self._cdf_single)(x)
        #     cdf_evals: np.ndarray = np.zeros(x.size)
        #     for i in range(x.size):
        #         cdf_evals[i] = self._cdf_single(x[i])
        #     return cdf_evals
        # # any other numeric literal (int, float)
        return np.asarray(self._cdf_single(x))

    def cdf_vec(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate CDF."""
        if self.sp_rv is not None:
            return self.sp_rv.cdf(x)
        if isinstance(x, np.ndarray):
            if NUMBA_SUPPORT:
                return cdf_njit(self.xk, self.pk, x)
            else:
                return np.vectorize(self._cdf_single)(x)
        else:
            return np.asarray(self._cdf_single(x))

    def improved_cdf(self, x: np.ndarray) -> np.ndarray:
        """Pure numpy cdf eval."""
        xks: np.ndarray = np.tile(self.xk, (x.size, 1))
        logger.info(f"xks created. Shape {xks.shape}")
        cond: np.ndarray = xks <= x[:, np.newaxis]
        logger.info(f"condition created. Shape {cond.shape}")
        pks: np.ndarray = np.tile(self.pk, (x.size, 1))
        logger.info(f"pks created. Shape {pks.shape}")
        return np.sum(pks * cond, axis=1)

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
        if self.sp_rv is not None:
            return self.sp_rv.ppf(u)

        if isinstance(u, float):
            return self.qf_single(u)

        if NUMBA_SUPPORT:
            return _qf_njit(self.xk, self.pk, u)  # type: ignore | u: np.ndarray
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

