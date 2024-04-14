from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import preliminary_tests
from preliminary_tests import RV_Discrete
from sklearn.neighbors import KernelDensity


def get_pseudo_density(rv: RV_Discrete, kernel: str="gaussian") -> Tuple[np.ndarray, np.ndarray]:
    """"Plot a pseudo density function of a discrete random variable."""
    xs: np.ndarray = rv.distr()[0]
    ps: np.ndarray = rv.distr()[1]

    # determine linspace / range to plot
    x_min: np.ndarray = np.min(xs)
    x_max: np.ndarray = np.max(xs)
    x_range: np.ndarray = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
    # x_range: np.ndarray = np.linspace(x_min, x_max, 1000)

    # create and fit kernel density estimator
    dens_est: KernelDensity = KernelDensity(kernel='gaussian', bandwidth=0.1)
    dens_est.fit(xs[:, np.newaxis], sample_weight=ps)
    log_dens: np.ndarray = dens_est.score_samples(x_range)
    return x_range, np.exp(log_dens)


def main():
    rv: RV_Discrete = preliminary_tests.main()[0]
    get_pseudo_density(rv)
    return None

if __name__ == '__main__':
    main()