from typing import Tuple, List, Callable, Union, Optional, Dict, Sequence
from collections import OrderedDict
import functools
from enum import Enum
import numpy as np
from .random_variables import DiscreteRV, RV
from .drl_primitives import (
    PPComponent,
    PPKey,
    ReturnDistributionFunction,
    RewardDistributionCollection,
    ProjectionParameter,
    MDP,
    State,
    Action,
    ParamAlgo,
)




class Projection:
    """Projection operator."""

    @classmethod
    def project(cls, rv: RV, projection_param: PPComponent, *args, **kwargs) -> DiscreteRV:
        """Project distribution."""
        raise NotImplementedError("Project method not implemented.")

    @classmethod
    def __call__(cls, rv: RV, projection_param: PPComponent, *args, **kwargs) -> DiscreteRV:
        """Apply projection to distribution.

        Projection depends on projection parameter component.
        """
        return cls.project(rv, projection_param, *args, **kwargs)


class RandomProjection(Projection):
    """Random Projection"""

    @classmethod
    def project(cls, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        assert isinstance(projection_param, int), \
            "Random Projection expects int parameter."

        atoms = rv.sample(projection_param)
        weights = np.ones(projection_param) / projection_param
        return DiscreteRV(atoms, weights)


class QuantileProjection(Projection):
    """Quantile projection."""

    @classmethod
    def project(cls, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        """Apply quantile projection."""
        assert isinstance(projection_param, int), \
            "Quantile Projection expects int parameter."
        return quantile_projection(rv, projection_param)


class GridValueProjection(Projection):
    """Grid Value Projection."""

    @classmethod
    def project(cls, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        assert isinstance(projection_param, np.ndarray), \
            "Grid Value Projection expects numpy ndarray parameter."

        assert projection_param.size % 2 == 1, \
            "projection param must be of size 2m - 1 where m in |N."

        return grid_value_projection(rv, projection_param)


# @njit
def quantile_projection(rv: RV,
                        no_quantiles: int) -> DiscreteRV:
    """Apply quantile projection as described in book to a distribution."""
    if rv.size <= no_quantiles and isinstance(rv, DiscreteRV): return rv  # type: ignore
    quantile_locs: np.ndarray = (2 * (np.arange(1, no_quantiles + 1)) - 1) /\
        (2 * no_quantiles)
    quantiles_at_locs = rv.qf(quantile_locs)
    return DiscreteRV(quantiles_at_locs, np.ones(no_quantiles) / no_quantiles)


def grid_value_projection(rv: RV, projection_param: np.ndarray) -> DiscreteRV:
    """Grid value projection."""
    param_size: int = (projection_param.size // 2) + 1
    xs: np.ndarray = projection_param[:param_size]
    ys: np.ndarray = projection_param[param_size:]
    y_evals: np.ndarray = rv.cdf(ys)
    pk: np.ndarray = np.concatenate([y_evals, np.asarray([1])]) - \
        np.concatenate([np.asarray([0]), y_evals])
    return DiscreteRV(xs, pk)


