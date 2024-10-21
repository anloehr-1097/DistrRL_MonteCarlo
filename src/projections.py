from typing import Tuple, List, Callable, Union, Optional, Dict
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
    transform_to_param_algo
)


MAX_ITERS: int = 100000


class Projection:
    """Projection operator."""

    def __init__(self) -> None:
        pass

    def project(self, rv: RV, projection_param: PPComponent, *args, **kwargs) -> DiscreteRV:
        """Project distribution."""
        raise NotImplementedError("Project method not implemented.")

    def __call__(self, rv: RV, projection_param: PPComponent, *args, **kwargs) -> DiscreteRV:
        """Apply projection to distribution.

        Projection depends on projection parameter component.
        """
        return self.project(rv, projection_param, *args, **kwargs)


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


# q_proj: QuantileProjection = QuantileProjection


class GridValueProjection(Projection):
    """Grid Value Projection."""

    @classmethod
    def project(cls, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        assert isinstance(projection_param, np.ndarray), \
            "Grid Value Projection expects numpy ndarray parameter."

        assert projection_param.size % 2 == 1, \
            "projection param must be of size 2m - 1 where m in |N."

        return grid_value_projection(rv, projection_param)


def poly_size_fun(x: int) -> PPComponent: return x**2
def exp_size_fun(x: int) -> PPComponent: return 2**x
def poly_decay(x: int) -> float: return 1/(x**2)
def exp_decay(x: int) -> float: return 1/2**(x)


class SizeFun(Enum):
    POLY: Callable[[int], PPComponent] = poly_size_fun
    EXP: Callable[[int], PPComponent] = exp_size_fun
    POLY_DECAY: Callable[[int], float] = poly_decay
    EXP_DECAY: Callable[[int], float] = exp_decay


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


# type ParamAlgo
def quant_projection_algo(
    iteration: int,
    ret_distr_fun: ReturnDistributionFunction,
    rew_distr_coll: Optional[RewardDistributionCollection],
    mdp: MDP,
    inner_index_set: List[Tuple[State, Action, State]],
    outer_index_set: List[State]
        ) -> Tuple[ProjectionParameter, ProjectionParameter]:

    return algo_size_fun(
        iteration_num=iteration,
        inner_index_set=inner_index_set,
        outer_index_set=outer_index_set,
        inner_size_fun=SizeFun.POLY,
        outer_size_fun=SizeFun.POLY)


# type ParamAlgo
def quant_projection_algo_template(
    inner_size_fun: Callable[[int], PPComponent],
    outer_size_fun: Callable[[int], PPComponent],
    iteration: int,
    ret_distr_fun: ReturnDistributionFunction,
    rew_distr_coll: Optional[RewardDistributionCollection],
    mdp: MDP,
    inner_index_set: List[Tuple[State, Action, State]],
    outer_index_set: List[State]
        ) -> Tuple[ProjectionParameter, ProjectionParameter]:

    return algo_size_fun(
        iteration_num=iteration,
        inner_index_set=inner_index_set,
        outer_index_set=outer_index_set,
        inner_size_fun=inner_size_fun,
        outer_size_fun=outer_size_fun)


q_proj_poly_poly: ParamAlgo = transform_to_param_algo(
    quant_projection_algo_template,
    SizeFun.POLY,
    SizeFun.POLY)


def make_grid_finer(
    rv: RV,
    rv_est_supp: np.ndarray,
    left_prec: float,
    right_prec: float,
        inter_prec: float
        ) -> np.ndarray:

    # prev_support: np.ndarray = np.asarray(rv_est_supp[0])
    new_support: np.ndarray = rv_est_supp
    real_probs: np.ndarray = rv.cdf(rv_est_supp)

    if left_prec < real_probs[0]:
        # expand support to the left
        new_support = np.concatenate(
            [np.asarray([rv_est_supp[0] - 2*(rv_est_supp[0])]),
             new_support]
        )

    if (1-right_prec) > real_probs[1]:
        # expand support to the right
        new_support = np.concatenate(
            [np.asarray([rv_est_supp[-1] + 2*(rv_est_supp[-1])]),
             new_support]
        )

    # determine where approx to coarse
    inter_k: List = []
    cdf_evals: np.ndarray = rv.cdf(new_support)
    cdf_diffs: np.ndarray = np.diff(cdf_evals)
    new_eval_positions: np.ndarray = cdf_diffs > inter_prec
    for i in range(new_eval_positions.size):
        if new_eval_positions[i]:
            inter_k.append((new_support[i+1] + new_support[i]) / 2)

    new_support = np.concatenate([new_support, inter_k])
    new_support = np.sort(np.unique(new_support))  # yi
    intermed_points: np.ndarray = (new_support[1:] + new_support[:-1]) / 2
    # return np.concatenate([intermed_points, new_support])
    return np.concatenate([new_support, intermed_points])


def algo_cdf_1(
    # iteration_num: int,
    inner_index_set: List[Tuple[State, Action, State]],
    previous_return_estimate: Optional[ReturnDistributionFunction],
    previous_reward_estimate: Optional[RewardDistributionCollection],
    mdp: MDP,
    f_min: Callable[[int], float] = SizeFun.POLY_DECAY,
    f_max: Callable[[int], float] = SizeFun.EXP_DECAY,
    f_inter: Callable[[int], float] = SizeFun.EXP_DECAY

        ) -> ProjectionParameter:
    """Yield projection parameter for grid proj as inner proj of rewards.

    This is an implementation of A_{inout{CDF}, 1}.
    """
    # Algo CDF 1
    min_prob: float
    max_prob: float
    inter_prob: float
    pp_val: Dict[PPKey, PPComponent] = {}
    num_iter: int = 1
    # stopping_criterion: bool = num_iter > 10

    for (state, action, next_state) in inner_index_set:
        num_iter = 1  # reset counter
        prev_rew_est: DiscreteRV = previous_reward_estimate[(state, action, next_state)]  # type: ignore

        print(prev_rew_est)
        grid: np.ndarray = prev_rew_est.xk

        while True:
            min_prob, max_prob = f_min(num_iter), f_max(num_iter)
            inter_prob = f_inter(num_iter)
            grid = make_grid_finer(
                mdp.rewards[(state, action, next_state)],
                grid,
                min_prob,
                max_prob,
                inter_prob)
            num_iter += 1

            if num_iter > 5:
                break

        pp_val[(state, action, next_state)] = grid
    return ProjectionParameter(pp_val)


def support_find(rv: RV, eps: float=1e-8) -> Tuple[float, float]:
    """Find support of distribution."""

    if isinstance(rv, DiscreteRV):
        return rv.xk[0], rv.xk[-1]

    else:
        r_min: float = 0
        r_max: float = 0
        k: int = 1

    # continuous case
    while True:
        if rv.cdf(r_min) > (eps / 2):
            r_min -= 2**k

        if rv.cdf(r_max) < 1 - (eps / 2):
            r_max += 2**k

        if rv.cdf(r_min) < (eps / 2) and rv.cdf(r_max) > 1 - (eps / 2): break
        k += 1

    return r_min, r_max


def bisect_single(quantile: float, rv: RV,  lower_bound: float, upper_bound: float, precision: float, max_iters: int=MAX_ITERS) -> float:
    """Bisection algorithm for quantile finding."""
    x: float
    count: int = 0
    while True:
        x = (lower_bound + upper_bound) / 2
        count += 1
        if abs(rv.cdf(x) - quantile) < precision: return x

        if rv.cdf(x) < quantile:
            lower_bound = x

        elif rv.cdf(x) > quantile:
            upper_bound = x

        if np.abs(upper_bound - lower_bound) < precision: return x
        elif count > MAX_ITERS:
            return x
        else:
            pass


# TODO if time permits, also pass array of bounds, see if faster
def bisect(rv: RV, quantile: Union[float, np.ndarray], lower_bound: float, upper_bound: float, precision: float) -> float:
    """Bisection algorithm for quantile finding."""
    bs: Callable[[float], float] = functools.partial(
        bisect_single,
        rv=rv, lower_bound=lower_bound,
        upper_bound=upper_bound,
        precision=precision)
    return np.vectorize(bs, excluded={"rv", "lower_bound", "upper_bound", "precision"})(quantile)  # Callable[[np.ndarrayl, np.ndarray]


def quantile_find(
    rv: RV,
    num_iter: int,
    q_table: OrderedDict[float, float],
        precision: float=1e-8) -> OrderedDict[float, float]:
    """Find quantiles of distribution."""
    if not len(q_table.keys()) >= 2:
        # TODO raise exception when keys are missing
        raise Exception("Quantile table must have at least 2 entries.")

    for i in range(2**(num_iter - 1)):
        q: float = bisect(
            rv=rv,
            quantile=((2*i + 1)/(2**num_iter)),
            lower_bound=q_table[(i/(2**(num_iter-1)))],
            upper_bound=q_table[((i+1)/(2**(num_iter-1)))],
            precision=precision
        )
        q_table[((2*i + 1)/(2**num_iter))] = q
    return q_table


def algo_cdf_2(
    num_iteration: int,
    inner_index_set: List[Tuple[State, Action, State]],
    previous_reward_estimate: RewardDistributionCollection,
    mdp: MDP,
    precision: float,
        ) -> ProjectionParameter:

    # TODO probably add q-table collection as argument
    r_min: float
    r_max: float
    pp_val: Dict[PPKey, PPComponent] = {}

    for (state, action, next_state) in inner_index_set:
        r_min, r_max = support_find(mdp.rewards[(state, action, next_state)], eps=precision)
        q_table: OrderedDict[float, float] = OrderedDict({0: r_min, 1: r_max})
        q_table = quantile_find(mdp.rewards[(state, action, next_state)], num_iteration, q_table, precision)
        grid: np.ndarray = np.asarray(list(q_table.values()))
        pp_val[(state, action, next_state)] = grid
    return ProjectionParameter(pp_val)


def param_algo_with_cdf_algo(
    iteration: int,
    return_distr_function: ReturnDistributionFunction,
    reward_approx: Optional[RewardDistributionCollection],
    mdp: MDP,
    inner_index_set: List[Tuple[State, Action, State]],
    outer_index_set: List[State],
        ) -> Tuple[ProjectionParameter, ProjectionParameter]:

    decay_funs: Tuple[Callable, Callable, Callable] = (SizeFun.POLY_DECAY, SizeFun.EXP_DECAY, SizeFun.EXP_DECAY)
    size_funs: Tuple[Callable, Callable] = (SizeFun.POLY, SizeFun.EXP)

    inner_param: ProjectionParameter = algo_cdf_1(
        inner_index_set=inner_index_set,
        previous_return_estimate=None,
        previous_reward_estimate=reward_approx,
        mdp=mdp,
        f_min=decay_funs[0],
        f_max=decay_funs[1],
        f_inter=decay_funs[2])

    outer_param: ProjectionParameter = algo_size_fun(
        iteration, inner_index_set, outer_index_set,
        *size_funs[:2])[-1]

    return inner_param, outer_param


def algo_size_fun(
    iteration_num: int,
    inner_index_set: List[Tuple[State, Action, State]],
    outer_index_set: List[State],
    inner_size_fun: Callable[[int], PPComponent],
    outer_size_fun: Callable[[int], PPComponent],
    previous_return_estimate: Optional[ReturnDistributionFunction]=None,
    previous_reward_estimate: Optional[RewardDistributionCollection]=None
        ) -> Tuple[ProjectionParameter, ProjectionParameter]:
    """Apply size functions to num_iteration.

    Provide any 2 size functions from |N -> |N
    """

    inner_params: Dict[Union[Tuple[State, Action, State], State], PPComponent] = {
        idx: inner_size_fun(iteration_num) for idx in inner_index_set
    }
    outer_params: Dict[Union[Tuple[State, Action, State], State], PPComponent] = {
        idx: outer_size_fun(iteration_num) for idx in outer_index_set
    }

    return ProjectionParameter(inner_params), ProjectionParameter(outer_params)
