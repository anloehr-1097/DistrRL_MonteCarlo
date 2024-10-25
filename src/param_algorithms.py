import functools
from typing import Tuple, List, Callable, Union, Optional, Dict, Sequence, Any
from collections import OrderedDict
from enum import Enum
import numpy as np
from .config import N_ITER
from .random_variables import DiscreteRV, RV
from .drl_primitives import (
    OneComponentParamAlgo,
    OuterParamAlgo,
    InnerParamAlgo,
    ParamAlgo,
    PPComponent,
    PPKey,
    ReturnDistributionFunction,
    RewardDistributionCollection,
    ProjectionParameter,
    MDP,
    State,
    Action,
)


def combine_to_param_algo(
    inner_algo: InnerParamAlgo,
    outer_algo: OuterParamAlgo,
        ) -> ParamAlgo:
    """Combine Param Algos for inner and outer parameters."""

    def param_algo(
        iteration_num: int,
        ret_distr_fun: ReturnDistributionFunction,
        reward_distr_coll: RewardDistributionCollection,
        mdp: MDP,
        inner_index_set: List[Tuple[State, Action, State]],
        outer_index_set: List[State]
            ) -> Tuple[ProjectionParameter, ProjectionParameter]:

        inner_param: ProjectionParameter = inner_algo(
            iteration_num=iteration_num,
            distr_coll=mdp.rewards,
            distr_coll_estimate=reward_distr_coll,
            mdp=mdp,
            index_set=inner_index_set
        )

        outer_param: ProjectionParameter = outer_algo(
            iteration_num=iteration_num,
            # None,
            distr_coll_estimate=ret_distr_fun,
            mdp=mdp,
            index_set=outer_index_set
        )
        return inner_param, outer_param

    return param_algo


def transform_to_param_algo(
    func: Callable[..., ProjectionParameter],
    *args: Any,
        **kwargs: Any) -> OneComponentParamAlgo:
    """Transform function to OneComponentParamAlgo.

    The role of this function is to produce a curried function with exactly
    the arguments a one component parameter algo expects according to its type
    definition.
    """
    return functools.partial(func, *args, **kwargs)


# Size functions
def poly_size_fun(exponent: float, x: int) -> PPComponent: return round(x**exponent)
def exp_size_fun(base: float, x: int) -> PPComponent: return round(base**x)


class SizeFun(Enum):
    POLY: Callable[[int, int], PPComponent] = poly_size_fun
    EXP: Callable[[int, int], PPComponent] = exp_size_fun


# Decay functions
def poly_decay(exponent: float, x: int) -> float: return 1/(x**exponent)
def exp_decay(base: float, x: int) -> float: return 1/base**(x)


class DecayFun(Enum):
    POLY: Callable[[int, int], float] = poly_decay
    EXP: Callable[[int, int], float] = exp_decay


# type: OneComponentParamAlgo
def param_algo_from_size_fun(
    size_fun: Callable[[int], int],
    iteration_num: int,
    distr_coll: Optional[Union[ReturnDistributionFunction, RewardDistributionCollection]],
    distr_coll_estimate: Union[ReturnDistributionFunction, RewardDistributionCollection],
    mdp: MDP,
    index_set: Sequence[PPKey]
        ) -> ProjectionParameter:
    """Return a projection parameter from a size function."""
    return size_fun_broadcast(distr_coll_estimate.index_set, size_fun, iteration_num)


# size funs n->n**2, n->n**3
q_proj_poly_poly = combine_to_param_algo(
    transform_to_param_algo(  # type: ignore
        param_algo_from_size_fun,
        size_fun=functools.partial(SizeFun.POLY, 2)
    ),
    transform_to_param_algo(
        param_algo_from_size_fun,  # type: ignore
        size_fun=functools.partial(SizeFun.POLY, 3),
        distr_coll=None
    )
)

# q_proj_poly_poly: ParamAlgo = transform_to_param_algo(
#     param_algo_2_size_fun_template,
#     functools.partial(SizeFun.POLY, 2),
#     functools.partial(SizeFun.POLY, 3))


def make_grid_finer(
    rv: RV,
    rv_est_supp: np.ndarray,
    left_prec: float,
    right_prec: float,
        inter_prec: float
        ) -> np.ndarray:
    """Refine grid of est dist by usign CDF eval from real dist.

    This function is used in algo_cdf_1 as introduced in paper.
    """
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


# after specifying size functions type OneComponentParamAlgo
def algo_cdf_1(
    iteration_num: int,
    distr_coll: Union[ReturnDistributionFunction, RewardDistributionCollection],
    distr_coll_estimate: Union[ReturnDistributionFunction, RewardDistributionCollection],
    mdp: MDP,
    index_set: Sequence[PPKey],
    f_min: Callable[[int], float],
    f_max: Callable[[int], float],
    f_inter: Callable[[int], float]
        ) -> ProjectionParameter:
    """Yield projection parameter for grid proj.
    This is an implementation of A_{CDF, 1}.
    """
    # Algo CDF 1
    min_prob: float
    max_prob: float
    inter_prob: float
    pp_val: Dict[PPKey, PPComponent] = {}
    num_iter: int = 1
    # stopping_criterion: bool = num_iter > 10

    # check if everything fits together
    # handle that some indices are not in estimate if prob of (s,a,bar{s}) = 0
    # assert index_set == distr_coll_estimate.index_set,  \
    #     f"Index set mismatch est. dist coll. {index_set} != {distr_coll_estimate.index_set}"
    # assert index_set == distr_coll.index_set, \
    #     f"Index set mismatch real dist coll.{index_set} != {distr_coll_estimate.index_set}"

    for idx in index_set:
        num_iter = 1  # reset counter
        if idx not in distr_coll_estimate.index_set:
            continue
        prev_est: DiscreteRV = distr_coll_estimate[idx]  # type: ignore
        grid: np.ndarray = prev_est.xk

        while True:
            min_prob, max_prob = f_min(num_iter), f_max(num_iter)
            inter_prob = f_inter(num_iter)
            grid = make_grid_finer(
                distr_coll[idx],  # type: ignore
                grid,
                min_prob,
                max_prob,
                inter_prob)
            num_iter += 1

            # TODO real stopping criterion here
            if num_iter > 5:
                break
        pp_val[idx] = grid

    return ProjectionParameter(pp_val)


def support_find(rv: RV, eps: float=1e-8) -> Tuple[float, float]:
    """Find support of distribution."""

    # rv is discrete -> clear how to determine support
    if isinstance(rv, DiscreteRV):
        return rv.xk[0], rv.xk[-1]

    # else RV is continuous
    r_min: float = 0
    r_max: float = 0
    k: int = 1

    # define soptting criterion based on eps provided in function def
    def stopping_crit(x: float, y: float, eps: float) -> np.bool_:
        return np.all(rv.cdf(x) < (eps / 2)) and np.all(rv.cdf(y) > (1 - (eps / 2)))

    # continuous case
    while True:
        if rv.cdf(r_min) > (eps / 2):
            r_min -= 2**k
        if rv.cdf(r_max) < 1 - (eps / 2):
            r_max += 2**k
        # if rv.cdf(r_min) < (eps / 2) and rv.cdf(r_max) > 1 - (eps / 2): break
        if stopping_crit(r_min, r_max, eps): break
        k += 1
    return r_min, r_max


def bisect_single(
    quantile: float,
    rv: RV,
    lower_bound: float,
    upper_bound: float,
    precision: float,
        max_iters: int=N_ITER) -> float:
    """Find quantile of rv within lower and upper bound up to precision.

    This implements bisection algorithm from thesis.
    precision ~ delta
    quantile ~ u

    The stopping criterion encoded here implicitly is based
    on the difference between the evaluation CDF(x) and the quantile
    searched for. At the same time, the candidate is returned if the 
    interval length < delta.
    """

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
        elif count > N_ITER:
            return x
        else:
            pass


# TODO if time permits, also pass array of bounds, see if faster
def bisect(
    rv: RV,
    quantile: Union[float, np.ndarray],
    lower_bound: float,
    upper_bound: float,
        precision: float) -> float:
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
        q: float = bisect_single(
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
    index_set: Sequence[PPKey],
    distr_coll_estimate: Union[ReturnDistributionFunction, RewardDistributionCollection],
    distr_coll: Union[ReturnDistributionFunction, RewardDistributionCollection],
    mdp: MDP,
    precision: float,
        ) -> ProjectionParameter:
    """Produce projection parameter with A_{CDF, 2}.

    Implementation of A_{CDF, 2} in thesis.
    """

    # TODO probably add q-table collection as argument
    r_min: float
    r_max: float
    pp_val: Dict[PPKey, PPComponent] = {}

    for idx in index_set:
        r_min, r_max = support_find(distr_coll[idx], eps=precision)  # type: ignore
        q_table: OrderedDict[float, float] = OrderedDict({0: r_min, 1: r_max})
        q_table = quantile_find(distr_coll[idx], num_iteration, q_table, precision)  # type: ignore
        xs: np.ndarray = np.asarray(q_table.values())[1::2]
        ys: np.ndarray = np.asarray(q_table.keys())[2:-1:2]
        # grid: np.ndarray = np.asarray(list(q_table.values()))
        grid: np.ndarray = np.concatenate([xs, ys])
        pp_val[idx] = grid
    return ProjectionParameter(pp_val)


param_algo_with_cdf_algo: ParamAlgo = combine_to_param_algo(
    transform_to_param_algo(
        algo_cdf_1,
        f_min=functools.partial(DecayFun.POLY, 2),
        f_max=functools.partial(DecayFun.EXP, 3),
        f_inter=functools.partial(DecayFun.EXP, 3)),  # type: ignore
    transform_to_param_algo(
        param_algo_from_size_fun,
        size_fun=functools.partial(SizeFun.POLY, 2),  # type: ignore
        distr_coll=None)
    )


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


def size_fun_broadcast(
    index_set: Sequence[PPKey],
    size_fun: Callable[[int], int],
        iteration_num: int) -> ProjectionParameter:
    """Return size_fun evaluated at iteration_num over index_set."""
    return ProjectionParameter({idx: size_fun(iteration_num) for idx in index_set})


def plain_parameter_algorithm(
    iteration_num: int,
    distr_coll: Union[ReturnDistributionFunction, RewardDistributionCollection],
    mdp: MDP,
    index_set: Sequence[PPKey],
    size_fun: Callable[[int], int],  # in paper this is M: N -> N
    width_of_mesh: Callable[[int], float],
        z: float) -> ProjectionParameter:
    """Return Grid value parameter implementing the Plain parameter algorithm.

    Plain Parameter algorith as defind in paper
    'On Policy Evaluation Algorithms in Distributional Reinforcement Learning'.
    """

    assert distr_coll.index_set == index_set, "Index Set mismatch."
    m: int = size_fun(iteration_num)
    w: float = width_of_mesh(iteration_num)

    xs: np.ndarray = 2 * np.arange(0, m) - m
    xs = (xs / (m-1)) * w
    xs += z

    ys: np.ndarray = 2 * np.arange(1, m+1) - m
    ys = (ys / (m-1)) * w
    ys += z
    grid_value_projection_param: np.ndarray = np.concatenate([xs, ys])

    return ProjectionParameter(
        {idx: grid_value_projection_param for idx in index_set}
    )


def adaptive_ppa() -> ProjectionParameter:
    # TODO
    return ProjectionParameter({})


def qsp() -> ProjectionParameter:
    # TODO
    return ProjectionParameter({})
