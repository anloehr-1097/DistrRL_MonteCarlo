"""Preliminary tests for master thesis.

TODO: replace every indexing scheme with numerical indexing just based on the
    pointer of the element
    inkeeping with that, hold probs as a 2d array with rows summing to one,
    which might make code faster, check that
"""

from __future__ import annotations
from collections import OrderedDict
import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools
import functools
import numpy as np
# import scipy.stats as sp
from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen
from numba import njit

from .nb_fun import _sort_njit, _qf_njit
from .utils import assert_probs_distr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEBUG: bool = False
MAX_TRAJ_LEN: int = 1000
TERMINAL: int = -1
NUMBA_SUPPORT: bool = True
NUM_PRECISION_DECIMALS: int = 20
MAX_ITERS: int = 100000


@dataclass(frozen=True)
class State:
    """State representation. Once created, immutable."""

    state: int
    name: str
    index: int
    is_terminal: bool = False

    def __int__(self) -> int:
        """Return state."""
        return self.state


@dataclass(frozen=True)
class Action:
    """Action representation. Once created immutable."""
    action: int
    name: str
    index: int

    def __int__(self) -> int:
        """Return state."""
        return self.action


class Policy:
    """Random policy."""

    def __init__(
        self, states: Sequence[State], actions: Sequence[Action],
        probs: Dict[State, np.ndarray]
    ):
        """Initialize policy.

        Args:
            states: List of States
            actions: List of Actions
            probs: Mapping of state, distribution pairs. Each distribution is a numpy array.
                each entry in the numpy array corresponds to the probability of the action at the same index.
        """
        for state in states:
            assert probs[state].size == len(actions), \
                "action - distr mismatch len mismatch.\
                    Check if every state is a distribution over actions."
        self.states: Sequence[State] = states
        self.actions: Sequence[Action] = actions
        self.action_indices: np.ndarray = np.asarray(
            [action.index for action in self.actions]
        )
        # Caution: action indices must align with order of probs

        # self.actions: np.ndarray[Action] = np.asarray([actions.index for action in actions])  # 1-d
        self.probs: Dict[State, np.ndarray] = probs

    def __getitem__(self, state: State) -> np.ndarray:
        """Return distribution over actions for given state."""
        return self.probs[state]

    def sample_action(self, state: State) -> Action:
        """Sample action from pi(state)."""
        action_index: int = np.random.choice(
            self.action_indices,
            p=self.probs[state]
        )
        return self.actions[action_index]


class TransitionKernel:
    """Transition kernel for MDP.

    Args:
    states: Sequence of states
    actions: Sequence of actions
    probs: Mapping of state, action pairs to probs to next states.
        Make sure that the order of states mathces the order of states
        in the states sequence.
    """

    def __init__(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        probs: Dict[Tuple[State, Action], np.ndarray]
    ):
        """Initialize transition kernel."""
        for state in states:
            for action in actions:
                assert probs[(state, action)].size == len(
                    states
                ), "state - distr mismatch."

        self.states: Sequence[State] = states
        self.actions: Sequence[Action] = actions
        self.state_action_probs: Dict[Tuple[State, Action], np.ndarray] = probs

    def __getitem__(self, key: Tuple[State, Action]) -> np.ndarray:
        """Return distribution over actions for given state, action pair."""
        return self.state_action_probs[key]


class RV:
    """Random variable base class used for subclassing."""

    def __init__(self, xk, pk) -> None:
        """Initialize discrete random variable."""
        assert isinstance(xk, np.ndarray) and isinstance(pk, np.ndarray), \
            "Not numpy arrays upon creation of RV_Discrete."
        assert xk.size == pk.size, "Size mismatch in xk and pk."

        self.xk: np.ndarray
        self.pk: np.ndarray

        # make sure x has unique values
        self.xk, self.pk = aggregate_conv_results((xk, pk))
        self.is_sorted: bool = False
        self.size = self.xk.size

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
            return np.vectorize(self.qf)(u)


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
            return np.vectorize(self.qf)(u)


class ReturnDistributionFunction:
    """Return Distribution Function to a given policy."""

    # TODO instead of using tuple directly, use RV_Discrete

    def __init__(self, states: Sequence[State],
                 distributions: Optional[Sequence[DiscreteRV]]) -> None:
        """Initialize collection of categorical distributions."""
        self.states: Sequence[State] = states
        if distributions:
            self.distr: Dict = {s: distributions[i] for i, s in enumerate(states)}
        else:
            self.distr: Dict = {s: None for s in states}

    def __getitem__(self, state: Union[State, int]) -> DiscreteRV:
        """Return distribution for state."""
        if isinstance(state, int):
            target_state = list(filter(lambda s: s.index == state, self.states))[0]
            return self.distr[target_state]
        else:
            return self.distr[state]

    def __len__(self) -> int:
        return len(self.states)

    def __setitem__(self, key: int, value: DiscreteRV) -> None:
        """Set distribution for state."""
        self.distr[key] = value


class RewardDistributionCollection:
    """Return Distribution with finite support."""
    # TODO generalize to arbitrary support

    def __init__(
        self,
            state_action_state_triples: List[Tuple[State, Action, State]],  # (s,a,s')
            distributions: List[RV],
    ) -> None:
        """Initialize return distribution."""
        assert len(state_action_state_triples) == len(distributions), \
            "Mismatch in number of triples and and number of distributions."
        self.rewards: Dict[Tuple[State, Action, State], RV] = {
            s: d for s, d in zip(state_action_state_triples, distributions)
        }

    def __getitem__(self, key: Tuple[State, Action, State]) -> RV:
        """Return RV_Discrete for (state, action, next_state) triple."""
        return self.rewards[key]


class MDP:
    """Markov decision process."""

    def __init__(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        rewards: RewardDistributionCollection,
        transition_probs: TransitionKernel,
        gamma: np.float64 = np.float64(0.5),
    ):
        """Initialize MDP."""
        self.states: Sequence[State] = states
        self.actions: Sequence[Action] = actions
        self.rewards: RewardDistributionCollection = rewards
        self.transition_probs: TransitionKernel = transition_probs
        self.current_policy: Optional[Policy] = None
        self.gamma: np.float64 = gamma

    def set_policy(self, policy: Policy) -> None:
        """Set policy."""
        self.current_policy = policy

    def sample_next_state_reward(self, state: State, action: Action) -> \
            Tuple[State, np.float64]:
        """Sample next state and reward."""
        if state.is_terminal: return (TERMINAL_STATE, np.float64(0.0))
        next_state_probs: np.ndarray = self.transition_probs[(state, action)]
        next_state: State = np.random.choice(np.asarray(self.states), p=next_state_probs)
        reward: np.float64 = self.rewards[(state, action, next_state)]()[0]  # __call__ -> np.ndarray, thus pick first element
        return next_state, reward

    def check_if_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state.is_terminal

    def generate_random_policy(self) -> Policy:
        """Generate random policy which may be used in the given MDP."""

        probs: Dict[State, np.ndarray] = {}
        for state in self.states:
            ps: np.ndarray = np.random.random(len(self.actions))
            probs[state] = ps / np.sum(ps)

        return Policy(self.states, self.actions, probs)


class Trajectory:
    """Trajectory is list of tuples (state, action, next_state, reward)."""

    def __init__(self) -> None:
        """Initialize history."""
        self.history: List[Tuple[State, Action, State, np.float64]] = []

    def write(self, state: State, action: Action, next_state: State, reward: np.float64) -> \
            None:
        """Write to history."""
        self.history.append((state, action, next_state, reward))

    def aggregate_returns(self, gamma: np.float64) -> np.float64:
        """Aggregate returns."""
        returns: np.ndarray = np.asarray(self._get_returns())
        exponentiator: np.ndarray = np.arange(returns.size)
        gamma_ar: np.ndarray = np.ones(returns.size) * gamma
        gamma_ar = np.power(gamma_ar, exponentiator)
        returns = returns * gamma_ar
        return np.sum(returns)

    def _get_returns(self) -> List[np.float64]:
        """Return list of returns."""
        return [t[-1] for t in self.history]


TERMINAL_STATE: State = State(
    state=-1,
    name="TERMINAL",
    is_terminal=True,
    index=-1
)


PPComponent = Union[int, np.float64, np.ndarray]
PPKey = Union[Tuple[State, Action, State], State]

@dataclass
class ProjectionParameter:
    # index_set: Union[List[Tuple[State, Action, State]], List[State]]
    value: Dict[PPKey, PPComponent]

    def __getitem__(self, idx: PPKey) -> PPComponent:
        return self.value[idx]


# TODO this could be implemented with typing.Protocol
ParamAlgo = Callable[
    [int,  # iteration
     ReturnDistributionFunction,  # return distribution function est.
     RewardDistributionCollection,  # reward distribution coll est.
     MDP,  # mdp
     List[Tuple[State, Action, State]],  # inner index set
     List[State]  # outer index set
     ],
    Tuple[ProjectionParameter, ProjectionParameter]]  # proj params


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


###################################
# Algorithm 5.1  - classical DBO  #
###################################
def dbo(mdp: MDP, ret_distr_function: ReturnDistributionFunction,
        reward_distr_coll: Optional[RewardDistributionCollection]=None) -> None:
    """Single application of the DBO.

    If rewards_distr is None, then use the rewards from the MDP.
    This corresponds to the original DBO. If rewards_distr is given, then
    use the given reward distribution. This corresponds to the extended DBO.
    """

    reward_distr_coll = reward_distr_coll if reward_distr_coll else mdp.rewards
    if mdp.current_policy is None:
        raise ValueError("No policy set for MDP.")

    eta_next = dict()
    for state, distr in ret_distr_function.distr.items():
        new_vals: List = []
        new_probs: List = []

        for action in mdp.actions:
            for next_state in mdp.states:
                transition_prob = mdp.transition_probs[(state, action)][next_state.index]
                prob = mdp.current_policy[state][action.index] * transition_prob
                if prob == 0:
                    continue
                reward_distr = reward_distr_coll[(state, action, next_state)]
                assert isinstance(reward_distr, DiscreteRV), "For application of the DBO, rewards need to be discrete."
                if next_state.is_terminal:
                    new_vals.append(reward_distr.xk)
                    new_probs.append(reward_distr.pk * prob)
                else:
                    distr_update = conv_njit(
                        scale(ret_distr_function[next_state], mdp.gamma).distr(),
                        reward_distr.distr()
                    )
                    # scale(ret_distr_function[next_state], mdp.gamma),
                    # reward_distr)
                    new_vals.append(distr_update[0])
                    new_probs.append(distr_update[1] * prob)

        eta_next[state] = aggregate_conv_results(
            (np.concatenate(new_vals), np.concatenate(new_probs))
        )
        eta_next[state] = DiscreteRV(*eta_next[state])

    for state, distr in eta_next.items():
        ret_distr_function[state] = distr


def ddp(
    mdp: MDP, inner_projection: Projection,
    outer_projection: Projection,
    param_algorithm: ParamAlgo,
    return_distr_function: ReturnDistributionFunction,
        iteration_num: int) -> ReturnDistributionFunction:
    """1 Step of Distributional dynamic programming in iteration iteration_num.

    Carry out one step of distributional dynamic programming.
    """

    inner_params: ProjectionParameter
    outer_params: ProjectionParameter
    # apply inner projection
    inner_params, outer_params = param_algorithm(
        iteration_num,
        return_distr_function,
        mdp.rewards,
        mdp,
        list(itertools.product(mdp.states, mdp.actions, mdp.states)),
        mdp.states,  # type: ignore
    )

    if DEBUG: logger.info(f" Inner & outer params: {inner_params, outer_params}")

    rewards_distr_coll = RewardDistributionCollection(
        list(mdp.rewards.rewards.keys()),
        [inner_projection(mdp.rewards[(s, a, s_bar)], inner_params[(s, a, s_bar)]) for
         (s, a, s_bar) in
         itertools.product(mdp.states, mdp.actions, mdp.states)
         if mdp.transition_probs[(s, a)][s_bar.index] > 0
         ]
    )
    # apply step of dbo

    dbo(mdp, return_distr_function, rewards_distr_coll)
    # apply outer projection
    return_distr_iterate: ReturnDistributionFunction = \
        ReturnDistributionFunction(
            return_distr_function.states,
            [outer_projection(return_distr_function[s], outer_params[s]) for
                s in return_distr_function.states]
        )
    return return_distr_iterate


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


def poly_size_fun(x: int) -> PPComponent: return x**2
def exp_size_fun(x: int) -> PPComponent: return 2**x
def poly_decay(x: int) -> float: return 1/(x**2)
def exp_decay(x: int) -> float: return 1/2**(x)


class SizeFun(Enum):
    POLY: Callable[[int], PPComponent] = poly_size_fun
    EXP: Callable[[int], PPComponent] = exp_size_fun
    POLY_DECAY: Callable[[int], float] = poly_decay
    EXP_DECAY: Callable[[int], float] = exp_decay

# iteration -> evaluated size functions as parameters
# quant_projection_algo: Callable[[int, List[Tuple[State, Action, State]], List[State]], Tuple[ProjectionParameter, ProjectionParameter]] = \
#     functools.partial(
#         algo_size_fun, inner_size_fun=poly_size_fun,
#         outer_size_fun=poly_size_fun,
#         previous_return_estimate=None,
#         previous_reward_estimate=None
#     )


def quant_projection_algo(
    iteration: int,
    ret_distr_fun: ReturnDistributionFunction,
    rew_distr_coll: RewardDistributionCollection,
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
    previous_reward_estimate: RewardDistributionCollection,
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
    reward_approx: RewardDistributionCollection,
    mdp: MDP,
    inner_index_set: List[Tuple[State, Action, State]],
    outer_index_set: List[State],
        ) -> Tuple[ProjectionParameter, ProjectionParameter]:

    decay_funs: Tuple[Callable, Callable, Callable] = (SizeFun.POLY_DECAY, SizeFun.EXP_DECAY, SizeFun.EXP_DECAY)
    size_funs: Tuple[Callable, Callable] = (SizeFun.POLY, SizeFun.EXP)

    inner_param: ProjectionParameter = algo_cdf_1(
        inner_index_set=inner_index_set,
        previous_reward_estimate=reward_approx,
        mdp=mdp,
        f_min=decay_funs[0],
        f_max=decay_funs[1],
        f_inter=decay_funs[2])

    outer_param: ProjectionParameter = algo_size_fun(
        iteration, inner_index_set, outer_index_set,
        *size_funs[:2])[-1]

    return inner_param, outer_param


def random_projection(num_samples: int, rv: RV) -> RV:
    """Random projection of distribution."""

    atoms = rv.sample(num_samples)
    weights = np.ones(num_samples) / num_samples
    return DiscreteRV(atoms, weights)


class RandomProjection(Projection):
    """Random Projection"""

    def project(self, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        assert isinstance(projection_param, int), \
            "Random Projection expects int parameter."

        atoms = rv.sample(projection_param)
        weights = np.ones(projection_param) / projection_param
        return DiscreteRV(atoms, weights)


class QuantileProjection(Projection):
    """Quantile projection."""

    def project(self, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        """Apply quantile projection."""
        assert isinstance(projection_param, int), \
            "Quantile Projection expects int parameter."
        return quantile_projection(rv, projection_param)


# @njit
def quantile_projection(rv: RV,
                        no_quantiles: int) -> DiscreteRV:
    """Apply quantile projection as described in book to a distribution."""
    if rv.size <= no_quantiles and isinstance(rv, DiscreteRV): return rv  # type: ignore
    quantile_locs: np.ndarray = (2 * (np.arange(1, no_quantiles + 1)) - 1) /\
        (2 * no_quantiles)
    quantiles_at_locs = rv.qf(quantile_locs)
    return DiscreteRV(quantiles_at_locs, np.ones(no_quantiles) / no_quantiles)


q_proj: QuantileProjection = QuantileProjection()


class GridValueProjection(Projection):
    """Grid Value Projection."""

    def project(self, rv: RV, projection_param: PPComponent) -> DiscreteRV:
        assert isinstance(projection_param, np.ndarray), \
            "Grid Value Projection expects numpy ndarray parameter."

        assert projection_param.size % 2 == 1, \
            "projection param must be of size 2m - 1 where m in |N."

        return grid_value_projection(rv, projection_param)


def grid_value_projection(rv: RV, projection_param: np.ndarray) -> DiscreteRV:
    """Grid value projection."""
    param_size: int = (projection_param.size // 2) + 1
    xs: np.ndarray = projection_param[:param_size]
    ys: np.ndarray = projection_param[param_size:]
    y_evals: np.ndarray = rv.cdf(ys)
    pk: np.ndarray = np.concatenate([y_evals, np.asarray([1])]) - \
        np.concatenate([np.asarray([0]), y_evals])
    return DiscreteRV(xs, pk)


def grid_value_algo2():
    pass


def categorical_projection(rv: DiscreteRV,
                           particles: np.ndarray)\
                           -> Tuple[np.ndarray, np.ndarray]:
    """Apply categorical projection as described in book to a distribution."""

    distr = rv.distr()
    # sort array
    hypo_insert_pos: np.ndarray = np.searchsorted(particles, distr[0])
    if DEBUG:
        print(f"Hypo insert pos: {hypo_insert_pos}")

    # determine closest neighbors
    left_neigh_index: np.ndarray = hypo_insert_pos - 1
    right_neigh_index: np.ndarray = hypo_insert_pos

    new_probs: np.ndarray = np.zeros(particles.size)

    assert left_neigh_index.size == right_neigh_index.size, \
        "Size mismatch in neighbors."
    # left_neigh: np.ndarray = particles[hypo_insert_pos - 1]
    # right_neigh: np.ndarray = particles[hypo_insert_pos]
    # determine weights to assign to neighbors
    # left_weight: np.ndarray = (right_neigh - distr[0]) / (right_neigh - left_neigh)
    # right_weight: np.ndarray = 1 - left_weight

    for i in range(left_neigh_index.size):
        if (left_neigh_index[i] == -1) and (distr[0][i] < particles[0]):
            # to the left of first particle -> left particle gets all of the mass
            new_probs[0] += distr[1][i]

        elif (right_neigh_index[i] == particles.size) and (distr[0][i] > particles[-1]):
            # to the right of last particle -> right particle gets all of the mass
            new_probs[-1] += distr[1][i]

        else:
            # assign mass to neighbors according to distance to neighbors
            left_mass_i: np.float64 = \
                1 - np.abs(particles[left_neigh_index[i]] - distr[0][i])

            right_mass_i: np.float64 = 1 - left_mass_i
            new_probs[left_neigh_index[i]] += left_mass_i * distr[1][i]
            new_probs[right_neigh_index[i]] += right_mass_i * distr[1][i]

    if DEBUG:
        print(f"Particles: {particles}")
        print(f"Probs: {new_probs}")
    return (particles, new_probs)




# TODO: possible enable @njit
# @njit
# def quantile_projection(rv: RV,
#                         no_quantiles: int) -> RV:
#     """Apply quantile projection as described in book to a distribution.
#
#     Assume that atoms in distribution are sorted in ascending order.
#     Assume that unique values in RV distribution, i.e. equal values are aggregated.
#     """
#     vals: np.ndarray = rv.xk
#     probs: np.ndarray = rv.pk
#
#     if probs.size < no_quantiles: return rv
#
#     # aggregate probs
#     cum_probs: np.ndarray = np.cumsum(probs)
#     locs: np.ndarray = (2 * (np.arange(1, no_quantiles + 1)) - 1) / (2 * no_quantiles)
#     quantiles_at_locs: np.ndarray = np.searchsorted(locs, cum_probs) + 1  # TODO double check this
#     values_at_quantiles: np.ndarray = vals[quantiles_at_locs]
#     # quantiles: np.ndarray = np.cumsum(np.ones(no_quantiles) / no_quantiles)
#     # assert np.isclose(quantiles[-1], 1), "Quantiles do not sum to 1."
#     # # determine indices for quantiles
#     # quantile_locs: np.ndarray = np.searchsorted(cum_probs, quantiles)
#     # quantile_locs = np.clip(quantile_locs, 0, len(vals) - 1)
#     # return quantile_locs, (np.ones(no_of_bins) / no_of_bins)
#     # return vals[quantile_locs], (np.ones(no_of_bins) / no_of_bins)
#     return RV(values_at_quantiles, np.ones(no_quantiles) / no_quantiles)
#

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


def time_it(debug: bool) -> Callable:
    """Time function.

    Print time for func call if debug is True.
    """

    def time_it_dec(func: Callable) -> Callable:
        if debug:

            def time_it_inner(*args, **kwargs):
                start: float = time.time()
                ret_val = func(*args, **kwargs)
                end: float = time.time()
                if "time_step" in dict(**kwargs).keys():
                    print(
                        f"Time taken for time step {kwargs['time_step']}: {end-start}"
                    )
                else:
                    print(f"Time taken: {end-start}")
                return ret_val

            return time_it_inner

        else:
            return func

    return time_it_dec


@njit
def conv_njit(
        a: Tuple[np.ndarray, np.ndarray],
        b: Tuple[np.ndarray, np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    Numba JIT compiled version.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


@njit
def aggregate_conv_results(distr: Tuple[np.ndarray, np.ndarray], accuracy: float=1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate results of convolution.
    Sum up probabilities of same values.
    """

    val_sorted_indices: np.ndarray = np.argsort(distr[0])  # n log n
    val_sorted: np.ndarray = distr[0][val_sorted_indices]
    probs_sorted: np.ndarray = distr[1][val_sorted_indices]

    ret_dist_v: List = []
    ret_dist_p: List = []
    current: int = 0
    # i: int = 1
    # ret_dist_v.append(val_sorted[current])
    # ret_dist_p.append(probs_sorted[current])

    for i in range(1, distr[0].size):
        # if np.abs(val_sorted[i] - val_sorted[i - 1]) < accuracy:
        if np.abs(val_sorted[i] - val_sorted[current]) < accuracy:
            probs_sorted[current] += probs_sorted[i]
        else:
            ret_dist_v.append(val_sorted[current])
            ret_dist_p.append(probs_sorted[current])
            current = i

            # ret_dist_v.append(val_sorted[i])
            # ret_dist_p.append(probs_sorted[i])
            # current = i

    ret_dist_v.append(val_sorted[current])
    ret_dist_p.append(probs_sorted[current])

    return np.asarray(ret_dist_v), np.asarray(ret_dist_p)


def monte_carlo_eval(mdp: MDP, num_trajectories: int=20,
                     trajectory_len: int=-1) -> ReturnDistributionFunction:
    """Monte Carlo Simulation with fixed policy.

    Run num_trajectories many simulations for each state with each trajectory
    running for num_epochs.

    Return the return distribution function estimate.
    """
    # create Dict[state, est_return_distr]
    est_return_distr_fun: ReturnDistributionFunction = ReturnDistributionFunction(mdp.states, None)
    traj_res_arary: Dict[State, List] = {i: [] for i in mdp.states}
    trajectories: Dict[State, Trajectory]

    for traj_no in range(num_trajectories):
        trajectories = \
            monte_carlo_trajectories(mdp, trajectory_len)
        for state in mdp.states:
            traj_res_arary[state].append(
                trajectories[state].aggregate_returns(mdp.gamma)
            )

    for state in mdp.states:
        # create distribution from trajectory results
        est_return_distr_fun.distr[state] = RV(
            np.asarray(traj_res_arary[state]),
            np.ones(num_trajectories) / num_trajectories)
    return est_return_distr_fun


def monte_carlo_trajectories(mdp: MDP, trajectory_len: int=-1) -> \
        Dict[State, Trajectory]:
    """Monte Carlo Simulation with fixed policy.

    Given mdp, policy, num epochs, return one sample trajectory for each state.
    Single trajectory runs for trajectory_len if trajectory_len > 0.
    Else runs until MAX_TRAJ_LEN is reached.
    """
    assert mdp.current_policy, "No policy set for MDP."
    trajectories: Dict[State, Trajectory] = {}
    for state in mdp.states:
        if DEBUG:
            print(f"Monte Carlo evaluation for state: {state}")
        trajectory: Trajectory = monte_carlo_eval_single_trajectory(
            mdp, state, trajectory_len)
        trajectories[state] = trajectory
    return trajectories


def monte_carlo_eval_single_trajectory(
        mdp: MDP, state: State, trajectory_length: int=-1) -> Trajectory:
    """Monte Carlo Simulation with a fixed policy and initial state."""
    trajectory: Trajectory = Trajectory()
    tlen: int = 0
    while True:
        if DEBUG:
            print(f"Epoch: {tlen+1}")
        state = one_step_monte_carlo(mdp, state, trajectory)
        tlen += 1
        if ((trajectory_length != -1) and (tlen >= trajectory_length)) \
           or (tlen > MAX_TRAJ_LEN):
            break
    return trajectory


def one_step_monte_carlo(
        mdp: MDP,
        state: State,
        trajectory: Trajectory) -> State:
    """One step of Monte Carlo Simulation.

    Run one monte carlo step, write to trajectory history & return next state.
    """
    next_state: State
    reward: np.float64
    # if not mdp.current_policy:
    #     logger.info("No policy set for MDP. Setting random policy.")
    #     mdp.generate_random_policy()
    assert mdp.current_policy, "No policy set for MDP."
    action: Action = mdp.current_policy.sample_action(state)
    next_state, reward = mdp.sample_next_state_reward(state, action)
    trajectory.write(state, action, next_state, reward)
    return next_state


def main():
    """Call main function."""
    from .sample_envs import cyclical_env
    # mdp = cyclical_env.mdp
    # policy = mdp.generate_random_policy()
    ddp(
        cyclical_env.mdp,
        QuantileProjection(),
        QuantileProjection(),
        quant_projection_algo,
        cyclical_env.return_distr_fun_est,
        2)
    return None


def wasserstein_beta(
    rv1: Union[DiscreteRV, ContinuousRV],
    rv2: Union[DiscreteRV, ContinuousRV],
        beta: float=1) -> float:
    """Wasserstein beta distance between two distributions.

    Assume that beta-moment exist, not checked here.
    """

    # if not discrete, make discrete
    if isinstance(rv1, ContinuousRV):
        rv1 = rv1.empirical()
    if isinstance(rv2, ContinuousRV):
        rv2 = rv2.empirical()
    common_support: np.ndarray = np.concatenate([rv1.xk, rv2.xk])
    common_support = np.sort(np.unique(common_support))
    cdf_rv1: np.ndarray = rv1.cdf(common_support[:-1])
    cdf_rv2: np.ndarray = rv2.cdf(common_support[:-1])
    diffs: np.ndarray = np.abs(cdf_rv1 - cdf_rv2)**beta
    weights = np.diff(common_support)
    return float(np.sum(weights * diffs))  # type: ignore


def extended_metric(metric: Callable[[DiscreteRV, DiscreteRV, float], float],
                    rv1s: Dict[Tuple[State, Action, State], DiscreteRV],
                    rv2s: Dict[Tuple[State, Action, State], DiscreteRV],
                    beta: float=1) -> float:
    """Extended metric for Wasserstein beta distance."""

    assert rv1s.keys() == rv2s.keys(), "Keys of distributions do not match."
    metric_evals: np.ndarray = np.asarray([metric(rv1s[key], rv2s[key], beta) for key in rv1s.keys()])  # type of elements: float
    return np.max(metric_evals)


if __name__ == "__main__":

    main()
