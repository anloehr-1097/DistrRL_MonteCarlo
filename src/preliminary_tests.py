"""Preliminary tests for master thesis."""

from __future__ import annotations
# import numpy as np


import random
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from numba import njit, jit

from .nb_fun import _sort_njit
from .utils import assert_probs_distr

# States: Dict[int, AnyType] holding possibly holding state representation as vector
# Actions: Dict[int, Tuple[List[int], List[float]]] holding possible actions, probablity pairs for each state


STATES: Mapping = {1: 1, 2: 2, 3: 3}  # states, potentially more complex this mapping
ACTIONS: Mapping = {1: ([2], [1.0]), 2: ([3], [1.0]), 3: ([1], [1.0])}
REWARDS: Mapping = {
    (1, 2): ([-3, 1], [0.5, 0.5]),
    (2, 3): ([5, 2], [0.5, 0.5]),
    (3, 1): ([0, 0.5], [0.5, 0.5]),
}


DEBUG = True
MAX_EPOCHS: int = 1000
TERMINAL: int = -1
NUMBA_SUPPORT: bool = True

# need some initial collection of distrs for the total reward =: nu^0
# need some return distributons
# need samples from return distr for each triple (s, a, s') (N * |S| * |A| * |S|) many)
# need Bellman operator
# need some policy (fixed), define state - action dynamics
# need some distributions such that everything is known and can be compared
# define random bellman operator as partial function


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
        self.states: Sequence[State] = states
        self.actions: np.ndarray = np.array(actions)  # 1-d
        for state in states:
            assert probs[state].size == len(actions), \
                "action - distr mismatch len mismatch.\
                    Check if every state is a distribution over actions."
        self.probs: Dict[State, np.ndarray] = probs

    def __getitem__(self, state: State) -> np.ndarray:
        """Return distribution over actions for given state."""
        return self.probs[state]

    def sample_action(self, state: State) -> int:
        action = np.random.choice(self.actions, p=self.probs[state])
        return action


class TransitionKernel:
    """Transition kernel for MDP."""

    # TODO: needs work, transiton kernel should yield probs (x, a) -> x'
    def __init__(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        probs: Dict[Tuple[State, Action], np.ndarray],  # ensure same order as states
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


class GeneralRV:
    # implement wrapper for scipy RV
    # TODO which methods have to be supported?
    pass


class RV:
    """Discrete atomic random variable."""

    def __init__(self, xk, pk) -> None:
        """Initialize discrete random variable."""
        assert isinstance(xk, np.ndarray) and isinstance(pk, np.ndarray), \
            "Not numpy arrays upon creation of RV_Discrete."
        self.xk: np.ndarray = xk
        self.pk: np.ndarray = pk

    def distr(self):
        """Return distribution as Tuple of numpy arrays."""
        return self.xk, self.pk

    def get_cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        self._sort_njit() if NUMBA_SUPPORT else self._sort()
        return self.xk, np.cumsum(self.pk)

    def _sort(self):
        """Sort values and probs."""
        indices = np.argsort(self.xk)
        self.xk = self.xk[indices]
        self.pk = self.pk[indices]

    def _sort_njit(self):
        self.xk, self.pk = _sort_njit(self.xk, self.pk)

    def sample(self) -> float:
        """Sample from distribution."""
        return np.random.choice(self.xk, p=self.pk)

    def __call__(self) -> float:
        """Sample from distribution."""
        return self.sample()


class ReturnDistributionFunction:
    """Return Distribution Function to a given policy."""

    # TODO instead of using tuple directly, use RV_Discrete

    def __init__(self, states: Sequence[State],
                 distributions: Sequence[RV]) -> None:
        """Initialize collection of categorical distributions."""
        self.states: Sequence[State] = states
        self.distr: Dict = {s: distributions[i] for i, s in enumerate(states)}

    def __getitem__(self, state: State) -> RV:
        """Return distribution for state."""
        return self.distr[state]

    def __len__(self) -> int:
        return len(self.states)

    def __setitem__(self, key: int, value: RV) -> None:
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
            Tuple[State, float]:
        """Sample next state and reward."""
        if state.is_terminal: return (TERMINAL_STATE, 0.0)
        next_state_probs: np.ndarray = self.trasition_probs[(state, action)]
        next_state: State = np.random.choice(np.asarray(self.states), p=next_state_probs)
        reward: float = self.rewards[(state, action, next_state)]()
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
        self.history: List[Tuple[State, Action, State, float]] = []

    def write(self, state: State, action: Action, next_state: State, reward: float) -> \
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

    def _get_returns(self) -> List[float]:
        """Return list of returns."""
        return [t[-1] for t in self.history]


TERMINAL_STATE: State = State(
    state=-1,
    name="TERMINAL",
    is_terminal=True,
    index=-1
)


####################
# Algorithm 5.1    #
####################
def dbo(mdp: MDP, ret_distr_function: ReturnDistributionFunction) -> None:
    """Single step DDP."""
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
                reward_distr = mdp.rewards[(state, action, next_state)]
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
        eta_next[state] = RV(*eta_next[state])

    for state, distr in eta_next.items():
        ret_distr_function[state] = distr


def random_projection(rv: RV, num_samples: int) -> RV:
    return rv


def categorical_projection(rv: RV) -> RV:
    return rv

####################
# Algorithm 5.3    #
####################
# TODO this is not used anymore but some pieces might be used
def categorical_dynamic_programming(mdp: MDP,
                                    pi: Policy,
                                    cat_distr_col: ReturnDistributionFunction,
                                    particles: np.ndarray)\
                                    -> ReturnDistributionFunction:

    """Categorical dynamic programming.
    Execute one step of the categorical dynamic programming algorithm.

    An implementation of Algorithm 5.3 from the DRL book.
    This algorithm applies the categorical projection after computing the convolution.
    For a fixed number of particles m, the algorithm projects an arbitrary distribution
    d onto a distribution with m atoms at the same location and ajusts the probabilities.
    """
    # assume cat_dist_col already categorical representation of (\theta_i, p_i)
    assert assert_equidistant_particles(particles) is False, \
        "Check initial particles, not equidistant."

    # ensure sorted particles
    particles = np.sort(particles)

    # apply categorical projection to initial distribution
    for idx, state in enumerate(cat_distr_col.states):
        cat_distr_col[idx] = RV(
            *categorical_projection(cat_distr_col[state].distr(),
                                    particles))

    # apply algo 5.1
    dbo_result: ReturnDistributionFunction = categorical_dbo(mdp, pi,
                                                             cat_distr_col)

    # for each state, apply quantalie projection
    for idx, state in enumerate(dbo_result.states):
        dbo_result[state] = RV(
            *categorical_projection(dbo_result[state].distr(), particles))
    return dbo_result


def assert_equidistant_particles(particles: np.ndarray) -> np.bool_:
    """Assert that particles are equidistant."""
    return np.all(np.diff(particles) ==
                  (particles[-1] - particles[0])/(particles.size - 1))


def categorical_projection(rv: RV,
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


####################
# Algorithm 5.4    #
####################
def quantile_dynamic_programming(
    mdp: MDP, pi: Policy,
    cat_distr_col: ReturnDistributionFunction,
        no_of_quantiles: int) -> ReturnDistributionFunction:
    r"""Quantile dynamic programming.

    Execute one step of the quantile dynamic programming algorithm.

    An implementation of Algorithm 5.4 from the DRL book.
    This algorithm applies the quantile projection after computing the convolution.
    For a fixed number of particles m, the algorithm projects an arbitrary distribution
    d onto a distribution with m atoms with equal probability. The i-th location / atom
    \\theta_i is obtained by calculating F^{-1}(2*i - 1 / 2m) where F^{-1} is the QF of d.
    """
    # get the initial quantile projection of cat_distr_col before any dbo application
    for idx, state in enumerate(cat_distr_col.states):
        cat_distr_col[idx] = RV(
            *quantile_projection(cat_distr_col[state].distr(), no_of_quantiles))

    # apply algo 5.1
    dbo_result: ReturnDistributionFunction = categorical_dbo(mdp, pi, cat_distr_col)

    # for each state, apply quantalie projection
    for idx, state in enumerate(dbo_result.states):
        dbo_result[state] = RV(
            *quantile_projection(dbo_result[state].distr(), no_of_quantiles))
    return dbo_result


# TODO resolve error, numba does not compile correctly, get type error 
# @njit 
def filter_and_aggregate(vals: np.ndarray, probs: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Filter and aggregate unique values / probs in a distribution.

    Assume that values are in increasing order.
    """
    current_val_idx: int = -1
    current_val: Optional[np.float64] = np.float64(vals[0] - 1) 
    new_probs: List[np.float64] = []

    for idx, val in enumerate(vals):
        if abs(val - current_val) < 1e-10:
            new_probs[current_val_idx] += probs[idx]
        else:
            current_val = val
            new_probs.append(probs[idx])
            current_val_idx += 1

    new_vals: np.ndarray = np.unique(vals)
    return new_vals, np.asarray(new_probs)


# TODO: possible enable @njit
@njit
def quantile_projection(rv: RV,
                        no_quantiles: int) -> RV:
    """Apply quantile projection as described in book to a distribution.

    Assume that atoms in distribution are sorted in ascending order.
    Assume that unique values in RV distribution, i.e. equal values are aggregated.
    """

    vals: np.ndarray = rv.xk
    probs: np.ndarray = rv.pk

    if probs.size < no_quantiles: return rv

    # aggregate probs
    cum_probs: np.ndarray = np.cumsum(probs)
    locs: np.ndarray = (2 * (np.arange(1, no_quantiles + 1)) - 1) / (2 * no_quantiles)
    quantiles_at_locs: np.ndarray = np.searchsorted(locs, cum_probs) + 1  # TODO double check this
    values_at_quantiles: np.ndarray = vals[quantiles_at_locs]
    # quantiles: np.ndarray = np.cumsum(np.ones(no_quantiles) / no_quantiles)
    # assert np.isclose(quantiles[-1], 1), "Quantiles do not sum to 1."
    # # determine indices for quantiles
    # quantile_locs: np.ndarray = np.searchsorted(cum_probs, quantiles)
    # quantile_locs = np.clip(quantile_locs, 0, len(vals) - 1)
    # return quantile_locs, (np.ones(no_of_bins) / no_of_bins)
    # return vals[quantile_locs], (np.ones(no_of_bins) / no_of_bins)
    return RV(values_at_quantiles, np.ones(no_quantiles) / no_quantiles)


def scale(distr: RV, gamma: np.float64) -> RV:
    """Scale distribution by factor."""
    distr.xk *= gamma
    return distr


def conv(a: RV, b: RV) -> RV:
    """Convolution of two distributions.
    0th entry values, 1st entry probabilities.
    """
    new_val: np.ndarray = np.add(a.xk, b.xk[:, None]).flatten()
    probs: np.ndarray = np.multiply(a.pk, b.pk[:, None]).flatten()
    return RV(new_val, probs)


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
    i: int = 1
    ret_dist_v.append(val_sorted[current])
    ret_dist_p.append(probs_sorted[current])

    for i in range(1, distr[0].size):
        if np.abs(val_sorted[i] - val_sorted[i - 1]) < accuracy:
            probs_sorted[current] += probs_sorted[i]
        else:
            ret_dist_v.append(val_sorted[i])
            ret_dist_p.append(probs_sorted[i])
            current = i

    # return RV(np.asarray(ret_dist_v), np.asarray(ret_dist_p))
    return np.asarray(ret_dist_v), np.asarray(ret_dist_p)


def apply_projection(func: Callable) -> Callable:
    """Apply binning as returned from func.

    Assumes that kwargs["probs"] is in kwargs.
    """

    def apply_projection_inner(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        bins: np.ndarray
        no_of_bins: int
        bin_values: np.ndarray

        bins, bin_values, no_of_bins = func(*args, **kwargs)
        # do binning, stupidly
        # new_values: np.ndarray = np.zeros(no_of_bins)
        new_probs: np.ndarray = np.zeros(no_of_bins)

        for i in range(no_of_bins):
            new_probs[i] = np.sum(kwargs["probs"][bins == i])

        new_probs = new_probs / np.sum(new_probs)

        return bin_values, new_probs

    return apply_projection_inner


@apply_projection
@njit
def project_eqi(
    values: np.ndarray, probs: np.ndarray, no_of_bins: int, state: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Project equisdistantly. Return bins."""
    v_min: np.float64
    v_max: np.float64
    v_min, v_max = np.min(values), np.max(values)
    breadth: np.float64 = v_max - v_min
    breadth /= no_of_bins

    bin_values: np.ndarray = np.linspace(v_min - breadth, v_max + breadth, no_of_bins)
    assert bin_values.size == no_of_bins, "wrong number of bins."
    bins: np.ndarray = np.digitize(values, bin_values)
    return bins, bin_values, no_of_bins


def simulate_update(
    time_steps: int,
    num_samples: int,
    reward_distr: Tuple[np.ndarray, np.ndarray],
    proj_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate update of aggregate return distribution over time_steps periods.

    Return aggregate return distribution.
    """

    approx_list: List = []
    # start with some random intial aggregate return distribution
    num_elements_g0: int = np.random.randint(1, 100)
    rand_values: np.ndarray = np.random.random(num_elements_g0)

    g_0: Tuple[np.ndarray, np.ndarray] = (
        rand_values,
        rand_values / np.sum(rand_values),
    )
    print(f"g_0: {g_0}")
    approx_list.append(g_0)

    for t in range(time_steps):
        # g_t: Tuple[np.ndarray, np.ndarray] = conv_jit(emp_distr, approx_list[-1])
        # g_t = proj_func(values=g_t[0], probs=g_t[1], no_of_bins=(t + 1) * 10, state=1)
        # approx_list.append(g_t)
        g_t: Tuple[np.ndarray, np.ndarray] = simulate_one_step(
            reward_distr, approx_list[-1], time_step=t, proj_func=proj_func
        )

        approx_list.append(g_t)

    return approx_list[-1]


@time_it(DEBUG)
def simulate_one_step(
    dist1: Tuple[np.ndarray, np.ndarray],
    dist2: Tuple[np.ndarray, np.ndarray],
    time_step: int,
    proj_func: Callable,
) -> Tuple[np.ndarray, np.ndarray]:

    g_t: Tuple[np.ndarray, np.ndarray] = aggregate_conv_results(RV(conv_njit(dist1, dist2)[0], conv_njit(dist1, dist2)[1]))
    g_t = proj_func(
        values=g_t[0], probs=g_t[1], no_of_bins=(time_step + 1) * 10, state=1
    )
    return g_t


def monte_carlo_eval(mdp: MDP, policy: Policy, num_trajectories: int=20,
                     num_epochs: int=-1) -> Dict[int, RV]:
    """Monte Carlo Simulation with fixed policy.

    Run num_trajectories many simulations for each state with each trajectory
    running for num_epochs.

    Return the return distributions for each state obtained in this way.
    """
    # create Dict[state, est_return_distr]
    est_return_distr: Dict[int, RV] = {}
    traj_res_arary: Dict[int, List] = {i: [] for i in mdp.states.keys()}
    trajectories: Dict[int, Trajectory]
    for traj_no in range(num_trajectories):
        trajectories = \
            monte_carlo_trajectories(mdp, policy, num_epochs)
        for state in mdp.states.keys():
            traj_res_arary[state].append(
                trajectories[state].aggregate_returns(mdp.gamma)
            )

    for state in mdp.states.keys():
        # create distribution from trajectory results
        est_return_distr[state] = RV(
            np.asarray(traj_res_arary[state]),
            np.ones(num_trajectories) / num_trajectories)
    return est_return_distr


def monte_carlo_trajectories(mdp: MDP, policy: Policy, num_epochs: int=-1) -> \
        Dict[int, Trajectory]:
    """Monte Carlo Simulation with fixed policy.

    Given mdp, policy, num epochs, return sample trajectory for each state.
    Single trajectory runs for num_epochs if num_epochs > 0.
    Else runs until MAX_EPOCHS is reached.
    """
    mdp.set_policy(policy)
    trajectories: Dict[int, Trajectory] = {}

    for state in mdp.states.keys():
        if DEBUG:
            print(f"Monte Carlo evaluation for state: {state}")
        trajectory: Trajectory = monte_carlo_eval_single_trajectory(
            mdp, state, policy, num_epochs)
        trajectories[state] = trajectory
    return trajectories


def monte_carlo_eval_single_trajectory(mdp: MDP, state: int, policy: Policy,
                                   num_epochs: int=-1) -> Trajectory:
    """Monte Carlo Simulation with a fixed policy and initial state."""
    mdp.set_policy(policy)
    trajectory: Trajectory = Trajectory()

    epoch: int = 0
    while True:
        if DEBUG:
            print(f"Epoch: {epoch+1}")
        state = one_step_monte_carlo(mdp, state, trajectory)
        epoch += 1
        if ((num_epochs != -1) and (epoch >= num_epochs)) \
           or (epoch > MAX_EPOCHS):
            break

    return trajectory


def one_step_monte_carlo(mdp: MDP, state: int, trajectory: Trajectory) -> int:
    """One step of Monte Carlo Simulation.

    TODO:
    Return action, next state and reward, write to history
    """
    next_state: int
    reward: float
    policy: Policy = mdp.current_policy if mdp.current_policy is not None \
        else mdp.generate_random_policy()
    action: int = policy.sample_action(state)
    next_state, reward = mdp.sample_next_state_reward(state, action)
    trajectory.write(state, action, next_state, reward)

    return next_state


def main():
    """Call main function."""
    from .sample_envs import cyclical_env
    mdp = cyclical_env.mdp
    policy = mdp.generate_random_policy()

    approx_distr_mc: Dict[int, RV] = monte_carlo_eval(
        mdp, policy, 10)

    return None
    # res = cyclical_env.total_reward_distr_estimate
    # for i in range(1000):
    # print(f"Iteration {i}")
    # res = quantile_dynamic_programming(mdp, mdp.current_policy, res, 100)
    # return res


if __name__ == "__main__":

    main()
