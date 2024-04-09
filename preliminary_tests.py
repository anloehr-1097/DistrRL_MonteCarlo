"""Preliminary tests for master thesis."""

# import numpy as np


import random
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from numba import njit

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
# need some initial collection of distrs for the total reward =: nu^0
# need some return distributons
# need samples from return distr for each triple (s, a, s') (N * |S| * |A| * |S|) many)
# need Bellman operator
# need some policy (fixed), define state - action dynamics
# need some distributions such that everything is known and can be compared
# define random bellman operator as partial function


class Policy:
    def __init__(
        self, states: Mapping, actions: Sequence, probs: Dict[int, np.ndarray]
    ):
        """Initialize policy.

        Args:
            states: Mapping of states
            actions: Sequence of actions
            probs: Mapping of state, distribution pairs. Each distribution is a numpy array.
                each entry in the numpy array corresponds to the probability of the action at the same index.
        """
        self.states: Mapping = states
        self.actions: Sequence = actions
        self.probs: Dict[int, np.ndarray] = probs

    def __getitem__(self, key: int):
        """Return distribution over actions for given state."""
        assert self.probs[key].size == len(self.actions), "action - distr mismatch."
        return self.probs[key]


class TransitionKernel:
    """Transition kernel for MDP."""

    # TODO: needs work, transiton kernel should yield probs (x, a) -> x'
    def __init__(
        self,
        states: Sequence,
        actions: Sequence,
        probs: Dict[Tuple[int, int], np.ndarray],
    ):
        """Initialize transition kernel."""
        for state in states:
            for action in actions:
                assert probs[(state, action)].size == len(
                    states
                ), "state - distr mismatch."

        self.states: Sequence[int] = states
        self.actions: Sequence[int] = actions
        self.state_action_probs: Dict[Tuple[int, int], np.ndarray] = (
            probs  # indexing with state
        )

    def __getitem__(self, key: Tuple[int, int]) -> np.ndarray:
        """Return distribution over actions for given state."""
        return self.state_action_probs[key]


class RV_Discrete:
    """Discrete atomic random variable."""

    def __init__(self, xk, pk) -> None:
        """Initialize discrete random variable."""
        self.xk: np.ndarray = xk
        self.pk: np.ndarray = pk

    def distr(self):
        """Return distribution as Tuple of numpy arrays."""
        return self.xk, self.pk


class CategoricalDistrCollection:
    """Collection of categorical distributions."""

    # TODO instead of using tuple directly, use RV_Discrete

    def __init__(self, states: Sequence[int], distributions: List[RV_Discrete]) -> None:
        """Initialize collection of categorical distributions."""
        self.states: Sequence = states
        self.distr: Dict = {s: distributions[i] for i, s in enumerate(states)}

    def __getitem__(self, key: int) -> RV_Discrete:
        """Return distribution for state."""
        return self.distr[key]


class CategoricalRewardDistr:
    """Return Distribution with finite support."""

    def __init__(
        self,
        state_action_pairs: List[Tuple[int, int, int]],
        distributions: List[RV_Discrete],
    ) -> None:
        """Initialize return distribution."""
        self.returns: Dict[Tuple[int, int, int], RV_Discrete] = {
            s: d for s, d in zip(state_action_pairs, distributions)
        }

    def __getitem__(self, key: Tuple[int, int, int]) -> RV_Discrete:
        """Return RV_Discrete for (state, action, next_state) triple."""
        return self.returns[key]


class MDP:
    """Markov decision process."""

    # def __init__(self, states: Sequence, actions: Mapping, rewards: CategoricalRewardDistr,
    #              transition_probs: TransitionKernel,terminal_states: Optional[Sequence[int]]=None):
    def __init__(
        self,
        states: Sequence,
        actions: Sequence,
        rewards: CategoricalRewardDistr,
        transition_probs: TransitionKernel,
        terminal_states: Optional[Sequence[int]] = [],
        gamma: np.float64 = 0.5,
    ):
        """Initialize MDP."""
        self.states: Dict = {i: s for i, s in enumerate(states)}
        self.actions: Sequence = actions
        self.rewards: CategoricalRewardDistr = rewards
        # self.rewards: RV_Discrete = rewards
        self.trasition_probs: TransitionKernel = transition_probs
        self.current_policy: Optional[Policy] = None
        self.terminal_states: Optional[Sequence[int]] = terminal_states
        self.gamma: np.float64 = gamma

    def set_policy(self, policy: Policy) -> None:
        """Set policy."""
        self.current_policy = policy


######################
# Algorithm 5.1      #
######################
def categorical_dbo(
    mdp: MDP, pi: Policy, cat_distr_col: CategoricalDistrCollection
) -> CategoricalDistrCollection:
    """Run Algorithm 5.1 categorical distributional bellman operator from book.

    Simple implementation of Algorithm 5.1 from the book without
    performance optimizations.

    Prerequisites:
        finite state space
        finite action space
        rewards with finite support

        ensure that all states in collection are states in the mdp
    """
    ret_distr: List[Tuple[np.ndarray, np.ndarray]] = []

    for state in cat_distr_col.states:
        new_vals: List = []
        new_probs: List = []

        for action in mdp.actions:
            # TODO possibly outsource this as function
            for next_state in mdp.states.values():
                reward_distr = mdp.rewards[(state, action, next_state)]
                prob = (
                    pi[state][action] * mdp.trasition_probs[(state, action)][next_state]
                )

                # if terminal state, no future rewards, only immediate
                if next_state in mdp.terminal_states:
                    new_vals.append(reward_distr.xk)
                    new_probs.append(reward_distr.pk * prob)

                else:
                    distr_update: Tuple[np.ndarray, np.ndarray] = conv_jit(
                        scale(cat_distr_col[next_state], mdp.gamma).distr(),
                        reward_distr.distr(),
                    )
                    new_vals.append(distr_update[0])
                    new_probs.append(distr_update[1] * prob)

        # ready to update \theta(x), store in list
        ret_distr.append(
            RV_Discrete(np.concatenate(new_vals), np.concatenate(new_probs))
        )

    # final collection of distributions along all states
    ret_cat_distr_coll: CategoricalDistrCollection = CategoricalDistrCollection(
        states=cat_distr_col.states, distributions=ret_distr
    )
    return ret_cat_distr_coll


def scale(distr: RV_Discrete, gamma: np.float64) -> RV_Discrete:
    """Scale distribution by factor."""
    distr.xk *= gamma
    return distr


def conv(
    a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    0th entry values, 1st entry probabilities.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


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
def conv_jit(
    a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolution of two distributions.

    Numba JIT compiled version.
    """
    new_val: np.ndarray = np.add(a[0], b[0][:, None]).flatten()
    probs: np.ndarray = np.multiply(a[1], b[1][:, None]).flatten()
    return new_val, probs


@njit
def aggregate_conv_results(
    distr: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
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
        if np.abs(val_sorted[i] - val_sorted[i - 1]) < 1e-10:
            probs_sorted[current] += probs_sorted[i]
        else:
            ret_dist_v.append(val_sorted[i])
            ret_dist_p.append(probs_sorted[i])
            current = i

    # values: np.ndarray = np.unique(distr[0])
    # probs: np.ndarray = np.zeros(values.size)

    # for i, val in enumerate(values):
    # probs[i] = np.sum(distr[1][distr[0] == val])
    # return values, probs

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
            # TODO: ensure that probs sum to 1, other behviour

        rem: np.float64 = 1 - np.sum(new_probs)
        new_probs[random.randint(0, no_of_bins - 1)] += rem

        # aufsummierren, durh  Summe  teilen

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

    g_t: Tuple[np.ndarray, np.ndarray] = aggregate_conv_results(conv_jit(dist1, dist2))
    g_t = proj_func(
        values=g_t[0], probs=g_t[1], no_of_bins=(time_step + 1) * 10, state=1
    )
    return g_t


def plot_atomic_distr(distr: Tuple[np.ndarray, np.ndarray]) -> None:
    """Plot atomic distribution."""
    num_atom: int = distr[0].size
    x_min: np.float64 = np.min(distr[0])
    x_max: np.float64 = np.max(distr[0])

    # bins: np.ndarray = np.digitize(distr[0], np.linspace(x_min, x_max, num_atom // 5))
    new_vals: np.ndarray
    new_probs: np.ndarray
    new_vals, new_probs = project_eqi(
        values=distr[0], probs=distr[1], no_of_bins=num_atom // 40, state=1
    )

    # print(np.sum(new_probs))
    plt.bar(new_vals, new_probs)
    # plt.show()
    return None


def main():
    """Call main function."""
    # trivial MDP bernoulli rewards
    states: List[int] = [0]
    actions: List[int] = [0]
    _rewards: RV_Discrete = RV_Discrete(
        xk=np.array([0.0, 1.0]), pk=np.array([0.5, 0.5])
    )
    rewards: CategoricalRewardDistr = CategoricalRewardDistr([(0, 0, 0)], [_rewards])
    discount_factor: np.float64 = np.array([0.5])
    probs = {(0, 0): np.array([1.0])}
    transition_kernel: TransitionKernel = TransitionKernel(states, actions, probs)
    pi: Policy = Policy(states=states, actions=actions, probs={0: np.array([1.0])})

    distribution_1: RV_Discrete = RV_Discrete(
        xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
    )
    distribution_2: RV_Discrete = RV_Discrete(
        xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
    )
    distribution_3: RV_Discrete = RV_Discrete(
        xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
    )
    distributions: List[RV_Discrete] = [distribution_1, distribution_2, distribution_3]
    total_reward_distr_estimate: CategoricalDistrCollection = (
        CategoricalDistrCollection(states, distributions)
    )

    mdp: MDP = MDP(states, actions, rewards, transition_kernel)
    mdp.set_policy(pi)

    res: CategoricalDistrCollection = total_reward_distr_estimate
    for i in range(10):
        print(f"Iteration {i} started.")
        res: CategoricalDistrCollection = categorical_dbo(mdp, pi, res)
        print(f"Iteration {i} stopped.")

    print(res[0].distr())
    return res


if __name__ == "__main__":

    # a_val = np.array([1, 2, 3])
    # a_probs = np.array([0.3, 0.4, 0.3])
    # b_val = np.array([-5, 5])
    # b_probs = np.array([0.5, 0.5])
    #
    # a = (a_val, a_probs)
    # b = (b_val, b_probs)
    #
    # c = conv(a, b)
    # vals = np.linspace(0, 100, 1000)
    # ps = np.ones(1000) * 1 / 1000
    # # apply_projection(project_eqi)(values=vals, probs=ps, no_of_bins=10, state=1)
    # new_vals, new_probs = project_eqi(values=vals, probs=ps, no_of_bins=10, state=1)
    #
    # print(new_vals, new_probs)
    # print(np.sum(new_probs))
    # print("completed binning.")
    main()
