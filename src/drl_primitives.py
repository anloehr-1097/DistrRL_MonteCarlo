from dataclasses import dataclass
import logging
from typing import Dict, Sequence, Tuple, Union, Optional, List, Callable
import itertools
import numpy as np
from .random_variables import DiscreteRV, ContinuousRV, RV
from .config import ATOL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


TERMINAL_STATE: State = State(
    state=-1,
    name="TERMINAL",
    is_terminal=True,
    index=-1
)


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


class ReturnDistributionFunction:
    """Return Distribution Function type.

    This class is used for eta in mathcal{P}^{mathcal{S}}.
    """

    # TODO instead of using tuple directly, use RV_Discrete

    def __init__(self, states: Sequence[State],
                 # distributions: Optional[Sequence[DiscreteRV]]) -> None:
                 distributions: Optional[Sequence[RV]]) -> None:
        """Initialize collection of categorical distributions."""
        self.states: Sequence[State] = states
        self.index_set = states

        # accomodate MC method implementation with the followign conditional
        if distributions:
            self.distr: Dict[State, RV] = {s: distributions[i] for i, s in enumerate(states)}  # type: ignore
        else:
            self.distr: Dict[State, Optional[RV]] = {s: None for s in states}

    def __getitem__(self, state: Union[State, int]) -> Optional[RV]:
        """Return distribution for state."""
        if isinstance(state, int):
            target_state = list(filter(lambda s: s.index == state, self.states))[0]
            return self.distr[target_state]
        else:
            return self.distr[state]

    def __len__(self) -> int:
        return len(self.states)

    def get_size(self) -> Tuple[Union[int, float], ...]:
        """Return size of the return distributions."""

        if self.distr[self.states[0]]:
            return tuple(self.distr[s].size for s in self.states)  # type: ignore
            # Make sure distributions are actually available

        logger.warning("Calling size on empty distributions.")
        return tuple(0 for _ in self.states)

    def get_max_size(self) -> Union[int, float]:
        return np.max(np.asarray(self.get_size()))

    def __setitem__(self, key: State, value: RV) -> None:
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
        self.index_set = state_action_state_triples

    def __getitem__(self, key: Tuple[State, Action, State]) -> RV:
        """Return RV_Discrete for (state, action, next_state) triple."""
        return self.rewards[key]


class MDP:
    """Markov decision process."""

    def __init__(
        self,
        states: List[State],
        actions: List[Action],
        rewards: RewardDistributionCollection,
        transition_probs: TransitionKernel,
        gamma: np.float64 = np.float64(0.5),
    ):
        """Initialize MDP."""
        self.states: List[State] = states
        self.actions: List[Action] = actions
        self.state_action_state_triples: List[Tuple[State, Action, State]] = \
            list(itertools.product(states, actions, states))
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


# Projection parameter component (for state or state-action-state triple)
PPComponent = Union[int, np.float64, np.ndarray]

# Projection parameter key (either state or state-action-state triple)
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
     ReturnDistributionFunction,  # return distr function estimate. Dim:S
     RewardDistributionCollection,  # reward distr coll est. Dim: S x A x S
     MDP,  # mdp
     List[Tuple[State, Action, State]],  # inner index set
     List[State],  # outer index set
     ],
    Tuple[ProjectionParameter, ProjectionParameter]]  # proj params


# The inner projection can use the properties of the actual rewards
# which are available according to assumptions.
InnerParamAlgo = Callable[
    [int,
     RewardDistributionCollection,  # MDP rewards
     RewardDistributionCollection,  # reward distribution estimate
     MDP,  # mdp
     Sequence[Tuple[State, Action, State]]  # inner index set
     ],
    ProjectionParameter]

# The outer projection can not use the ReturnDistributionFunction
OuterParamAlgo = Callable[
    [int,  # iteration num
     ReturnDistributionFunction,  # return distribution function estimate
     MDP,  # MDP
     Sequence[State]  # outer index set
     ],
    ProjectionParameter]

OneComponentParamAlgo = Union[InnerParamAlgo, OuterParamAlgo]


def wasserstein_beta(
    rv1: RV,
    rv2: RV,
    beta: float=1,
        smallest_nonzero: float=ATOL) -> float:
    """Wasserstein beta metric."""

    assert beta >= 1, "beta must be greater than or equal to 1."
    u: np.ndarray = np.linspace(smallest_nonzero, 1 - smallest_nonzero, 100000)
    rv1_qs: np.ndarray = rv1.qf(u)  # type: ignore
    rv2_qs: np.ndarray = rv2.qf(u)  # type: ignore
    diffs: np.ndarray = np.abs(rv1_qs - rv2_qs)**beta
    return (np.sum(diffs) / u.size)**(1 / beta)


def birnb_orl_avg_dist_beta(
    rv1: Union[DiscreteRV, ContinuousRV],
    rv2: Union[DiscreteRV, ContinuousRV],
        beta: float=1) -> float:
    """
    TODO
    Wasserstein beta distance between two distributions.

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

    if beta == np.inf:
        return np.max(np.abs(cdf_rv1 - cdf_rv2))

    diffs: np.ndarray = np.abs(cdf_rv1 - cdf_rv2)**beta
    weights = np.diff(common_support)
    return float(np.sum(weights * diffs)**(1 / beta))  # type: ignore


def extended_metric(metric: Callable[[RV, RV, float], float],
                    rv1s: Dict[PPKey, RV],
                    rv2s: Dict[PPKey, RV],
                    beta: float=1) -> float:
    """Extended metric for Wasserstein beta distance."""

    assert rv1s.keys() == rv2s.keys(), "Keys of distributions do not match."
    metric_evals: np.ndarray = np.asarray([metric(rv1s[key], rv2s[key], beta) for key in rv1s.keys()])  # type of elements: float
    return np.max(metric_evals)
