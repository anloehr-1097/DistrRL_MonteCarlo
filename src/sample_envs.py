"""Definition of some sample envs (mdp + return distr funcs) for testing."""

# from sys import thread_info
from typing import Iterator, List, Dict, Sequence
# from numba import itertools
import itertools
import numpy as np
import scipy.stats as sp
from .drl_primitives import (
    State,
    Action,
    RewardDistributionCollection,
    ReturnDistributionFunction,
    TransitionKernel,
    Policy,
    MDP
)
from .random_variables import (
    ContinuousRV,
    RV,
    DiscreteRV,
)
from .distributions import emp_normal

EMP_APPROX_SAMPLES = 100

class SimulationEnv:
    def __init__(self, mdp: MDP,
                 return_distr_fun_est: ReturnDistributionFunction):

        self.mdp = mdp
        self.return_distr_fun_est = return_distr_fun_est
        return None


### Define Bernoulli test case ###
bernoulli_states = [State(0, "0", 0)]
bernoulli_actions = [Action(0, "0", 0)]
bernoulli_rewards = RewardDistributionCollection(
    [(bernoulli_states[0], bernoulli_actions[0], bernoulli_states[0])],
    [DiscreteRV(np.array([0.0, 1.0]), np.array([0.5, 0.5]))])
bernoulli_transitions = TransitionKernel(
    bernoulli_states,
    bernoulli_actions,
    {(bernoulli_states[0], bernoulli_actions[0]): np.array([1.0])})

bernoulli_mdp: MDP = MDP(
    states=bernoulli_states,
    actions=bernoulli_actions,
    rewards=bernoulli_rewards,
    gamma=np.float64(0.5),
    transition_probs=bernoulli_transitions
)

bernoulli_pi: Policy = Policy(
    states=bernoulli_mdp.states,
    actions=bernoulli_mdp.actions,
    probs={bernoulli_states[0]: np.array([1.0])},
)

bernoulli_distribution_1: DiscreteRV = DiscreteRV(
    xk=np.array([0, 1.0]), pk=np.array([0.5, 0.5])
)
bernoulli_distributions: List[DiscreteRV] = [bernoulli_distribution_1]

bernoulli_total_reward_distr_estimate: ReturnDistributionFunction = (
    ReturnDistributionFunction(bernoulli_mdp.states, bernoulli_distributions)
)
bernoulli_mdp.set_policy(bernoulli_pi)


bernoulli_env: SimulationEnv = SimulationEnv(
    bernoulli_mdp, bernoulli_total_reward_distr_estimate
)


### Define cyclical test case from paper ###
cyclical_states: Sequence[State] = [State(i, f'{i}', i) for i in range(3)]
# cyclical_states: Mapping[str, int] = {str(i): i for i in [0, 1, 2]}
cyclical_actions: Sequence[Action] = [Action(0, "0", 0)]
cyclical_state_action_probs: Dict[State, np.ndarray] = {
    cyclical_states[i]: np.array([1.0]) for i, _ in enumerate(cyclical_states)
}
cyclical_pi = Policy(states=cyclical_states, actions=cyclical_actions,
                     probs=cyclical_state_action_probs)


bernoulli_transitions = TransitionKernel(
    bernoulli_states,
    bernoulli_actions,
    {(bernoulli_states[0], bernoulli_actions[0]): np.array([1.0])})

cyclical_transition_probs: TransitionKernel = TransitionKernel(
    states=cyclical_states,
    actions=cyclical_actions,
    probs={
        (cyclical_states[0], cyclical_actions[0]): np.array([0.0, 1.0, 0.0]),
        (cyclical_states[1], cyclical_actions[0]): np.array([0.0, 0.0, 1.0]),
        (cyclical_states[2], cyclical_actions[0]): np.array([1.0, 0.0, 0.0])}
)

# cyclical_transition_kernel = TransitionKernel(
#     cyclical_states, cyclical_actions,
#     cyclical_transition_probs)

# reward distributions for (state, action, next_state) triples
# set samples for empirical approximation
# cyclical_r_001 = emp_normal(-3, np.sqrt(1), EMP_APPROX_SAMPLES)
# cyclical_r_102 = emp_normal(5, np.sqrt(2), EMP_APPROX_SAMPLES)
# cyclical_r_200 = emp_normal(0, np.sqrt(0.5), EMP_APPROX_SAMPLES)

cyclical_r_001 = ContinuousRV(scipy_rv_cont=sp.norm(loc=-3, scale=np.sqrt(1)))
cyclical_r_102 = ContinuousRV(scipy_rv_cont=sp.norm(loc=5, scale=np.sqrt(2)))
cyclical_r_200 = ContinuousRV(scipy_rv_cont=sp.norm(loc=0, scale=np.sqrt(0.5)))

cyclical_state_action_state_triples: Iterator = \
    itertools.product(cyclical_states, cyclical_actions, cyclical_states)

cyclical_state_action_state_triples: Iterator = \
    itertools.filterfalse(
        lambda x: cyclical_transition_probs[(x[0], x[1])][x[2].index] == 0,
        cyclical_state_action_state_triples
    )

# cyclical_state_action_state_triples = [(0, 0, 1), (1, 0, 2), (2, 0, 0)]

cyclical_rewards = RewardDistributionCollection(
    state_action_state_triples=list(cyclical_state_action_state_triples),
    distributions=[cyclical_r_001, cyclical_r_102, cyclical_r_200])

# initial total reward distributions for each state
cyclical_distribution_1: DiscreteRV = DiscreteRV(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distribution_2: DiscreteRV = DiscreteRV(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distribution_3: DiscreteRV = DiscreteRV(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distributions: List[DiscreteRV] = [cyclical_distribution_1,
                                            cyclical_distribution_2,
                                            cyclical_distribution_3]

cyclical_return_distr_estimate: ReturnDistributionFunction = (
    ReturnDistributionFunction(cyclical_states,
                               cyclical_distributions)
)

cyclical_mdp: MDP = MDP(
    states=cyclical_states,
    actions=cyclical_actions,
    rewards=cyclical_rewards,
    transition_probs=cyclical_transition_probs,
    gamma=np.float64(0.7)
    )
cyclical_mdp.set_policy(cyclical_pi)


cyclycal_real_return_distr_fun: ReturnDistributionFunction = ReturnDistributionFunction(
    states=cyclical_states,
    distributions=[
        ContinuousRV(sp.norm(loc=0.761, scale=np.sqrt(2.380))),
        ContinuousRV(sp.norm(loc=5.373, scale=np.sqrt(2.816))),
        ContinuousRV(sp.norm(loc=0.533, scale=np.sqrt(1.666)))
    ]
)
cyclical_env: SimulationEnv = SimulationEnv(
    cyclical_mdp,
    cyclical_return_distr_estimate
)


# CAUCHY ENVIRONMENT - shares a lot with cyclical env
cauchy_r_001 = ContinuousRV(scipy_rv_cont=sp.cauchy(loc=-3, scale=0.5))
cauchy_r_102 = ContinuousRV(scipy_rv_cont=sp.cauchy(loc=5, scale=0.1))
cauchy_r_200 = ContinuousRV(scipy_rv_cont=sp.cauchy(loc=0, scale=5))

# cauchy_r_001 = ContinuousRV(scipy_rv_cont=sp.norm(loc=-3, scale=np.sqrt(0.5)))
# cauchy_r_102 = ContinuousRV(scipy_rv_cont=sp.norm(loc=5, scale=np.sqrt(0.1)))
# cauchy_r_200 = ContinuousRV(scipy_rv_cont=sp.norm(loc=0, scale=np.sqrt(5)))
#
cauchy_state_action_state_triples: Iterator = \
    itertools.product(cyclical_states, cyclical_actions, cyclical_states)

cauchy_state_action_state_triples: Iterator = \
    itertools.filterfalse(
        lambda x: cyclical_transition_probs[(x[0], x[1])][x[2].index] == 0,
        cauchy_state_action_state_triples
    )

cauchy_rewards: RewardDistributionCollection = RewardDistributionCollection(
    state_action_state_triples=list(cauchy_state_action_state_triples),
    distributions=[cauchy_r_001, cauchy_r_102, cauchy_r_200])


cauchy_mdp: MDP = MDP(
    states=cyclical_states,
    actions=cyclical_actions,
    rewards=cauchy_rewards,
    transition_probs=cyclical_transition_probs,
    gamma=np.float64(0.7)
    )
cauchy_mdp.set_policy(cyclical_pi)

cauchy_env: SimulationEnv = SimulationEnv(
    cauchy_mdp,
    cyclical_return_distr_estimate
)
