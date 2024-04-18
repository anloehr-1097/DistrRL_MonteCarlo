"""Definition of some sample envs (mdp + return distr funcs) for testing."""

from typing import List, Dict
import numpy as np
from preliminary_tests import (
    RV_Discrete,
    CategoricalRewardDistr,
    CategoricalDistrCollection,
    TransitionKernel,
    Policy,
    MDP,
)
from distributions import emp_normal

EMP_APPROX_SAMPLES = 500

class SimulationEnv:
    def __init__(
        self, mdp: MDP, total_reward_distr_estimate: CategoricalDistrCollection
    ):
        self.mdp = mdp
        self.total_reward_distr_estimate = total_reward_distr_estimate
        return None


### Define Bernoulli test case ###
bernoulli_mdp: MDP = MDP(
    states=[0],
    actions=[0],
    rewards=CategoricalRewardDistr(
        [(0, 0, 0)], [RV_Discrete(np.array([0.0, 1.0]), np.array([0.5, 0.5]))]
    ),
    gamma=np.array([0.5]),
    transition_probs=TransitionKernel([0], [0], {(0, 0): np.array([1.0])}),
)

bernoulli_pi: Policy = Policy(
    states=bernoulli_mdp.states,
    actions=bernoulli_mdp.actions,
    probs={0: np.array([1.0])},
)

bernoulli_distribution_1: RV_Discrete = RV_Discrete(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
bernoulli_distributions: List[RV_Discrete] = [bernoulli_distribution_1]

bernoulli_total_reward_distr_estimate: CategoricalDistrCollection = (
    CategoricalDistrCollection(bernoulli_mdp.states, bernoulli_distributions)
)
bernoulli_mdp.set_policy(bernoulli_pi)


bernoulli_env: SimulationEnv = SimulationEnv(
    bernoulli_mdp, bernoulli_total_reward_distr_estimate
)


### Define cyclical test case from paper ###

cyclical_states = [0, 1, 2]
cyclical_actions = [0]

cyclical_state_action_probs = {0: np.array([1.0]),
                               1: np.array([1.0]),
                               2: np.array([1.0])}

cyclical_pi = Policy(states=cyclical_states, actions=cyclical_actions,
                     probs=cyclical_state_action_probs)

cyclical_transition_probs = {(0, 0): np.array([0.0, 1.0, 0.0]),
                             (1, 0): np.array([0.0, 0.0, 1.0]),
                             (2, 0): np.array([1.0, 0.0, 0.0])}

cyclical_transition_kernel = TransitionKernel(cyclical_states, cyclical_actions,
                                              cyclical_transition_probs)

# reward distributions for (state, action, next_state) triples
# set samples for empirical approximation
cyclical_r_001 = emp_normal(-3, 1, EMP_APPROX_SAMPLES)
cyclical_r_102 = emp_normal(5, 2, EMP_APPROX_SAMPLES)
cyclical_r_200 = emp_normal(0, 0.5, EMP_APPROX_SAMPLES)
cyclical_state_action_state = [(0, 0, 1), (1, 0, 2), (2, 0, 0)]

cyclical_rewards = CategoricalRewardDistr(state_action_pairs=\
                                          cyclical_state_action_state,
                                          distributions=\
                                          [cyclical_r_001, cyclical_r_102,
                                           cyclical_r_200])


# initial total reward distributions for each state 
cyclical_distribution_1: RV_Discrete = RV_Discrete(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distribution_2: RV_Discrete = RV_Discrete(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distribution_3: RV_Discrete = RV_Discrete(
    xk=np.array([-3.0, 1.0]), pk=np.array([0.5, 0.5])
)
cyclical_distributions: List[RV_Discrete] = [cyclical_distribution_1,
                                             cyclical_distribution_2,
                                             cyclical_distribution_3]

cyclical_total_reward_distr_estimate: CategoricalDistrCollection = (
    CategoricalDistrCollection(cyclical_states,
                               cyclical_distributions)
)

cyclical_mdp: MDP = MDP(
    states=cyclical_states,
    actions=cyclical_actions,
    rewards=cyclical_rewards,
    transition_probs=cyclical_transition_kernel
    )
cyclical_mdp.set_policy(cyclical_pi)

cyclical_env: SimulationEnv = SimulationEnv(
    cyclical_mdp,
    cyclical_total_reward_distr_estimate
)
