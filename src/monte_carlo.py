from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .random_variables import DiscreteRV
from .drl_primitives import State, Action, ReturnDistributionFunction, MDP
from .config import DEBUG

MAX_TRAJ_LEN: int = 1000


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
        est_return_distr_fun.distr[state] = DiscreteRV(
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
