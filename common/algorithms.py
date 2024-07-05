import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from gymnasium import Env

from common.params import Params
from common.policies import Policy
from common.rewards import RewardFunction, Rewards


@dataclass
class EpisodeHistory:
    """
    Dataclass to store the history of an episode.
    """
    states: list
    actions: list
    rewards: list


@dataclass
class AlgorithmHistory:
    """
    Dataclass to store the history of an algorithm training.
    """
    episodes_histories: List[EpisodeHistory] = field(default_factory=list)
    average_rewards: List[float] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)


class Evaluation:
    """
    Class that implements the evaluation part for Algorithms.
    """

    env: Env
    params: Params

    Q: np.ndarray

    def __init__(self, env: Env, params: Params, model_name: str) -> None:
        """
        Initialize the evaluation.
        :param env: environment
        :param params: parameters to run the algorithm with
        :return: None
        """
        self.env = env
        self.params = params
        self.load(model_name)

    def load(self, run_name: str = "") -> None:
        """
        Load the algorithm model.
        :param run_name: name of the run. If empty, it will load the model with the current run name.
        :return: None
        """
        self.Q = np.load(os.path.join(self.params.savemodel_folder, run_name or self.params.run_name + ".npy"))

    @staticmethod
    def get_policy_from_q(Q: np.ndarray) -> np.ndarray:
        """
        Get the optimal policy from a given Q-table.
        :param Q: Q-table
        :return: policy
        """
        return np.argmax(Q, axis=1)

    @property
    def computed_policy(self) -> np.ndarray:
        """
        Current policy computed from the Q-table.
        :return: policy
        """
        return self.get_policy_from_q(self.Q)

    def evaluate(self, rewards: Rewards, n_runs: int = None) -> Tuple[list, list]:
        """
        Evaluate by playing a number of games without learning.
        :param rewards: rewards to use
        :param n_runs: number of games to play. If None, it will play the number of games defined in the parameters.
        :return: list of steps, number of losses
        """

        steps = []
        losses = []

        for _ in range(n_runs or self.params.n_runs):
            n_steps, lost = self.evaluate_once(rewards)
            steps.append(n_steps)
            losses.append(lost)

        self.env.close()

        return steps, losses

    def evaluate_once(self, rewards: Rewards) -> Tuple[int, bool]:
        """
        Evaluate by playing a game without learning.
        :param rewards: rewards to use
        :return: number of steps, if the game was lost
        """

        state, _ = self.env.reset() if self.params.random_seed else self.env.reset(seed=self.params.seed)
        self.env.render()

        steps = 0
        lost = False
        while True:
            action = self.computed_policy[state]
            state, reward, terminated, truncated, _ = self.env.step(action)
            steps += 1

            if terminated or truncated:
                if reward in [rewards.lose_reward, rewards.playing_reward]:
                    lost = True
                break

            self.env.render()

        return steps, lost


class Algorithm(Evaluation):
    """
    Class for reinforcement learning algorithms. Should be inherited by specific algorithms.
    """

    env: Env
    params: Params
    policy: Policy
    reward_function: RewardFunction

    Q: np.ndarray

    historic: AlgorithmHistory

    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        self.env = env
        self.params = params
        self.policy = policy
        self.reward_function = reward_function

        self.Q = self.init_q_table(self.env)
        self.params.run_name, self.params.run_description = self.init_run_infos(self.env, self.params, self.policy)
        self.historic = AlgorithmHistory()

    @staticmethod
    def init_q_table(env: Env) -> np.ndarray:
        """
        Initialize the Q-table. By default, it is a matrix of zeros. Override this method to customize the initialization
        (in case of SARSA, for example).
        :param env: environment
        :return: Q-table
        """
        return np.zeros((env.observation_space.n, env.action_space.n))

    def init_run_infos(self, env: Env, params: Params, policy: Policy) -> Tuple[str, str]:
        """
        Initialize the run name and description.
        :param env: environment
        :param algorithm_name: name of the algorithm
        :param params: parameters
        :param policy: policy
        :return: run name
        """
        return (
            # Name of the run
            f"{env.spec.id}_{self.__class__.__name__}_"
            f"{params.n_episodes}_{params.map_size[0]}x{params.map_size[1]}_"
            f"{policy.__class__.__name__}",
            # Description of the run
            f"{env.spec.id} - {self.__class__.__name__} - {params.n_episodes} episodes - "
            f"{params.map_size[0]}x{params.map_size[1]} - "
            f"{params.learning_rate} α - {params.gamma} γ - "
            f"{policy.description}"
        )

    def run(self) -> None:
        """
        Run the algorithm.
        :return: None
        """
        for episode in range(self.params.n_episodes):
            state, _ = self.env.reset() if self.params.random_seed else self.env.reset(seed=self.params.seed)
            episode_history = EpisodeHistory(states=[state], actions=[], rewards=[])

            while True:
                next_state, done = self.play(episode, state, episode_history)
                if not done:
                    state = next_state
                else:
                    break

            self.complete_episode(episode, episode_history)

        self.env.close()

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode. Must be overridden if updating the policy has to be made at the end of the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        # Update the policy at the end of the episode
        self.policy.on_episode_end(current_episode=episode)
        self.save()

    def step(self, episode: int, state: int, episode_history: EpisodeHistory) -> Tuple[int, bool]:
        """
        Play one episode of the game.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state and if the episode is done
        """

        action, computed_epsilon = self.policy.choose_action(self.env, state, self.Q)

        if len(self.historic.epsilons) < episode:
            self.historic.epsilons.insert(episode, computed_epsilon)

        # Take the action
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        # Save the action
        episode_history.actions.append(action)

        # Customize the obtained reward depending on the reward function
        if self.reward_function is not None:
            reward = self.reward_function.compute(self.env, terminated, truncated, reward, state, next_state)

        # Save the reward
        episode_history.rewards.append(reward)

        # Save the next state in case the episode is not done
        episode_history.states.append(next_state)

        return next_state, terminated or truncated

    def play(self, episode: int, state: int, episode_history: EpisodeHistory) -> Tuple[int, bool]:
        """
        Method to play the game. Must be overridden if updating the action-value function has to be
        made at each step.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state and if the episode is done
        """
        return self.step(episode, state, episode_history)

    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.
        :return: None
        """
        self.Q = self.init_q_table(self.env)
        self.historic = AlgorithmHistory()

    def save(self) -> None:
        """
        Save the algorithm model.
        :return: None
        """

        if not os.path.exists(self.params.savemodel_folder):
            os.makedirs(self.params.savemodel_folder)

        np.save(self.params.savemodel_folder / self.params.run_name, self.Q)


class MonteCarlo(Algorithm):
    """
    Monte Carlo algorithm.
    """

    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def bellman_backup(self, episode_history: EpisodeHistory) -> List[float]:
        """
        Bellman backup to compute the total rewards for the episode per state, so it backpropagates the rewards.
        :param episode_history: history of the episode
        :return: total rewards for the episode per state
        """
        G = 0
        rewards = []

        for reward in episode_history.rewards[::-1]:
            G = self.params.gamma * G + reward
            rewards.insert(0, G)

        return rewards

    def update_action_value_function(self, episode_history: EpisodeHistory) -> None:
        """
        Update the action-value function.
        :param episode_history: history of the episode
        :return: None
        """
        for state, action, reward in zip(episode_history.states, episode_history.actions, episode_history.rewards):
            self.Q[state, action] += self.params.learning_rate * (reward - self.Q[state, action])

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        # Compute the total rewards for the episode per state
        total_rewards = self.bellman_backup(episode_history)
        self.historic.average_rewards.append(np.mean(total_rewards))

        # Update the action-value function depending on the bellman backup
        self.update_action_value_function(episode_history)
        self.historic.episodes_histories.append(episode_history)

        super().complete_episode(episode, episode_history)


class QLearning(Algorithm):
    """
    Q-Learning algorithm.
    """

    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def update_action_value_function(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the action-value function.
        :param state: current state
        :param action: action
        :param reward: reward
        :param next_state: next state
        :return: None
        """
        self.Q[state, action] = self.Q[state, action] + self.params.learning_rate * (
                reward + self.params.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]
        )

    def play(self, episode: int, state: int, episode_history: EpisodeHistory) -> int or None:
        """
        Method to play the game.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state or None if the episode is done
        """

        next_state, done = self.step(episode, state, episode_history)
        self.update_action_value_function(state, episode_history.actions[-1], episode_history.rewards[-1], next_state)
        return next_state, done

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        self.historic.average_rewards.append(np.mean(episode_history.rewards))
        self.historic.episodes_histories.append(episode_history)
        super().complete_episode(episode, episode_history)


class SARSA(Algorithm):
    """
    SARSA algorithm.
    """

    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def play(self, episode: int, state: int, episode_history: EpisodeHistory) -> int or None:
        """
        Method to play the game.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state or None if the episode is done
        """

        next_state, done = self.step(episode, state, episode_history)
        next_action, _ = self.policy.choose_action(self.env, next_state, self.Q)
        self.update_action_value_function(state, episode_history.actions[-1], episode_history.rewards[-1], next_state,
                                          next_action)
        return next_state, done

    def update_action_value_function(self, state: int, action: int, reward: float, next_state: int,
                                     next_action: int) -> None:
        """
        Update the action-value function.
        :param state: current state
        :param action: action
        :param reward: reward
        :param next_state: next state
        :param next_action: next action
        :return: None
        """
        self.Q[state, action] = self.Q[state, action] + self.params.learning_rate * (
                reward + self.params.gamma * self.Q[next_state, next_action] - self.Q[state, action]
        )

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        self.historic.average_rewards.append(np.mean(episode_history.rewards))
        self.historic.episodes_histories.append(episode_history)
        super().complete_episode(episode, episode_history)


class BruteForce(Algorithm):
    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def play(self, episode: int, state: int, episode_history: EpisodeHistory) -> int or None:
        """
        Method to play the game.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state or None if the episode is done
        """

        return self.step(episode, state, episode_history)

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        self.historic.average_rewards.append(np.mean(episode_history.rewards))
        self.historic.episodes_histories.append(episode_history)
        print(f"Episode {episode} - Average reward: {np.mean(episode_history.rewards)} - steps: {len(episode_history.rewards)}")
        super().complete_episode(episode, episode_history)


class ValueIteration(Algorithm):
    """
    Value Iteration algorithm.
    """

    def __init__(self, env: Env, params: Params, policy: Policy, reward_function: RewardFunction = None):
        super().__init__(env, params, policy, reward_function)
        self.V = np.zeros(self.env.observation_space.n)  # Initialize the value table

    def run(self):
        """
        Perform value iteration to find the optimal policy.
        """
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                max_value = max([sum([p * (r + self.params.gamma * self.V[s_])
                                      for p, s_, r, _ in self.env.P[s][a]])
                                 for a in range(self.env.action_space.n)])
                self.V[s] = max_value
                delta = max(delta, abs(v - max_value))
            if delta < self.params.theta:  # convergence threshold
                break

        # Once the value function has converged, extract the policy
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.env.observation_space.n):
            self.Q[s] = [sum([p * (r + self.params.gamma * self.V[s_])
                              for p, s_, r, _ in self.env.P[s][a]])
                         for a in range(self.env.action_space.n)]

    def play(self, episode: int, state: int, episode_history: EpisodeHistory) -> Tuple[int, bool]:
        """
        Use the computed policy to play one step in the environment.
        """
        action = np.argmax(self.Q[state])
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        episode_history.states.append(next_state)
        episode_history.actions.append(action)
        episode_history.rewards.append(reward)
        return next_state, terminated or truncated




