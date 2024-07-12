import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    loss_history: List[float] = field(default_factory=list)
    episodes_steps: List[int] = field(default_factory=list)
    cumulative_rewards: List[float] = field(default_factory=list)


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
        if run_name.endswith(".npy"):
            # Load Q-table for traditional methods
            self.Q = np.load(os.path.join(self.params.savemodel_folder, run_name))
            self.is_deep_q = False
        elif run_name.endswith(".pth"):
            # Load Deep Q-Network
            self.dqn = DQN(self.env.observation_space.n, self.env.action_space.n)
            self.dqn.load_state_dict(
                torch.load(os.path.join(self.params.savemodel_folder, run_name))
            )
            self.dqn.eval()  # Set the network to evaluation mode
            self.is_deep_q = True
        else:
            raise ValueError(f"Unsupported model format: {run_name}")

    def get_action(self, state):
        if self.is_deep_q:
            state_tensor = torch.FloatTensor(self.one_hot_encode(state)).unsqueeze(0)
            with torch.no_grad():
                q_values = self.dqn(state_tensor)
            return q_values.argmax().item()
        else:
            return np.argmax(self.Q[state])

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

    def one_hot_encode(self, state):
        return np.eye(self.env.observation_space.n)[state]

    def evaluate_once(self, rewards: Rewards) -> Tuple[int, bool]:
        state, _ = self.env.reset()
        self.env.render()

        steps = 0
        lost = False
        while True:
            action = self.get_action(state)
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

    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
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
        self.params.run_name, self.params.run_description = self.init_run_infos(
            self.env, self.params, self.policy
        )
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

    def init_run_infos(
        self, env: Env, params: Params, policy: Policy
    ) -> Tuple[str, str]:
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
            f"{policy.description}",
        )

    def run(self) -> None:
        """
        Run the algorithm.
        :return: None
        """
        for episode in range(self.params.n_episodes):
            state, _ = (
                self.env.reset()
                if self.params.random_seed
                else self.env.reset(seed=self.params.seed)
            )
            episode_history = EpisodeHistory(states=[state], actions=[], rewards=[])

            while True:
                next_state, done = self.play(episode, state, episode_history)
                if not done:
                    state = next_state
                else:
                    break

            self.complete_episode(episode, episode_history)

        self.env.close()

        self.save()

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        """
        Method to complete the episode. Must be overridden if updating the policy has to be made at the end of the episode.
        :param episode: episode number
        :param episode_history: history of the episode
        :return: None
        """
        # Update the policy at the end of the episode
        self.policy.on_episode_end(current_episode=episode)

    def step(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> Tuple[int, bool]:
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
            reward = self.reward_function.compute(
                self.env, terminated, truncated, reward, state, next_state
            )

        # Save the reward
        episode_history.rewards.append(reward)

        # Save the next state in case the episode is not done
        episode_history.states.append(next_state)

        return next_state, terminated or truncated

    def play(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> Tuple[int, bool]:
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

    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
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
        for state, action, reward in zip(
            episode_history.states, episode_history.actions, episode_history.rewards
        ):
            self.Q[state, action] += self.params.learning_rate * (
                reward - self.Q[state, action]
            )

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
    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
        super().__init__(env, params, policy, reward_function)
        self.episode_steps = 0
        self.episode_reward = 0
        self.wins = 0
        self.losses = 0

    def update_action_value_function(
        self, state: int, action: int, reward: float, next_state: int
    ) -> None:
        self.Q[state, action] = self.Q[state, action] + self.params.learning_rate * (
            reward
            + self.params.gamma * np.max(self.Q[next_state, :])
            - self.Q[state, action]
        )

    def play(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> Tuple[int, bool]:
        next_state, done = self.step(episode, state, episode_history)
        self.update_action_value_function(
            state, episode_history.actions[-1], episode_history.rewards[-1], next_state
        )
        self.episode_steps += 1
        self.episode_reward += episode_history.rewards[-1]
        return next_state, done

    def complete_episode(self, episode: int, episode_history: EpisodeHistory) -> None:
        self.historic.average_rewards.append(np.mean(episode_history.rewards))
        self.historic.episodes_histories.append(episode_history)

        # Record steps per episode
        self.historic.episodes_steps.append(self.episode_steps)

        # Record cumulative reward
        if not self.historic.cumulative_rewards:
            self.historic.cumulative_rewards.append(self.episode_reward)
        else:
            self.historic.cumulative_rewards.append(
                self.historic.cumulative_rewards[-1] + self.episode_reward
            )

        # Determine win/loss (this may need to be adjusted based on your specific environment)
        if self.episode_reward > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Reset episode-specific variables
        self.episode_steps = 0
        self.episode_reward = 0

        super().complete_episode(episode, episode_history)

    def run(self) -> None:
        for episode in range(self.params.n_episodes):
            state, _ = (
                self.env.reset()
                if self.params.random_seed
                else self.env.reset(seed=self.params.seed)
            )
            episode_history = EpisodeHistory(states=[state], actions=[], rewards=[])

            while True:
                next_state, done = self.play(episode, state, episode_history)
                if not done:
                    state = next_state
                else:
                    break

            self.complete_episode(episode, episode_history)

        self.env.close()
        self.save()

    def evaluate(self, rewards: Rewards, n_runs: int = None) -> Tuple[list, list]:
        steps = []
        losses = []

        for _ in range(n_runs or self.params.n_runs):
            state, _ = self.env.reset()
            episode_steps = 0
            episode_reward = 0
            done = False

            while not done:
                action = np.argmax(self.Q[state, :])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                episode_steps += 1
                episode_reward += reward

            steps.append(episode_steps)
            losses.append(
                episode_reward <= 0
            )  # Consider it a loss if reward is not positive

        self.env.close()

        return steps, losses


class SARSA(Algorithm):
    """
    SARSA algorithm.
    """

    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def play(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> int or None:
        """
        Method to play the game.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state or None if the episode is done
        """

        next_state, done = self.step(episode, state, episode_history)
        next_action, _ = self.policy.choose_action(self.env, next_state, self.Q)
        self.update_action_value_function(
            state,
            episode_history.actions[-1],
            episode_history.rewards[-1],
            next_state,
            next_action,
        )
        return next_state, done

    def update_action_value_function(
        self, state: int, action: int, reward: float, next_state: int, next_action: int
    ) -> None:
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
            reward
            + self.params.gamma * self.Q[next_state, next_action]
            - self.Q[state, action]
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
    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)

    def play(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> int or None:
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
        print(
            f"Episode {episode} - Average reward: {np.mean(episode_history.rewards)} - steps: {len(episode_history.rewards)}"
        )
        super().complete_episode(episode, episode_history)


class ValueIteration(Algorithm):
    """
    Value Iteration algorithm.
    """

    V: np.ndarray

    def __init__(
        self,
        env: Env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ) -> None:
        """
        Initialize the algorithm.
        :param env: environment
        :param params: parameters to run the algorithm with
        :param policy: policy to use
        :param reward_function: reward function to use, can be None
        :return: None
        """
        super().__init__(env, params, policy, reward_function)
        self.V = np.zeros(self.env.observation_space.n)

    def run(self) -> None:
        """
        Perform value iteration to find the optimal policy.
        :return: None
        """
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.V[s]
                max_value = max(
                    [
                        sum(
                            [
                                p * (r + self.params.gamma * self.V[s_])
                                for p, s_, r, _ in self.env.P[s][a]
                            ]
                        )
                        for a in range(self.env.action_space.n)
                    ]
                )

                self.V[s] = max_value
                delta = max(delta, abs(v - max_value))

            # Convergence threshold
            if delta < self.params.theta:
                break

        # Once the value function has converged, extract the policy
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.env.observation_space.n):
            self.Q[s] = [
                sum(
                    [
                        p * (r + self.params.gamma * self.V[s_])
                        for p, s_, r, _ in self.env.P[s][a]
                    ]
                )
                for a in range(self.env.action_space.n)
            ]

        self.env.close()

        self.save()

    def play(
        self, episode: int, state: int, episode_history: EpisodeHistory
    ) -> Tuple[int, bool]:
        """
        Use the computed policy to play one step in the environment.
        :param episode: episode number
        :param state: current state
        :param episode_history: history of the episode
        :return: next state and if the episode is done
        """
        action = self.policy.choose_action(self.env, state, self.Q)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        episode_history.states.append(next_state)
        episode_history.actions.append(action)
        episode_history.rewards.append(reward)
        return next_state, terminated or truncated


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DeepQLearning(Algorithm):
    def __init__(
        self,
        env,
        params: Params,
        policy: Policy,
        reward_function: RewardFunction = None,
    ):
        super().__init__(env, params, policy, reward_function)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(env.observation_space.n, env.action_space.n).to(self.device)
        self.target_dqn = DQN(env.observation_space.n, env.action_space.n).to(
            self.device
        )
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=params.learning_rate)
        self.memory = deque(maxlen=params.memory_size)
        self.batch_size = params.batch_size
        self.update_target_every = params.update_target_every
        self.steps = 0
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() < self.policy.epsilon:
            return self.env.action_space.sample()
        state_tensor = (
            torch.FloatTensor(self.one_hot_encode(state)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.dqn(state_tensor)
        return q_values.argmax().item()

    def one_hot_encode(self, state):
        return np.eye(self.env.observation_space.n)[state]

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(self.one_hot_encode(states)).to(self.device)
        next_states = torch.FloatTensor(self.one_hot_encode(next_states)).to(
            self.device
        )
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.params.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.steps += 1
        return loss.item()

    def run(self):
        for episode in range(self.params.n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_losses = []
            episode_steps = 0

            while True:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if self.reward_function:
                    reward = self.reward_function.compute(
                        self.env, terminated, truncated, reward, state, next_state
                    )

                self.remember(state, action, reward, next_state, done)
                loss = self.train()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward
                episode_steps += 1

                if done:
                    break

            self.historic.average_rewards.append(total_reward)
            self.historic.epsilons.append(self.policy.epsilon)
            self.historic.episodes_steps.append(episode_steps)
            self.historic.cumulative_rewards.append(
                total_reward
                if episode == 0
                else self.historic.cumulative_rewards[-1] + total_reward
            )
            if episode_losses:
                self.historic.loss_history.append(np.mean(episode_losses))
            self.policy.on_episode_end(current_episode=episode)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Steps: {episode_steps}, Total Reward: {total_reward}, Epsilon: {self.policy.epsilon}, Avg Loss: {np.mean(episode_losses) if episode_losses else 'N/A'}"
                )

        self.env.close()
        self.save()

    def save(self):
        torch.save(
            self.dqn.state_dict(),
            self.params.savemodel_folder / f"{self.params.run_name}_dqn.pth",
        )

    def load(self, run_name: str = ""):
        self.dqn.load_state_dict(
            torch.load(
                self.params.savemodel_folder
                / f"{run_name or self.params.run_name}_dqn.pth"
            )
        )
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def evaluate(self, rewards: Rewards, n_runs: int = None) -> Tuple[list, list]:
        """
        Evaluate by playing a number of games without learning.
        :param rewards: rewards to use
        :param n_runs: number of games to play. If None, it will play the number of games defined in the parameters.
        :return: list of steps, list of results (True for win, False for loss)
        """
        steps = []
        results = []

        for _ in range(n_runs or self.params.n_runs):
            state, _ = self.env.reset()
            total_steps = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_steps += 1

            steps.append(total_steps)
            # Consider it a win if the final reward is the win_reward
            results.append(reward == rewards.win_reward)

        self.env.close()

        return steps, results
