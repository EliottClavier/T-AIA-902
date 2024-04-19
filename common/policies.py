import random
from typing import Tuple

import numpy as np
from abc import abstractmethod

from gymnasium import Env
from gymnasium.core import ObsType


class Policy:

    @abstractmethod
    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        """
        Choose an action based on the policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        pass

    def on_episode_end(self, **kwargs) -> None:
        """
        Update the policy at the end of an episode.
        :param kwargs: keyword arguments matching attributes of the policy class
        :return: None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EpsilonGreedy(Policy):

    epsilon: float

    def __init__(self, epsilon: float):
        """
        Initialize the policy.
        :param epsilon: exploration probability
        """
        self.epsilon = epsilon

    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        """
        Epsilon-greedy policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample(), self.epsilon
        else:
            # If all Q-values are the same, choose a random action
            if np.all(Q[state, :]) == Q[state, 0]:
                return env.action_space.sample(), self.epsilon
            return np.argmax(Q[state, :]), self.epsilon


class DecayedEpsilonGreedy(Policy):

    initial_epsilon: float
    min_epsilon: float
    n_episodes: int
    current_episode: int = 0
    linear: bool = False

    def __init__(self, initial_epsilon: float, min_epsilon: float, n_episodes: int, linear: bool = False) -> None:
        """
        Initialize the policy.
        :param initial_epsilon: starting exploration probability
        :param min_epsilon: minimum exploration probability
        :param n_episodes: number of episodes
        :param linear: linear decay
        """
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.n_episodes = n_episodes
        self.linear = linear

    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        """
        Decaying epsilon-greedy policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        if self.linear:
            epsilon = self.initial_epsilon - (self.initial_epsilon - self.min_epsilon) * self.current_episode / self.n_episodes
        else:
            decay_rate = (self.min_epsilon / self.initial_epsilon) ** (1.0 / self.n_episodes)
            epsilon = max(self.min_epsilon, self.initial_epsilon * (decay_rate ** self.current_episode))

        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample(), epsilon
        else:
            # If all Q-values are the same, choose a random action
            if np.all(Q[state, :]) == Q[state, 0]:
                return env.action_space.sample(), epsilon
            return np.argmax(Q[state, :]), epsilon


class Softmax(Policy):

    tau: float

    def __init__(self, tau: float) -> None:
        """
        Initialize the policy.
        :param tau: temperature
        """
        self.tau = tau

    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        """
        Softmax policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        # Compute the probabilities of each action
        probabilities = np.exp(Q[state] / self.tau) / np.sum(np.exp(Q[state] / self.tau))
        # Choose an action according to the probabilities
        return np.random.choice(env.action_space.n, p=probabilities)
