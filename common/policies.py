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

    @property
    def description(self) -> str:
        """
        Get the description of the policy.
        :return: description
        """
        return self.__class__.__name__


class EpsilonGreedy(Policy):
    epsilon: float

    def __init__(self, epsilon: float):
        """
        Initialize the policy.
        :param epsilon: exploration probability
        """

        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in the interval [0, 1].")

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

    @property
    def description(self) -> str:
        """
        Get the description of the policy.
        :return: description
        """
        return f"{self.__class__.__name__} - {self.epsilon} ε"


class DecayedEpsilonGreedy(Policy):
    epsilon: float
    min_epsilon: float
    n_episodes: int
    current_episode: int = 0
    linear: bool = False
    manual_decay_rate: float | None = None

    def __init__(self, epsilon: float, min_epsilon: float, n_episodes: int,
                 linear: bool = False, manual_decay_rate: float = None) -> None:
        """
        Initialize the policy.
        :param epsilon: starting exploration probability
        :param min_epsilon: minimum exploration probability
        :param n_episodes: number of episodes
        :param linear: linear decay
        :param manual_decay_rate: manual decay rate
        """

        if epsilon < min_epsilon:
            raise ValueError("Initial epsilon must be greater than or equal to min epsilon.")

        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Initial epsilon must be in the interval (0, 1).")

        if min_epsilon < 0.0 or min_epsilon > 1.0:
            raise ValueError("Min epsilon must be in the interval (0, 1).")

        if manual_decay_rate is not None and (manual_decay_rate < 0.0 or manual_decay_rate > 1.0):
            raise ValueError("Decay rate must be in the interval (0, 1).")

        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.n_episodes = n_episodes
        self.linear = linear
        self.manual_decay_rate = manual_decay_rate

    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        """
        Decaying epsilon-greedy policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        if self.linear:
            epsilon = self.initial_epsilon - (
                        self.initial_epsilon - self.min_epsilon) * self.current_episode / self.n_episodes
        else:
            epsilon = max(self.min_epsilon, self.initial_epsilon * ((1 - self.decay_rate) ** self.current_episode))

        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample(), epsilon
        else:
            # If all Q-values are the same, choose a random action
            if np.all(Q[state, :]) == Q[state, 0]:
                return env.action_space.sample(), epsilon
            return np.argmax(Q[state, :]), epsilon

    @property
    def decay_rate(self) -> float:
        """
        Compute the decay rate.
        :return: decay rate
        """
        return self.manual_decay_rate or (1 - (self.min_epsilon / self.initial_epsilon) ** (1.0 / self.n_episodes))

    @property
    def description(self) -> str:
        """
        Get the description of the policy.
        :return: description
        """
        return (
            f"{self.__class__.__name__} - {self.initial_epsilon} ε max - {self.min_epsilon} ε min"
            f" - {self.decay_rate:.6f} decay rate"
        )


class Softmax(Policy):
    tau: float

    def __init__(self, tau: float) -> None:
        """
        Initialize the policy.
        :param tau: temperature
        """

        if tau < 0.0 or tau > 1.0:
            raise ValueError("Tau must be greater than 0.")

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

    @property
    def description(self) -> str:
        """
        Get the description of the policy.
        :return: description
        """
        return f"{self.__class__.__name__} - {self.tau} τ"


class Random(Policy):
    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> Tuple[int, float]:
        return env.action_space.sample(), 0
