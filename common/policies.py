import random
from abc import abstractmethod
from typing import Tuple

import numpy as np
from gymnasium import Env
from gymnasium.core import ObsType


class Policy:

    @abstractmethod
    def choose_action(
        self, env: Env, state: ObsType, Q: np.ndarray
    ) -> Tuple[int, float]:
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

    def choose_action(
        self, env: Env, state: ObsType, Q: np.ndarray
    ) -> Tuple[int, float]:
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


class DecayedEpsilonGreedy(EpsilonGreedy):
    def __init__(
        self,
        epsilon: float = None,
        min_epsilon: float = 0.01,
        n_episodes: int = 1000,
        initial_epsilon: float = None,
        decay_rate: float = None,
    ):
        self.initial_epsilon = initial_epsilon or epsilon or 1.0
        super().__init__(self.initial_epsilon)
        self.min_epsilon = min_epsilon
        self.n_episodes = n_episodes
        self._decay_rate = decay_rate
        if self._decay_rate is None:
            self._decay_rate = 1 - (self.min_epsilon / self.initial_epsilon) ** (
                1.0 / self.n_episodes
            )

    def on_episode_end(self, current_episode: int):
        self.epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (1 - self._decay_rate) ** current_episode,
        )

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, value: float):
        self._decay_rate = value

    @property
    def description(self) -> str:
        return (
            f"{self.__class__.__name__} - {self.initial_epsilon} ε max - {self.min_epsilon} ε min"
            f" - {self._decay_rate:.6f} decay rate"
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

    def choose_action(
        self, env: Env, state: ObsType, Q: np.ndarray
    ) -> Tuple[int, float]:
        """
        Softmax policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        # Compute the probabilities of each action
        probabilities = np.exp(Q[state] / self.tau) / np.sum(
            np.exp(Q[state] / self.tau)
        )
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
    def choose_action(
        self, env: Env, state: ObsType, Q: np.ndarray
    ) -> Tuple[int, float]:
        return env.action_space.sample(), 0


class Max(Policy):

    def choose_action(self, env: Env, state: ObsType, Q: np.ndarray) -> int:
        """
        Epsilon-greedy policy.
        :param env: environment
        :param state: current state
        :param Q: Q-table
        :return: action
        """
        return np.argmax(Q[state])

    @property
    def description(self) -> str:
        """
        Get the description of the policy.
        :return: description
        """
        return f"{self.__class__.__name__}"
