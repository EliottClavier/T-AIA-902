from abc import abstractmethod
from dataclasses import dataclass
from typing import SupportsFloat

from gymnasium import Env


class RewardFunction:
    """
    Abstract class for reward functions.
    """

    @abstractmethod
    def compute(self, env: Env, terminated: bool, truncated: bool, reward: SupportsFloat, state: int, next_state: int):
        """
        Compute the reward.
        :param env: environment
        :param terminated: agent has reached the goal
        :param truncated: agent has fallen into a hole or took too many steps
        :param reward: reward received by the agent
        :param state: current state
        :param next_state: next state
        :return: computed reward
        """
        pass


@dataclass
class FrozenLakeRewards:
    """
    Default rewards for the FrozenLake environment.
    """
    win_reward: float = 1.0
    lose_reward: float = 0.0
    playing_reward: float = 0.0


class FrozenLakeTweakedRewardFunction(RewardFunction, FrozenLakeRewards):
    """
    Custom reward function for the FrozenLake environment.
    """

    tweaked_win_reward: float
    tweaked_lose_reward: float
    tweaked_playing_reward: float

    # Should reward be computed relative to the state ?
    relative: bool = False

    def __init__(
            self, tweaked_win_reward: float = 10.0, tweaked_lose_reward: float = -100.0,
            tweaked_playing_reward: float = 1.0, relative: bool = False
    ) -> None:
        """
        Initialize the reward function.
        :param tweaked_win_reward: tweaked reward for winning
        :param tweaked_lose_reward: tweaked reward for losing
        :param tweaked_playing_reward: tweaked reward when agent is playing
        :param relative: should reward be computed relative to the state
        :return: None
        """
        self.tweaked_win_reward = tweaked_win_reward
        self.tweaked_lose_reward = tweaked_lose_reward
        self.tweaked_playing_reward = tweaked_playing_reward
        self.relative = relative

    def compute(self, env: Env, terminated: bool, truncated: bool, reward: SupportsFloat, state: int, next_state: int):
        """
        Custom reward function for the FrozenLake environment.
        :param env: environment
        :param terminated: agent has reached the goal
        :param truncated: agent has fallen into a hole or took too many steps
        :param reward: reward received by the agent
        :param state: current state
        :param next_state: next state
        :return: computed reward
        """
        if terminated or truncated:
            # If the agent falls into a hole, we penalize it
            if reward == self.lose_reward:
                # Compute negative reward depending on how far the agent is from the goal
                return self.tweaked_lose_reward * next_state / env.observation_space.n if self.relative else self.tweaked_lose_reward

            # If the agent reaches the goal, we reward it
            elif reward == self.win_reward:
                return self.tweaked_win_reward

        # If the agent moves to a different state and doesn't fall into a hole, we reward it
        elif reward == self.playing_reward and next_state != state:
            return self.tweaked_playing_reward * next_state / env.observation_space.n if self.relative else self.tweaked_playing_reward
        # If no condition is met, we reward the agent normally
        return reward
