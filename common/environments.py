from abc import abstractmethod

import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from common.params import Params, FrozenLakeParams, TaxiDriverParams
from common.taxi import TaxiEnv


class Game:
    """
    Abstract class for games.
    """

    # Environment
    env: Env

    # Name of the game
    name: str

    def __init__(self, params: Params) -> None:
        """
        Initialize the game.
        :param params: parameters for the game
        :return: None
        """

        self.env = self.make(params)

        self.render_mode = params.render_mode

        if v := params.max_n_steps:
            self.set_max_episode_steps(v)

        if params.scale_n_steps:
            self.set_max_episode_steps(self.scale_max_n_steps(self.env))

    @abstractmethod
    def make(self, params: Params) -> Env:
        """
        Make the game. Method to be implemented by subclasses.
        :param params: parameters for the game
        :return: environment
        """
        pass

    def set_max_episode_steps(self, n: int) -> None:
        """
        Set the maximum number of steps.
        :param n: maximum number of steps
        :return: None
        """
        self.env._max_episode_steps = n

    @staticmethod
    def scale_max_n_steps(env: Env) -> int:
        """
        Scale the maximum number of steps based on the map size.
        :param env: environment
        :return: scaled maximum number of steps
        """
        return env.action_space.n * env.observation_space.n * 2


class FrozenLake(Game):
    """
    FrozenLake game.
    """

    # Name of the game
    name: str = "FrozenLake-v1"

    def make(self, params: FrozenLakeParams) -> Env:
        """
        Make the FrozenLake game.
        :param params: parameters for the game
        :return: environment
        """
        # Get map_size from kwargs if it exists, otherwise default to 4
        return gym.make(
            self.name,
            is_slippery=params.is_slippery,
            render_mode=params.render_mode,
            desc=generate_random_map(
                size=params.map_size[0], p=params.proba_frozen, seed=params.seed
            ),
        )


class TaxiDriver(Game):
    """
    FrozenLake game.
    """

    # Name of the game
    name: str = "Custom-TaxiDriver-v1"

    def make(self, params: TaxiDriverParams) -> Env:
        """
        Make the FrozenLake game.
        :param params: parameters for the game
        :return: environment
        """

        """
        gym.envs.register(
            id=self.name,
            entry_point=TaxiEnv
        )
        """

        return gym.make(
            #self.name,
            "Taxi-v3",
            render_mode=params.render_mode,
        )
