from abc import abstractmethod

import gymnasium as gym
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import RecordVideo, capped_cubic_video_schedule, TimeLimit

from common.params import Params, FrozenLakeParams, TaxiDriverParams


class Game:
    """
    Abstract class for games.
    """

    # Environment
    env: Env

    # Name of the game
    name: str

    def __init__(self, params: Params, should_record: bool = True) -> None:
        """
        Initialize the game.
        :param params: parameters for the game
        :return: None
        """

        self.env = self.make(params)

        self.render_mode = params.render_mode

        # TODO: Could be deleted since we register custom envs
        if v := params.max_n_steps:
            self.set_max_episode_steps(v)

        if params.scale_n_steps:
            self.set_max_episode_steps(self.scale_max_n_steps(self.env))

        if should_record:
            if params.saveepisode_folder.exists():
                for f in params.saveepisode_folder.iterdir():
                    f.unlink()

            self.env = RecordVideo(
                self.env,
                video_folder=str(params.saveepisode_folder),
                episode_trigger=capped_cubic_video_schedule
            )

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
        self.env = TimeLimit(self.env, max_episode_steps=n)

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
    name: str = "Custom-FrozenLake-v1"
    #name: str = "FrozenLake-v1"

    def make(self, params: FrozenLakeParams) -> Env:
        """
        Make the FrozenLake game.
        :param params: parameters for the game
        :return: environment
        """

        gym.envs.register(
            id=self.name,
            entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
            max_episode_steps=params.max_n_steps or 200,
        )

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
    name: str = "Custom-TaxiDriver-v3"
    #name: str = "Taxi-v3"

    def make(self, params: TaxiDriverParams) -> Env:
        """
        Make the FrozenLake game.
        :param params: parameters for the game
        :return: environment
        """

        gym.envs.register(
            id=self.name,
            entry_point='gymnasium.envs.toy_text:TaxiEnv',
            max_episode_steps=params.max_n_steps or 200,
        )

        return gym.make(
            self.name,
            render_mode=params.render_mode,
        )
