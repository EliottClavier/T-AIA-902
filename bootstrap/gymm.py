import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.params import FrozenLakeParams
from common.algorithms import QLearning
from common.policies import EpsilonGreedy
from common.environments import FrozenLake
from common.plots import Plots, FrozenLakePlots


def main():
    params = FrozenLakeParams(
        n_episodes=10000,
        n_runs=100,
        learning_rate=0.5,
        gamma=0.90,
        epsilon=0.2,
        min_epsilon=0.0,
        random_seed=False,
        seed=123,
        is_slippery=True,
        proba_frozen=0.9,
        savefig_folder=Path("./static/img/tutorials/"),
    )

    map_sizes = [(8, 8)]
    max_steps = [200]

    for map_size, max_steps in zip(map_sizes, max_steps):
        params.map_size = map_size[0]
        params.max_n_steps = max_steps

        env = FrozenLake(params).env

        algorithm = QLearning(
            env=env,
            params=params,
            policy=EpsilonGreedy(
                epsilon=params.epsilon
            ),
        )

        algorithm.run()

        FrozenLakePlots.plot(
            policy=algorithm.get_current_policy(),
            algorithm=algorithm,
            env=env,
            params=params,
        )


if __name__ == "__main__":
    main()
