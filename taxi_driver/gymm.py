import sys
import os
from pathlib import Path

from common.rewards import TaxiDriverRewards

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.params import TaxiDriverParams
from common.algorithms import QLearning, SARSA, BruteForce
from common.policies import DecayedEpsilonGreedy, Random
from common.environments import TaxiDriver
from common.plots import TaxiDriverPlots


def main():
    params = TaxiDriverParams(
        n_episodes=250,
        n_runs=1,  # attention sur le random laissez a 1
        learning_rate=0.85,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.001,
        random_seed=True,
        seed=123,
        max_n_steps=25000,
        savefig_folder=Path("./static/img/taxi_driver/"),
        savemodel_folder=Path("./static/models/taxi_driver/"),
    )

    env = TaxiDriver(params).env

    algorithm = BruteForce(
        env=env,
        params=params,
        policy=Random(),
    )

    algorithm.run()

    TaxiDriverPlots.plot(
        policy=algorithm.computed_policy,
        algorithm=algorithm,
        env=env,
        params=params,
    )


if __name__ == "__main__":
    main()
