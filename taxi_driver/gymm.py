import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.params import TaxiDriverParams
from common.algorithms import QLearning, ValueIteration
from common.policies import DecayedEpsilonGreedy, Max
from common.environments import TaxiDriver
from common.plots import TaxiDriverPlots


def main():
    params = TaxiDriverParams(
        n_episodes=10000,
        n_runs=10,
        learning_rate=0.85,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.001,
        manual_decay_rate=0.001,
        random_seed=True,
        seed=123,
        max_n_steps=100,
        theta=0.01,  # Convergence threshold for Value Iteration
        savefig_folder=Path("./static/img/taxi_driver/"),
        savemodel_folder=Path("./static/models/taxi_driver/"),
    )

    env = TaxiDriver(params).env

    # algorithm = QLearning(
    #     env=env,
    #     params=params,
    #     policy=DecayedEpsilonGreedy(
    #         epsilon=params.epsilon,
    #         min_epsilon=params.min_epsilon,
    #         n_episodes=params.n_episodes,
    #         manual_decay_rate=params.manual_decay_rate,
    #     ),
    # )

    algorithm = ValueIteration(
        env=env,
        params=params,
        policy=Max() 
    )

    algorithm.run()

    env = TaxiDriver(params, should_record=False).env

    TaxiDriverPlots.plot(
        policy=algorithm.computed_policy,
        algorithm=algorithm,
        env=env,
        params=params,
    )


if __name__ == "__main__":
    main()
