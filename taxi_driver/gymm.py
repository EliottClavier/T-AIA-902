import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.params import TaxiDriverParams
from common.algorithms import QLearning
from common.policies import DecayedEpsilonGreedy
from common.environments import TaxiDriver
from common.plots import TaxiDriverPlots


def main():
    params = TaxiDriverParams(
        n_episodes=100,
        n_runs=100,
        learning_rate=0.85,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.001,
        manual_decay_rate=0.001,
        random_seed=True,
        seed=123,
        max_n_steps=100,
    )

    env = TaxiDriver(params).env

    algorithm = QLearning(
        env=env,
        params=params,
        policy=DecayedEpsilonGreedy(
            epsilon=params.epsilon,
            min_epsilon=params.min_epsilon,
            n_episodes=params.n_episodes,
            manual_decay_rate=params.manual_decay_rate,
        ),
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
