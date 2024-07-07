import sys
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.params import TaxiDriverParams
from common.algorithms import QLearning, ValueIteration, Algorithm, SARSA, MonteCarlo
from common.policies import DecayedEpsilonGreedy, Max
from common.environments import TaxiDriver


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
        max_n_steps=200,
        theta=0.01,  # Convergence threshold for Value Iteration
        savefig_folder=Path("./static/img/taxi_driver/training_time/"),
        savemodel_folder=Path("./static/models/taxi_driver/"),
    )

    decayed_epsilon_greedy_policy = DecayedEpsilonGreedy(
        epsilon=params.epsilon,
        min_epsilon=params.min_epsilon,
        n_episodes=params.n_episodes,
    )

    training_times = []
    n_episodes_list = [100, 1000, 10000, 50000]

    env = TaxiDriver(params, should_record=False).env
    for i, n_episodes in enumerate(n_episodes_list):
        params.n_episodes = n_episodes
        training_times.append([])

        print("Running for n_episodes:", n_episodes)

        for algorithm in [
            QLearning(env=env, params=params, policy=decayed_epsilon_greedy_policy),
            SARSA(env=env, params=params, policy=decayed_epsilon_greedy_policy),
            MonteCarlo(env=env, params=params, policy=decayed_epsilon_greedy_policy),
        ]:
            print("Running for algorithm:", algorithm.__class__.__name__)
            start = time.time()
            algorithm.run()
            end = time.time()

            training_times[i].append((algorithm.__class__.__name__, end - start))
            env.reset()

    # We train Value Iteration once since it doesn't depend on episodes number
    algorithm = ValueIteration(env=env, params=params, policy=Max())
    start = time.time()
    algorithm.run()
    end = time.time()

    for i, _ in enumerate(n_episodes_list):
        training_times[i].append((algorithm.__class__.__name__, end - start))

    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    # Plot training times in bar chart
    for i, n_episodes in enumerate(n_episodes_list):
        # Plot training times for each algorithm in training_times
        ax[i // 2, i % 2].bar(
            [t[0] for t in training_times[i]],
            [t[1] for t in training_times[i]],
            color=["b", "g", "r", "y"],
        )
        ax[i // 2, i % 2].set_title(f"Training Time for {n_episodes} episodes")
        ax[i // 2, i % 2].set_xlabel("Algorithm")
        ax[i // 2, i % 2].set_ylabel("Training Time (s)")
        ax[i // 2, i % 2].set_ylim([0, max([t[1] for t in training_times[i]]) * 1.1])

        # Annotate the bars
        for j, t in enumerate(training_times[i]):
            ax[i // 2, i % 2].text(j, t[1], f"{t[1]:.2f}", ha="center", va="bottom")

        # Add a horizontal line for Value Iteration
        ax[i // 2, i % 2].axhline(y=training_times[i][-1][1], color="r", linestyle="--")
        ax[i // 2, i % 2].text(
            len(training_times[i]) - 1,
            training_times[i][-1][1],
            f"{training_times[i][-1][1]:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    # Save the plot
    if not params.savefig_folder.exists():
        params.savefig_folder.mkdir(parents=True)
    fig.savefig(params.savefig_folder / "training_time.png")


if __name__ == "__main__":
    main()
