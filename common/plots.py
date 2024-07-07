import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium import Env

from common.algorithms import Algorithm, AlgorithmHistory
from common.params import Params
from common.rewards import Rewards, FrozenLakeRewards, TaxiDriverRewards

sns.set_theme()


class Plots:
    rewards: Rewards = Rewards()

    @staticmethod
    def plot(policy: np.ndarray, env: Env, algorithm: Algorithm, params: Params) -> None:
        """
        Plot the results of the algorithm, the policy, results of games and more. Should be implemented by subclasses.
        :param policy: policy to plot
        :param env: environment
        :param algorithm: algorithm used
        :param params: parameters used
        :return: None
        """
        pass

    @staticmethod
    def plot_last_frame(env: Env, ax: plt.Axes) -> None:
        """
        Plot the last frame of the environment
        :param env: environment
        :param ax: axis to plot on
        :return: None
        """

        ax.imshow(env.render())
        ax.axis("off")
        ax.set_title("Last frame")

    @staticmethod
    def plot_games_results(n_losses: int, n_runs: int, ax: plt.Axes) -> None:
        """
        Plot the number of wins and losses over the games
        :param n_losses: number of losses
        :param n_runs: number of games
        :param ax: axis to plot on
        :return: None
        """

        wins_losses = [n_runs - n_losses, n_losses]
        ax.bar(["Win", "Loss"], wins_losses)
        ax.set_ylabel("Games")
        ax.set_title(
            f"Win vs Losses ({wins_losses[0]}/{n_runs} games won, {(1 - n_losses / n_runs) * 100:.2f}% win rate )")

    @staticmethod
    def plot_games_steps(steps: list, steps_when_winning: list, n_runs: int, ax: plt.Axes) -> None:
        """
        Plot the number of steps over the games
        :param steps: list of steps
        :param n_runs: number of games
        :param ax: axis to plot on
        :return: None
        """

        ax.plot(steps)
        ax.set_xlabel("Game")
        ax.set_ylabel("Steps")
        ax.set_title(
            f"Number of steps over the games "
            f"\n{round(sum(steps) / n_runs)} steps in average"
            f"\n{round(sum(steps_when_winning) / (len(steps_when_winning) or 1))} steps in average when winning"
        )

    @staticmethod
    def plot_epsilons(epsilons: list, ax: plt.Axes) -> None:
        """
        Plot the epsilon value over the episodes
        :param epsilons: list of epsilon values
        :param ax: axis to plot on
        :return: None
        """

        ax.plot(epsilons)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon over episodes")

    @staticmethod
    def plot_average_rewards(average_rewards: list, ax: plt.Axes) -> None:
        """
        Plot the average rewards over the episodes
        :param average_rewards: list of average rewards
        :param ax: axis to plot on
        :return: None
        """

        ax.plot(average_rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average reward")
        ax.set_title("Average reward over episodes")

    @staticmethod
    def plot_cumulative_rewards(history: AlgorithmHistory, ax: plt.Axes):
        """
        Plot the cumulative rewards over the episodes
        :param history: history of the algorithm
        :param ax: axis to plot on
        :return: None
        """

        cumulative_rewards = [sum(episode.rewards) for episode in history.episodes_histories]

        ax.plot(cumulative_rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative reward over episodes")

    @staticmethod
    def plot_episodes_number_of_steps(history: AlgorithmHistory, ax: plt.Axes) -> None:
        """
        Plot the number of steps over the episodes
        :param history: history of the algorithm
        :param ax: axis to plot on
        :return: None
        """

        steps = [len(episode.actions) for episode in history.episodes_histories]

        ax.plot(steps)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Number of steps over episodes")

    @staticmethod
    def plot_history_heatmap(history: AlgorithmHistory, params: Params, env: Env, ax: plt.Axes) -> None:
        # Plot the states the agent has the most visited
        states = np.zeros(env.observation_space.n)
        for episode in history.episodes_histories:
            for state in episode.states:
                states[state] += 1

        sns.heatmap(
            states.reshape(params.map_size[0], params.map_size[1]),
            # annot=policy_directions,
            fmt="",
            ax=ax,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
        ).set(title="States visited\nLighter color means more visits")

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        ax.set_xlabel("States")
        ax.set_ylabel("Visits")


class FrozenLakePlots(Plots):
    rewards: Rewards = FrozenLakeRewards()

    @staticmethod
    def plot(policy: np.ndarray, env: Env, algorithm: Algorithm, params: Params) -> None:
        """
        Plot the results of the algorithm, the policy, results of games and more
        :param policy: policy to plot
        :param env: environment
        :param algorithm: algorithm used
        :param params: parameters used
        :return: None
        """
        fig, ax = plt.subplots(4, 2, figsize=(16, 16))
        fig.suptitle(params.run_description)

        #Plots.plot_last_frame(env, ax[0][0])
        FrozenLakePlots.plot_policy_map(policy, params, ax[0][1])

        Plots.plot_epsilons(algorithm.historic.epsilons, ax[1][0])
        Plots.plot_average_rewards(algorithm.historic.average_rewards, ax[1][1])

        Plots.plot_episodes_number_of_steps(algorithm.historic, ax[2][0])
        Plots.plot_history_heatmap(algorithm.historic, params, env, ax[2][1])

        steps, losses = algorithm.evaluate(FrozenLakePlots.rewards)
        steps_when_winning = [steps[i] for i in range(len(steps)) if not losses[i]]
        n_losses = len([loss for loss in losses if loss])

        Plots.plot_games_results(n_losses, params.n_runs, ax[3][0])
        Plots.plot_games_steps(steps, steps_when_winning, params.n_runs, ax[3][1])

        plt.tight_layout()
        plt.show()

        if not os.path.exists(params.savefig_folder):
            os.makedirs(params.savefig_folder)
        fig.savefig(params.savefig_folder / params.run_name, bbox_inches="tight")

    @staticmethod
    def get_policy_directions(policy: np.ndarray, map_size: Tuple[int]) -> np.ndarray:
        """
        Convert a policy to a matrix of directions
        :param policy: policy
        :param map_size: size of the map
        :return: policy directions
        """

        directions = ['←', '↓', '→', '↑']

        policy_directions = [directions[action] for action in policy]
        policy_directions = np.array(policy_directions).reshape(map_size[0], map_size[1])
        return policy_directions

    @staticmethod
    def plot_policy_map(policy: np.ndarray, params: Params, ax: plt.Axes) -> None:
        """
        Plot a given policy map
        :param policy: policy to plot
        :param params: parameters
        :param ax: axis to plot on
        :return: None
        """

        policy_directions = FrozenLakePlots.get_policy_directions(policy, params.map_size)

        sns.heatmap(
            policy.reshape(params.map_size[0], params.map_size[1]),
            annot=policy_directions,
            fmt="",
            ax=ax,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title="Learned policy\nArrow indicates the action to take")

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")


class TaxiDriverPlots(Plots):
    rewards: Rewards = TaxiDriverRewards()

    @staticmethod
    def plot(policy: np.ndarray, env: Env, algorithm: Algorithm, params: Params) -> None:
        """
        Plot the results of the algorithm, the policy, results of games and more
        :param policy: policy to plot
        :param env: environment
        :param algorithm: algorithm used
        :param params: parameters used
        :return: None
        """
        fig, ax = plt.subplots(3, 2, figsize=(16, 16))
        fig.suptitle(params.run_description)

        Plots.plot_epsilons(algorithm.historic.epsilons, ax[0][0])
        # Plots.plot_history_heatmap(algorithm.historic, params, env, ax[0][1])

        Plots.plot_episodes_number_of_steps(algorithm.historic, ax[1][0])
        Plots.plot_cumulative_rewards(algorithm.historic, ax[1][1])

        steps, losses = algorithm.evaluate(TaxiDriverPlots.rewards)
        steps_when_winning = [steps[i] for i in range(len(steps)) if not losses[i]]
        n_losses = len([loss for loss in losses if loss])

        Plots.plot_games_results(n_losses, params.n_runs, ax[2][0])
        Plots.plot_games_steps(steps, steps_when_winning, params.n_runs, ax[2][1])

        plt.tight_layout()
        plt.show()

        if not os.path.exists(params.savefig_folder):
            os.makedirs(params.savefig_folder)

        fig.savefig(params.savefig_folder / params.run_name, bbox_inches="tight")

    @staticmethod
    def plot_brute_force_plot(policy: np.ndarray, env: Env, algorithm: Algorithm, params: Params) -> None:
        """
        Plot the results of the algorithm, the policy, results of games and more
        :param policy: policy to plot
        :param env: environment
        :param algorithm: algorithm used
        :param params: parameters used
        :return: None
        """

        fig, ax = plt.subplots(1, 2, figsize=(16, 16))
        fig.suptitle(params.run_description)

        steps = [len(episode.actions) for episode in algorithm.historic.episodes_histories]
        ax[0].hist(steps)
        ax[0].set_xlabel("Steps")
        ax[0].set_title("Number of steps over episodes")

        cumulative_rewards = [sum(episode.rewards) for episode in algorithm.historic.episodes_histories]
        ax[1].hist(cumulative_rewards)
        ax[1].set_xlabel("Cumulative reward")
        ax[1].set_title("Cumulative reward over episodes")

        plt.tight_layout()
        plt.show()

        if not os.path.exists(params.savefig_folder):
            os.makedirs(params.savefig_folder)

        fig.savefig(params.savefig_folder / params.run_name, bbox_inches="tight")
