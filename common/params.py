from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    """
    Parameters for RL algorithms.
    """

    # Number of episodes
    n_episodes: int

    # Number of runs
    n_runs: int

    # Learning rate
    learning_rate: float

    # Discounting rate
    gamma: float

    # Define a seed so that we get reproducible results
    seed: int

    # Root folder where plots are saved
    savefig_folder: Path

    # Epsilon value for epsilon-greedy policy
    epsilon: float = 1.0

    # Minimum epsilon value
    min_epsilon: float = 0.1

    # Temperature for softmax policy
    tau: float = 1.0

    # Render mode
    render_mode: str = "rgb_array"

    # Maximum number of steps
    max_n_steps: int = None

    # Scale the maximum number of steps based on map size
    scale_n_steps: bool = False

    # Map size
    map_size: int = 4


@dataclass
class FrozenLakeParams(Params):
    """
    Parameters for the FrozenLake environment.
    """

    # If true, agent will randomly slip on the ice. If false, agent will go in the direction it chooses.
    is_slippery: bool = True

    # Probability that a tile is frozen
    proba_frozen: float = 0.9
