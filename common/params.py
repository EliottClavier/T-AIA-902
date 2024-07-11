from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from common.policies import DecayedEpsilonGreedy, EpsilonGreedy, Softmax


@dataclass
class Params:
    """
    Parameters for RL algorithms.
    """

    # If true, set a random seed
    random_seed: bool = field(
        default=True,
        metadata={"description": "Should the run use a random seed ?", "type": bool},
    )

    # Define a seed so that we get reproducible results
    seed: int = field(
        default=42,
        metadata={
            "description": "Fixed seed",
            "type": int,
            "prerequisite": ("random_seed", False),
        },
    )

    # Number of episodes
    n_episodes: int = field(
        default=10000, metadata={"description": "Number of episodes", "type": int}
    )

    # Number of runs
    n_runs: int = field(
        default=100,
        metadata={"description": "Number of runs to evaluate the agent", "type": int},
    )

    # Scale the maximum number of steps based on map size
    scale_n_steps: bool = field(
        default=False,
        metadata={
            "description": "Scale the maximum number of steps based on map size ?",
            "type": bool,
        },
    )

    # Maximum number of steps
    max_n_steps: Optional[int] = field(
        default=None,
        metadata={
            "description": "[Optional] Maximum number of steps",
            "type": int,
            "optional": True,
            "prerequisite": ("scale_n_steps", False),
        },
    )

    # Learning rate
    learning_rate: float = field(
        default=0.85,
        metadata={
            "description": "Learning rate (between 0 and 1)",
            "type": float,
            "min": 0.0,
            "max": 1.0,
        },
    )

    # Discounting rate
    gamma: float = field(
        default=0.99,
        metadata={
            "description": "Discounting rate (between 0 and 1)",
            "type": float,
            "min": 0.0,
            "max": 1.0,
        },
    )

    # Deep Q-Learning specific parameters
    batch_size: int = field(
        default=32,
        metadata={
            "description": "Batch size for Deep Q-Learning",
            "type": int,
            "min": 1,
        },
    )

    update_target_every: int = field(
        default=100,
        metadata={
            "description": "Update target network every n steps",
            "type": int,
            "min": 1,
        },
    )

    memory_size: int = field(
        default=10000,
        metadata={"description": "Size of replay memory", "type": int, "min": 100},
    )

    hidden_size: int = field(
        default=64,
        metadata={
            "description": "Size of hidden layers in the neural network",
            "type": int,
            "min": 16,
        },
    )

    learning_rate: float = field(
        default=0.001,
        metadata={
            "description": "Learning rate for the optimizer",
            "type": float,
            "min": 0.0001,
            "max": 0.1,
        },
    )

    # Epsilon value for epsilon-greedy policy
    epsilon: float = field(
        default=1.0,
        metadata={
            "description": "Epsilon value for epsilon-greedy policies (between 0 and 1)",
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "prerequisite": (
                "policy",
                (EpsilonGreedy.__name__, DecayedEpsilonGreedy.__name__),
            ),
        },
    )

    # Minimum epsilon value
    min_epsilon: float = field(
        default=0.001,
        metadata={
            "description": "Minimum epsilon value reached when decaying epsilon (between 0 and 1)",
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "prerequisite": ("policy", DecayedEpsilonGreedy.__name__),
        },
    )

    # Manual decay rate for epsilon (float or None)
    manual_decay_rate: Optional[float] = field(
        default=None,
        metadata={
            "description": "[Optional] Manual decay rate for epsilon (between 0 and 1)",
            "type": float,
            "optional": True,
            "min": 0.0,
            "max": 1.0,
            "prerequisite": ("policy", DecayedEpsilonGreedy.__name__),
        },
    )

    # Temperature for softmax policy
    tau: float = field(
        default=0.5,
        metadata={
            "description": "Tau for softmax policy (between 0 and 1)",
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "prerequisite": ("policy", Softmax.__name__),
        },
    )

    # Theta for Value Iteration
    theta: float = field(
        default=0.01,
        metadata={
            "description": "Convergence threshold (between 0 and 1) for Value Iteration",
            "type": float,
            "min": 0.0,
            "max": 1.0,
        },
    )

    # Render mode
    render_mode: str = field(
        default="rgb_array",
        metadata={"description": "Render mode", "type": str, "configurable": False},
    )

    # Map size
    map_size: Tuple[int] = field(
        default=(0, 0),
        metadata={"description": "Map size", "type": tuple, "configurable": False},
    )

    # Run name, serves for files naming
    run_name: str = field(
        default="name",
        metadata={"description": "Run name", "type": str, "configurable": False},
    )

    # Run description, serves for plot titles
    run_description: str = field(
        default="description",
        metadata={"description": "Run description", "type": str, "configurable": False},
    )

    # Root folder where plots are saved
    savefig_folder: Path = field(
        default=Path("./static/img/"),
        metadata={
            "description": "Root folder where plots are saved",
            "type": Path,
            "configurable": False,
        },
    )

    # Root folder where models are saved
    savemodel_folder: Path = field(
        default=Path("./static/models/"),
        metadata={
            "description": "Root folder where models are saved",
            "type": Path,
            "configurable": False,
        },
    )

    saveepisode_folder: Path = field(
        default=Path("./static/episodes/"),
        metadata={
            "description": "Root folder where episodes are saved",
            "type": Path,
            "configurable": False,
        },
    )


@dataclass
class FrozenLakeParams(Params):
    """
    Parameters for the FrozenLake environment.
    """

    # If true, agent will randomly slip on the ice. If false, agent will go in the direction it chooses.
    is_slippery: bool = field(
        default=True,
        metadata={
            "description": "Is map slippery ?",
            "type": bool,
        },
    )

    # Probability that a tile is frozen
    proba_frozen: float = field(
        default=1.0,
        metadata={
            "description": "Probability that a tile is frozen",
            "type": float,
            "min": 0.0,
            "max": 1.0,
        },
    )

    # Map size
    map_size: Tuple[int] = field(
        default=(4, 4),
        metadata={
            "description": "Map size",
            "type": tuple,
        },
    )

    # Root folder where plots are saved
    savefig_folder: Path = field(
        default=Path("./static/img/frozen_lake/"),
        metadata={
            "description": "Root folder where plots are saved",
            "type": Path,
            "configurable": False,
        },
    )

    # Root folder where models are saved
    savemodel_folder: Path = field(
        default=Path("./static/models/frozen_lake/"),
        metadata={
            "description": "Root folder where models are saved",
            "type": Path,
            "configurable": False,
        },
    )

    saveepisode_folder: Path = field(
        default=Path("./static/episodes/frozen_lake/"),
        metadata={
            "description": "Root folder where episodes are saved",
            "type": Path,
            "configurable": False,
        },
    )


@dataclass
class TaxiDriverParams(Params):
    """
    Parameters for the TaxiDriver environment.
    """

    # Map size
    map_size: Tuple[int] = field(
        default=(5, 5),
        metadata={"description": "Map size", "type": Tuple[int], "configurable": False},
    )

    # Root folder where plots are saved
    savefig_folder: Path = field(
        default=Path("./static/img/taxi_driver/"),
        metadata={
            "description": "Root folder where plots are saved",
            "type": Path,
            "configurable": False,
        },
    )

    # Root folder where models are saved
    savemodel_folder: Path = field(
        default=Path("./static/models/taxi_driver/"),
        metadata={
            "description": "Root folder where models are saved",
            "type": Path,
            "configurable": False,
        },
    )

    # Root folder where episodes are saved
    saveepisode_folder: Path = field(
        default=Path("./static/episodes/taxi_driver/"),
        metadata={
            "description": "Root folder where episodes are saved",
            "type": Path,
            "configurable": False,
        },
    )
