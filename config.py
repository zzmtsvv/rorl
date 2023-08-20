import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class rorl_config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "walker2d-medium-v2"
    seed: int = 42
    eval_seed: int = 0
    eval_freq: int = int(1e3)
    num_episodes: int = 10
    max_timesteps: int = int(3e5)
    
    max_action : float = 1.0

    # SAC-N
    num_critics: int = 20
    buffer_size: int = 1_000_000  # Replay buffer size
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate

    # RORL
    epsilon: float = 0.01
    tau_rorl: float = 0.2
    beta_smooth: float = 1e-4
    beta_ood: float = 0.1
    beta_divergence: float = 1.0

    # Wandb logging
    project: str = "RORL"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
