"""GRPO (Group Relative Policy Optimisation) training subpackage."""

from train.grpo.config import GRPOConfig
from train.grpo.dataset import trajectories_to_dataset
from train.grpo.trainer import KantGRPOTrainer

__all__ = ["GRPOConfig", "trajectories_to_dataset", "KantGRPOTrainer"]
