"""DPO (Direct Preference Optimisation) training subpackage."""

from train.dpo.config import DPOConfig
from train.dpo.pairs import generate_preference_pairs
from train.dpo.trainer import MachiavelliDPOTrainer

__all__ = ["DPOConfig", "generate_preference_pairs", "MachiavelliDPOTrainer"]
