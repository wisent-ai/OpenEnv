"""DPO training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from constant_definitions.train.dpo_constants import (
    DPO_BATCH_SIZE,
    DPO_BETA_DENOMINATOR,
    DPO_BETA_NUMERATOR,
    DPO_GRADIENT_ACCUMULATION_STEPS,
    DPO_LR_DENOMINATOR,
    DPO_LR_NUMERATOR,
    DPO_MAX_LENGTH,
    DPO_MIN_REWARD_MARGIN_DENOMINATOR,
    DPO_MIN_REWARD_MARGIN_NUMERATOR,
    DPO_NUM_EPOCHS,
    DPO_TRAJECTORIES_PER_PAIR,
    DPO_WARMUP_RATIO_DENOMINATOR,
    DPO_WARMUP_RATIO_NUMERATOR,
)


@dataclass(frozen=True)
class DPOConfig:
    """Configuration for DPO training."""

    # Core hyperparameters
    beta_numerator: int = DPO_BETA_NUMERATOR
    beta_denominator: int = DPO_BETA_DENOMINATOR
    learning_rate_numerator: int = DPO_LR_NUMERATOR
    learning_rate_denominator: int = DPO_LR_DENOMINATOR
    batch_size: int = DPO_BATCH_SIZE
    num_epochs: int = DPO_NUM_EPOCHS
    max_length: int = DPO_MAX_LENGTH
    gradient_accumulation_steps: int = DPO_GRADIENT_ACCUMULATION_STEPS

    # Warmup
    warmup_ratio_numerator: int = DPO_WARMUP_RATIO_NUMERATOR
    warmup_ratio_denominator: int = DPO_WARMUP_RATIO_DENOMINATOR

    # Pair generation
    trajectories_per_pair: int = DPO_TRAJECTORIES_PER_PAIR
    min_reward_margin_numerator: int = DPO_MIN_REWARD_MARGIN_NUMERATOR
    min_reward_margin_denominator: int = DPO_MIN_REWARD_MARGIN_DENOMINATOR

    # Model
    model_name: str = ""
    output_dir: str = "checkpoints/dpo"

    @property
    def beta(self) -> float:
        """Effective beta (KL penalty coefficient)."""
        return self.beta_numerator / self.beta_denominator

    @property
    def learning_rate(self) -> float:
        """Effective learning rate."""
        return self.learning_rate_numerator / self.learning_rate_denominator

    @property
    def warmup_ratio(self) -> float:
        """Effective warmup ratio."""
        return self.warmup_ratio_numerator / self.warmup_ratio_denominator

    @property
    def min_reward_margin(self) -> float:
        """Minimum reward margin for preference pair filtering."""
        return self.min_reward_margin_numerator / self.min_reward_margin_denominator

    def to_trl_kwargs(self) -> dict:
        """Return keyword arguments suitable for TRL DPOConfig."""
        return {
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "num_train_epochs": self.num_epochs,
            "max_length": self.max_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "output_dir": self.output_dir,
        }
