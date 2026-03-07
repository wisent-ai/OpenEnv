"""GRPO training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from constant_definitions.train.grpo_constants import (
    GRPO_BATCH_SIZE,
    GRPO_CHECKPOINT_EVERY,
    GRPO_CURRICULUM_EXPANSION_STEP,
    GRPO_CURRICULUM_INITIAL_GAMES,
    GRPO_GRADIENT_ACCUMULATION_STEPS,
    GRPO_LOG_EVERY,
    GRPO_LR_DENOMINATOR,
    GRPO_LR_NUMERATOR,
    GRPO_MAX_COMPLETION_LENGTH,
    GRPO_NUM_EPOCHS,
    GRPO_NUM_GENERATIONS,
    GRPO_SHAPING_ALPHA_DENOMINATOR,
    GRPO_SHAPING_ALPHA_NUMERATOR,
    GRPO_WARMUP_RATIO_DENOMINATOR,
    GRPO_WARMUP_RATIO_NUMERATOR,
    GRPO_WEIGHT_DECAY_DENOMINATOR,
    GRPO_WEIGHT_DECAY_NUMERATOR,
)


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training."""

    # Core hyperparameters (derived from constants)
    learning_rate_numerator: int = GRPO_LR_NUMERATOR
    learning_rate_denominator: int = GRPO_LR_DENOMINATOR
    batch_size: int = GRPO_BATCH_SIZE
    num_generations: int = GRPO_NUM_GENERATIONS
    num_epochs: int = GRPO_NUM_EPOCHS
    max_completion_length: int = GRPO_MAX_COMPLETION_LENGTH
    gradient_accumulation_steps: int = GRPO_GRADIENT_ACCUMULATION_STEPS

    # Warmup and regularisation
    warmup_ratio_numerator: int = GRPO_WARMUP_RATIO_NUMERATOR
    warmup_ratio_denominator: int = GRPO_WARMUP_RATIO_DENOMINATOR
    weight_decay_numerator: int = GRPO_WEIGHT_DECAY_NUMERATOR
    weight_decay_denominator: int = GRPO_WEIGHT_DECAY_DENOMINATOR

    # Shaping
    shaping_alpha_numerator: int = GRPO_SHAPING_ALPHA_NUMERATOR
    shaping_alpha_denominator: int = GRPO_SHAPING_ALPHA_DENOMINATOR

    # Scheduling
    checkpoint_every: int = GRPO_CHECKPOINT_EVERY
    log_every: int = GRPO_LOG_EVERY
    curriculum_initial_games: int = GRPO_CURRICULUM_INITIAL_GAMES
    curriculum_expansion_step: int = GRPO_CURRICULUM_EXPANSION_STEP

    # Model
    model_name: str = ""
    output_dir: str = "checkpoints/grpo"

    @property
    def learning_rate(self) -> float:
        """Effective learning rate as a float."""
        return self.learning_rate_numerator / self.learning_rate_denominator

    @property
    def warmup_ratio(self) -> float:
        """Effective warmup ratio."""
        return self.warmup_ratio_numerator / self.warmup_ratio_denominator

    @property
    def weight_decay(self) -> float:
        """Effective weight decay."""
        return self.weight_decay_numerator / self.weight_decay_denominator

    @property
    def shaping_alpha(self) -> float:
        """Shaping reward coefficient."""
        return self.shaping_alpha_numerator / self.shaping_alpha_denominator

    def to_trl_kwargs(self) -> dict:
        """Return keyword arguments suitable for TRL GRPOConfig."""
        return {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "num_generations": self.num_generations,
            "num_train_epochs": self.num_epochs,
            "max_completion_length": self.max_completion_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "output_dir": self.output_dir,
            "logging_steps": self.log_every,
            "save_steps": self.checkpoint_every,
        }
