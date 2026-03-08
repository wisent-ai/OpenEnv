"""Configuration for self-play GRPO training."""

from __future__ import annotations

from dataclasses import dataclass

from constant_definitions.train.grpo_constants import (
    GRPO_BATCH_SIZE,
    GRPO_LR_DENOMINATOR,
    GRPO_LR_NUMERATOR,
    GRPO_MAX_COMPLETION_LENGTH,
    GRPO_NUM_GENERATIONS,
)
from constant_definitions.var.meta.self_play_constants import (
    SELF_PLAY_DEFAULT_EPISODES_PER_STEP,
    SELF_PLAY_DEFAULT_MAX_STEPS,
    SELF_PLAY_OPPONENT_UPDATE_INTERVAL,
    SELF_PLAY_POOL_MAX_SIZE,
    SELF_PLAY_WARMUP_EPISODES,
)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play GRPO training.

    Combines self-play-specific settings (opponent pool management,
    update frequency) with standard GRPO training parameters.
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "./kantbench-self-play"

    # Self-play specific
    opponent_update_interval: int = SELF_PLAY_OPPONENT_UPDATE_INTERVAL
    pool_max_size: int = SELF_PLAY_POOL_MAX_SIZE
    episodes_per_step: int = SELF_PLAY_DEFAULT_EPISODES_PER_STEP
    warmup_episodes: int = SELF_PLAY_WARMUP_EPISODES

    # GRPO params
    learning_rate_numerator: int = GRPO_LR_NUMERATOR
    learning_rate_denominator: int = GRPO_LR_DENOMINATOR
    batch_size: int = GRPO_BATCH_SIZE
    num_generations: int = GRPO_NUM_GENERATIONS
    max_completion_length: int = GRPO_MAX_COMPLETION_LENGTH
    max_steps: int = SELF_PLAY_DEFAULT_MAX_STEPS

    # Cross-model mode: if set, opponent is loaded from this path
    cross_model_path: str = ""

    @property
    def learning_rate(self) -> float:
        """Compute learning rate from numerator/denominator."""
        return self.learning_rate_numerator / self.learning_rate_denominator
