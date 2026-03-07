"""Constants for Group Relative Policy Optimisation (GRPO) training."""

# Learning rate as numerator / denominator (1e-5)
GRPO_LR_NUMERATOR = 1
GRPO_LR_DENOMINATOR = 100_000

# Batch size (number of prompts per optimisation step)
GRPO_BATCH_SIZE = 8

# Number of completions generated per prompt for GRPO grouping
GRPO_NUM_GENERATIONS = 4

# Training epochs over the collected dataset
GRPO_NUM_EPOCHS = 3

# Maximum completion length per round (tokens)
GRPO_MAX_COMPLETION_LENGTH = 64

# Gradient accumulation steps
GRPO_GRADIENT_ACCUMULATION_STEPS = 4

# Warmup ratio (numerator / denominator)
GRPO_WARMUP_RATIO_NUMERATOR = 3
GRPO_WARMUP_RATIO_DENOMINATOR = 100

# Weight decay (numerator / denominator)
GRPO_WEIGHT_DECAY_NUMERATOR = 1
GRPO_WEIGHT_DECAY_DENOMINATOR = 100

# Per-step shaping reward coefficient alpha (numerator / denominator)
GRPO_SHAPING_ALPHA_NUMERATOR = 1
GRPO_SHAPING_ALPHA_DENOMINATOR = 10

# Checkpoint interval (steps)
GRPO_CHECKPOINT_EVERY = 500

# Curriculum: number of base games to start with
GRPO_CURRICULUM_INITIAL_GAMES = 6

# Curriculum: games added per expansion step
GRPO_CURRICULUM_EXPANSION_STEP = 8

# Logging interval (steps)
GRPO_LOG_EVERY = 10
