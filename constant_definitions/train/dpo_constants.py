"""Constants for Direct Preference Optimisation (DPO) training."""

# DPO beta parameter (KL penalty coefficient), as numerator / denominator
DPO_BETA_NUMERATOR = 1
DPO_BETA_DENOMINATOR = 10

# Learning rate as numerator / denominator (5e-6)
DPO_LR_NUMERATOR = 5
DPO_LR_DENOMINATOR = 1_000_000

# Batch size (preference pairs per step)
DPO_BATCH_SIZE = 4

# Training epochs
DPO_NUM_EPOCHS = 1

# Number of trajectories to collect per (game, strategy) pair
DPO_TRAJECTORIES_PER_PAIR = 5

# Quantile threshold for chosen/rejected selection (top / bottom quartile)
DPO_TOP_QUANTILE_NUMERATOR = 1
DPO_TOP_QUANTILE_DENOMINATOR = 4

DPO_BOTTOM_QUANTILE_NUMERATOR = 1
DPO_BOTTOM_QUANTILE_DENOMINATOR = 4

# Minimum reward margin between chosen and rejected (numerator / denominator)
DPO_MIN_REWARD_MARGIN_NUMERATOR = 2
DPO_MIN_REWARD_MARGIN_DENOMINATOR = 10

# Maximum sequence length for DPO (tokens)
DPO_MAX_LENGTH = 512

# Gradient accumulation steps
DPO_GRADIENT_ACCUMULATION_STEPS = 4

# Warmup ratio (numerator / denominator)
DPO_WARMUP_RATIO_NUMERATOR = 5
DPO_WARMUP_RATIO_DENOMINATOR = 100
