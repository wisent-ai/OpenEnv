"""Constants for deterministic train/eval game split."""

# Seed for reproducible splitting
SPLIT_SEED = 42

# Fraction of games allocated to training (remainder goes to eval).
# Expressed as numerator / denominator to avoid float literals.
TRAIN_FRACTION_NUMERATOR = 78
TRAIN_FRACTION_DENOMINATOR = 100

# Minimum fraction of each domain tag that must appear in eval split.
# Ensures every domain has representation in the held-out set.
MIN_EVAL_TAG_FRACTION_NUMERATOR = 20
MIN_EVAL_TAG_FRACTION_DENOMINATOR = 100
