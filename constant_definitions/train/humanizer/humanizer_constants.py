"""Constants for the automated batch AuthorMist LaTeX humanizer pipeline."""

# Minimum character length for a paragraph to be worth humanizing
MIN_PARAGRAPH_CHARS = 100

# Minimum character length for the humanizer model input
MIN_MODEL_INPUT_CHARS = 50

# Index for last element in split (used for citation key extraction)
LAST_ELEMENT_INDEX = -1

# Zero index
ZERO_INDEX = 0

# Single step
ONE_STEP = 1

# Year century prefixes for citation regex matching
YEAR_PREFIX_TWENTIETH = 19
YEAR_PREFIX_TWENTYFIRST = 20

# Digit count for year suffix
YEAR_SUFFIX_DIGITS = 2

# Similarity ratio threshold: reject humanized text below this
# (prevents accepting truncated or completely rewritten output)
SIMILARITY_LOWER_BOUND_NUMER = 15
SIMILARITY_LOWER_BOUND_DENOM = 100

# Similarity ratio upper bound: reject if too similar (no real change)
SIMILARITY_UPPER_BOUND_NUMER = 98
SIMILARITY_UPPER_BOUND_DENOM = 100

# Minimum ratio of humanized length to original length
# (rejects severely truncated output)
LENGTH_RATIO_FLOOR_NUMER = 60
LENGTH_RATIO_FLOOR_DENOM = 100

# Maximum ratio of humanized length to original length
# (rejects wildly expanded output with prompt leakage)
LENGTH_RATIO_CEILING_NUMER = 160
LENGTH_RATIO_CEILING_DENOM = 100

# Maximum retries per paragraph before keeping original
MAX_RETRIES_PER_PARAGRAPH = 2

# Chunk size for processing long paragraphs (characters)
CHUNK_SIZE_CHARS = 500

# Chunk overlap for context preservation (characters)
CHUNK_OVERLAP_CHARS = 50

# Temperature for AuthorMist generation
TEMPERATURE_NUMER = 7
TEMPERATURE_DENOM = 10

# Top-p nucleus sampling parameter
TOP_P_NUMER = 9
TOP_P_DENOM = 10

# Repetition penalty (scaled by 10 to avoid float)
REPETITION_PENALTY_NUMER = 11
REPETITION_PENALTY_DENOM = 10

# Max token length for model generation
MAX_MODEL_TOKENS = 2048

# Minimum sentence count: reject if humanized has fewer sentences
# than this fraction of original sentence count
MIN_SENTENCE_RATIO_NUMER = 70
MIN_SENTENCE_RATIO_DENOM = 100
