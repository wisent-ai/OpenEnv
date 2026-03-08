# Reputation and gossip variant constants

# Variant names
VARIANT_GOSSIP = "gossip"
VARIANT_MEMORY = "memory"

# Gossip ratings (single tokens, no underscores)
RATING_TRUSTWORTHY = "trustworthy"
RATING_UNTRUSTWORTHY = "untrustworthy"
RATING_NEUTRAL = "neutral"
DEFAULT_RATINGS = (RATING_TRUSTWORTHY, RATING_UNTRUSTWORTHY, RATING_NEUTRAL)

# Action prefixes
GOSSIP_PREFIX = "gossip"
GOSSIP_SEPARATOR = "_"
GOSSIP_SPLIT_LIMIT = 2

# Reputation defaults (numerator / denominator)
DEFAULT_REPUTATION_SCORE_NUMERATOR = 5
DEFAULT_REPUTATION_SCORE_DENOMINATOR = 10
REPUTATION_DECAY_NUMERATOR = 9
REPUTATION_DECAY_DENOMINATOR = 10

# Metadata keys
META_KEY_REPUTATION = "opponent_reputation"
META_KEY_GOSSIP_HISTORY = "gossip_history"
META_KEY_INTERACTION_COUNT = "interaction_count"
META_KEY_COOPERATION_RATE = "cooperation_rate"

# Cognee dataset name
COGNEE_DATASET_NAME = "kant_interactions"
COGNEE_SEARCH_TYPE = "GRAPH_COMPLETION"
