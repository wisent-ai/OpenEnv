"""Numeric and string constants for the metagame arena orchestrator."""

# Phase names
PHASE_COMMUNICATION = "communication"
PHASE_GOVERNANCE = "governance"
PHASE_GAME_SELECTION = "game_selection"
PHASE_PLAY = "play"
PHASE_EVALUATE = "evaluate"

ARENA_PHASES = (
    PHASE_COMMUNICATION,
    PHASE_GOVERNANCE,
    PHASE_GAME_SELECTION,
    PHASE_PLAY,
    PHASE_EVALUATE,
)

# Roster limits
ROSTER_MIN_MODELS = 3
ROSTER_MAX_MODELS = 12

# Round configuration
DEFAULT_TOTAL_ROUNDS = 5
DEFAULT_GAMES_PER_ROUND = 2

# Game pool
DEFAULT_POOL_SIZE = 6

# Governance limits
MAX_PROPOSALS_PER_ROUND = 3

# Proposal types
PROPOSAL_BAN = "ban"
PROPOSAL_ADD = "add"
PROPOSAL_RULE = "rule"
PROPOSAL_NEW_GAME = "new_game"

PROPOSAL_TYPES = (
    PROPOSAL_BAN,
    PROPOSAL_ADD,
    PROPOSAL_RULE,
    PROPOSAL_NEW_GAME,
)

# Voting thresholds (numerator / denominator)
BAN_THRESHOLD_NUMERATOR = 2
BAN_THRESHOLD_DENOMINATOR = 3
RULE_THRESHOLD_NUMERATOR = 1
RULE_THRESHOLD_DENOMINATOR = 2

# Model type labels
MODEL_TYPE_API = "api"
MODEL_TYPE_LOCAL = "local"
MODEL_TYPE_STRATEGY = "strategy"
