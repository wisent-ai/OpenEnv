"""Constants for the LLM agent prompt builder and action parser."""

# Maximum tokens for generated action response
MAX_ACTION_TOKENS = 64

# Temperature for training-time generation (numerator / denominator)
TRAIN_TEMPERATURE_NUMERATOR = 7
TRAIN_TEMPERATURE_DENOMINATOR = 10

# Temperature for evaluation-time generation (greedy)
EVAL_TEMPERATURE_NUMERATOR = 0
EVAL_TEMPERATURE_DENOMINATOR = 1

# Top-p sampling parameter (numerator / denominator)
TOP_P_NUMERATOR = 95
TOP_P_DENOMINATOR = 100

# Maximum history rounds shown in prompt (to limit context length)
MAX_PROMPT_HISTORY_ROUNDS = 10

# Section delimiters for structured prompt
PROMPT_SECTION_GAME = "GAME"
PROMPT_SECTION_HISTORY = "HISTORY"
PROMPT_SECTION_SCORES = "SCORES"
PROMPT_SECTION_ACTIONS = "AVAILABLE ACTIONS"
PROMPT_SECTION_INSTRUCTION = "INSTRUCTION"

# Default system prompt (no opponent strategy name -- prevents shortcutting)
SYSTEM_PROMPT = (
    "You are playing a game-theory game. Analyse the situation and choose "
    "the best action. Respond with ONLY the action name, nothing else."
)

# Sentinel returned when LLM output cannot be parsed
PARSE_FAILURE_SENTINEL = "__PARSE_FAILURE__"

# --- N-player prompt section headers ---
NPLAYER_PROMPT_SECTION_PLAYERS = "PLAYERS"
NPLAYER_PROMPT_SECTION_ALL_SCORES = "ALL SCORES"

# --- Coalition prompt section headers ---
COALITION_PROMPT_SECTION_PHASE = "PHASE"
COALITION_PROMPT_SECTION_PROPOSALS = "PENDING PROPOSALS"
COALITION_PROMPT_SECTION_COALITIONS = "ACTIVE COALITIONS"

# --- Governance prompt section headers ---
GOVERNANCE_PROMPT_SECTION_RULES = "GOVERNANCE RULES"
GOVERNANCE_PROMPT_SECTION_PENDING = "PENDING GOVERNANCE"

# N-player system prompt
NPLAYER_SYSTEM_PROMPT = (
    "You are playing an N-player game-theory game. Analyse the situation "
    "and choose the best action. Respond with ONLY the action name, "
    "nothing else."
)

# Coalition system prompt
COALITION_SYSTEM_PROMPT = (
    "You are playing a coalition formation game. You can form coalitions "
    "with other players and propose governance changes. Respond with "
    "valid JSON when negotiating, or ONLY the action name when acting."
)

# Maximum tokens for coalition JSON response
COALITION_MAX_ACTION_TOKENS = 256
