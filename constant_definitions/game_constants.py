# Pydantic / model defaults
DEFAULT_ZERO_FLOAT = float()
DEFAULT_ZERO_INT = int()
DEFAULT_FALSE = False
DEFAULT_NONE = None
MIN_STEP_COUNT = int()

# Episode configuration
DEFAULT_NUM_ROUNDS = 10
SINGLE_SHOT_ROUNDS = 1
DEFAULT_TWO_PLAYERS = 2

# --- Prisoner's Dilemma payoffs ---
PD_CC_PAYOFF = 3  # Both cooperate
PD_CD_PAYOFF = 0  # I cooperate, they defect
PD_DC_PAYOFF = 5  # I defect, they cooperate
PD_DD_PAYOFF = 1  # Both defect

# --- Stag Hunt payoffs ---
SH_SS_PAYOFF = 4  # Both hunt stag
SH_SH_PAYOFF = 0  # I hunt stag, they hunt hare
SH_HS_PAYOFF = 3  # I hunt hare, they hunt stag
SH_HH_PAYOFF = 2  # Both hunt hare

# --- Hawk-Dove payoffs ---
HD_HH_PAYOFF = -1  # Both hawk (conflict)
HD_HD_PAYOFF = 3   # I hawk, they dove
HD_DH_PAYOFF = 1   # I dove, they hawk
HD_DD_PAYOFF = 2   # Both dove

# --- Ultimatum Game ---
ULTIMATUM_POT = 10

# --- Trust Game ---
TRUST_MULTIPLIER = 3
TRUST_ENDOWMENT = 10

# --- Public Goods Game ---
PG_MULTIPLIER_NUMERATOR = 3
PG_MULTIPLIER_DENOMINATOR = 2
PG_ENDOWMENT = 20
PG_DEFAULT_NUM_PLAYERS = 4

# --- Strategy parameters ---
GENEROUS_TFT_COOPERATION_PROB = 9  # out of 10 (90%)
GENEROUS_TFT_DENOMINATOR = 10
ADAPTIVE_THRESHOLD_NUMERATOR = 1
ADAPTIVE_THRESHOLD_DENOMINATOR = 2
MIXED_STRATEGY_COOPERATE_PROB_NUMERATOR = 7
MIXED_STRATEGY_COOPERATE_PROB_DENOMINATOR = 10

# Ultimatum strategy defaults
ULTIMATUM_FAIR_OFFER = 5
ULTIMATUM_LOW_OFFER = 3
ULTIMATUM_HIGH_OFFER = 7
ULTIMATUM_ACCEPT_THRESHOLD = 3

# Trust strategy defaults
TRUST_FAIR_RETURN_NUMERATOR = 1
TRUST_FAIR_RETURN_DENOMINATOR = 3
TRUST_GENEROUS_RETURN_NUMERATOR = 1
TRUST_GENEROUS_RETURN_DENOMINATOR = 2

# Public goods strategy defaults
PG_FAIR_CONTRIBUTION_NUMERATOR = 1
PG_FAIR_CONTRIBUTION_DENOMINATOR = 2
PG_FREE_RIDER_CONTRIBUTION = 2

# Port
SERVER_PORT = 8000

# Max concurrent environments
MAX_CONCURRENT_ENVS = 1

# --- Opponent mode ---
OPPONENT_MODE_STRATEGY = "strategy"
OPPONENT_MODE_SELF = "self_play"
OPPONENT_MODE_CROSS = "cross_model"

# --- Evaluation module constants ---
EVAL_ZERO = 0
EVAL_ONE = 1
EVAL_TWO = 2
EVAL_THREE = 3
EVAL_FOUR = 4
EVAL_DEFAULT_EPISODES = 3
EVAL_HUNDRED = 100
EVAL_INDENT_SPACES = 4
EVAL_PERFECT_SCORE = 1
EVAL_ZERO_FLOAT = 0.0
EVAL_ONE_FLOAT = 1.0
EVAL_HALF = 0.5
EVAL_NEGATIVE_ONE = -1

# --- N-player / coalition evaluation constants ---
NPLAYER_EVAL_DEFAULT_EPISODES = 3
COALITION_EVAL_DEFAULT_EPISODES = 3

# --- External benchmark constants ---
EVAL_EIGHT = 8
EVAL_TEN = 10
EVAL_EIGHTY = 80
EVAL_FIVE_TWELVE = 512
