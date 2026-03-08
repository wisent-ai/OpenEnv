# Optional PD -- exit gives a safe payoff between CC and DD
OPD_EXIT_PAYOFF = 2

# Asymmetric PD -- first player has alibi advantage
APD_A_TEMPTATION = 5
APD_A_REWARD = 3
APD_A_PUNISHMENT = 2
APD_A_SUCKER = 1
APD_B_TEMPTATION = 5
APD_B_REWARD = 3
APD_B_PUNISHMENT = 1
APD_B_SUCKER = 0

# Donation Game -- pay cost c to give benefit b to opponent
DONATION_BENEFIT = 5
DONATION_COST = 2

# Friend or Foe (game show) -- both defect yields zero unlike PD
FOF_SHARE_PAYOFF = 1
FOF_STEAL_WIN_PAYOFF = 2

# Peace-War Game (arms race framing of PD)
PW_DISARM_DISARM = 4
PW_DISARM_ARM = -1
PW_ARM_DISARM = 6
PW_ARM_ARM = 0

# Composable variant system -- name strings and prefixes
VARIANT_CHEAP_TALK = "cheap_talk"
VARIANT_EXIT = "exit"
VARIANT_BINDING_COMMITMENT = "binding_commitment"
VARIANT_NOISY_ACTIONS = "noisy_actions"
VARIANT_NOISY_PAYOFFS = "noisy_payoffs"
CT_MSG_PREFIX = "msg"
CT_SEPARATOR = "_"
BC_COMMIT_PREFIX = "commit"
BC_FREE_PREFIX = "free"
EXIT_ACTION = "exit"

# Bayesian variant parameters
DEFAULT_TREMBLE_PROB_NUMERATOR = 1
DEFAULT_TREMBLE_PROB_DENOMINATOR = 10
DEFAULT_NOISE_SCALE_NUMERATOR = 1
DEFAULT_NOISE_SCALE_DENOMINATOR = 2
