# --- Beer-Quiche Signaling Game ---
BQ_TOUGH_BEER_PAYOFF = 3      # Tough type prefers beer
BQ_TOUGH_QUICHE_PAYOFF = 1    # Tough type dislikes quiche
BQ_WEAK_BEER_PAYOFF = 1       # Weak type dislikes beer
BQ_WEAK_QUICHE_PAYOFF = 3     # Weak type prefers quiche
BQ_CHALLENGE_COST = -2        # Cost of being challenged
BQ_NO_CHALLENGE_BONUS = 2     # Bonus for not being challenged
BQ_CHALLENGE_TOUGH_PAYOFF = -1  # Challenger loses vs tough
BQ_CHALLENGE_WEAK_PAYOFF = 2    # Challenger wins vs weak

# --- Spence Job Market Signaling ---
SPENCE_HIGH_ABILITY = 4       # High-type productivity
SPENCE_LOW_ABILITY = 2        # Low-type productivity
SPENCE_EDU_COST_HIGH = 1      # Education cost for high type
SPENCE_EDU_COST_LOW = 3       # Education cost for low type
SPENCE_HIGH_WAGE = 4          # Wage offered to educated workers
SPENCE_LOW_WAGE = 2           # Wage offered to uneducated workers

# --- Cheap Talk ---
CT_ALIGNED_MATCH = 3          # Both benefit from correct action
CT_ALIGNED_MISMATCH = 0       # Misaligned outcomes
CT_BIAS = 1                   # Sender's preferred deviation

# --- Lemon Market ---
LEMON_GOOD_QUALITY_VALUE = 8  # Buyer value for good car
LEMON_BAD_QUALITY_VALUE = 3   # Buyer value for lemon
LEMON_GOOD_SELLER_COST = 6    # Seller cost for good car
LEMON_BAD_SELLER_COST = 2     # Seller cost for lemon
LEMON_MAX_PRICE = 10          # Maximum price in market

# --- Bayesian Persuasion ---
BP_GOOD_STATE_VALUE = 5       # Value of action in good state
BP_BAD_STATE_PENALTY = -3     # Penalty for action in bad state
BP_SAFE_PAYOFF = 0            # Safe action payoff
BP_REVEAL_COST = 0            # Cost of revealing (zero for sender)
