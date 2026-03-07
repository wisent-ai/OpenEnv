# --- Dictator Game ---
DICTATOR_ENDOWMENT = 10   # Amount the dictator allocates

# --- Centipede Game ---
CENTIPEDE_INITIAL_POT = 4         # Starting pot size
CENTIPEDE_GROWTH_MULTIPLIER = 2   # Pot multiplier each pass
CENTIPEDE_MAX_STAGES = 6          # Maximum number of stages
CENTIPEDE_LARGE_SHARE_NUMERATOR = 3   # Large share = pot * 3/4
CENTIPEDE_LARGE_SHARE_DENOMINATOR = 4
CENTIPEDE_SMALL_SHARE_NUMERATOR = 1   # Small share = pot * 1/4
CENTIPEDE_SMALL_SHARE_DENOMINATOR = 4

# --- Stackelberg Competition ---
STACKELBERG_DEMAND_INTERCEPT = 12   # a in P = a - b*Q
STACKELBERG_DEMAND_SLOPE = 1        # b in P = a - b*Q
STACKELBERG_MARGINAL_COST = 2       # Constant marginal cost c
STACKELBERG_MAX_QUANTITY = 10       # Max production quantity
