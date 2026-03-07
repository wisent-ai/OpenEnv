# --- Bank Run ---
BR_PATIENCE_REWARD = 5            # Payoff for waiting when bank survives
BR_EARLY_WITHDRAW = 3             # Payoff for early withdrawal
BR_BANK_FAIL_PAYOFF = 1           # Payoff when bank collapses

# --- Global Stag Hunt ---
GSH_STAG_PAYOFF = 6               # Mutual stag payoff (higher than normal)
GSH_HARE_PAYOFF = 3               # Hare regardless payoff
GSH_STAG_ALONE_PAYOFF = 0         # Hunting stag alone

# --- Beauty Contest / p-Guessing ---
BC_MAX_NUMBER = 10                 # Range of numbers to choose from
BC_TARGET_FRACTION_NUM = 2         # p = two thirds
BC_TARGET_FRACTION_DEN = 3
BC_WIN_PAYOFF = 5                  # Winner payoff
BC_LOSE_PAYOFF = 0                 # Loser payoff
BC_TIE_PAYOFF = 2                  # Tie payoff

# --- Hawk-Dove-Bourgeois ---
HDB_RESOURCE_VALUE = 6            # Value of contested resource
HDB_FIGHT_COST = 8                # Cost of mutual hawk fight
HDB_SHARE_DIVISOR = 2             # Split resource equally

# --- Gift Exchange ---
GE_MAX_WAGE = 10                   # Maximum wage employer can offer
GE_MAX_EFFORT = 10                 # Maximum effort worker can exert
GE_EFFORT_COST_PER_UNIT = 1       # Marginal cost of effort
GE_PRODUCTIVITY_PER_EFFORT = 2    # Revenue per unit of effort

# --- Moral Hazard ---
MH_BASE_OUTPUT = 3                 # Output without effort
MH_EFFORT_BOOST = 5               # Additional output from effort
MH_EFFORT_COST = 2                # Cost to agent of exerting effort
MH_MAX_BONUS = 10                  # Maximum bonus principal can offer

# --- Screening ---
SCR_HIGH_TYPE_VALUE = 8            # High type's private value
SCR_LOW_TYPE_VALUE = 4             # Low type's private value
SCR_PREMIUM_PRICE = 6             # Premium contract price
SCR_BASIC_PRICE = 3               # Basic contract price
