# --- Shapley Value Allocation ---
SHAPLEY_GRAND_COALITION_VALUE = 12   # v({all players})
SHAPLEY_SINGLE_VALUE = 2            # v({single player})
SHAPLEY_PAIR_VALUE = 8              # v({pair})
SHAPLEY_MAX_CLAIM = 12              # Max individual claim

# --- Core / Divide-the-Dollar ---
CORE_POT = 10                       # Amount to divide
CORE_MAJORITY_THRESHOLD = 2         # Votes needed for majority

# --- Weighted Voting ---
WV_QUOTA = 6                        # Votes needed to pass
WV_PLAYER_WEIGHT = 3                # First player weight
WV_OPPONENT_WEIGHT = 4              # Second player weight
WV_PASS_BENEFIT = 5                 # Benefit if proposal passes
WV_FAIL_PAYOFF = 0                  # Payoff if proposal fails
WV_OPPOSITION_BONUS = 2             # Bonus for blocking

# --- Stable Matching ---
SM_NUM_OPTIONS = 3                   # Number of partners to rank
SM_TOP_MATCH_PAYOFF = 5             # Payoff for top choice match
SM_MID_MATCH_PAYOFF = 3             # Payoff for middle choice
SM_LOW_MATCH_PAYOFF = 1             # Payoff for last choice

# --- Median Voter ---
MV_POSITION_RANGE = 10              # Policy positions from zero to this
MV_DISTANCE_COST = 1                # Payoff loss per unit distance

# --- Approval Voting ---
AV_NUM_CANDIDATES = 4               # Number of candidates
AV_PREFERRED_WIN = 5                # Payoff if preferred wins
AV_ACCEPTABLE_WIN = 2               # Payoff if acceptable wins
AV_DISLIKED_WIN = -2                # Payoff if disliked wins
