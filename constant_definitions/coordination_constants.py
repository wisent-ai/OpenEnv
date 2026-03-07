# --- Battle of the Sexes payoffs ---
# Player 1 prefers opera, Player 2 prefers football
BOS_PREFERRED_PAYOFF = 3     # Coordinating on your preferred option
BOS_COMPROMISE_PAYOFF = 2    # Coordinating on the other's preferred option
BOS_MISMATCH_PAYOFF = 0      # Failing to coordinate

# --- Pure Coordination payoffs ---
PC_MATCH_PAYOFF = 2          # Both choose same action
PC_MISMATCH_PAYOFF = 0       # Choices differ

# --- Deadlock payoffs (defection dominant for both) ---
# Ordering: DC > DD > CC > CD
DL_DC_PAYOFF = 4    # I defect, they cooperate
DL_DD_PAYOFF = 3    # Both defect (NE)
DL_CC_PAYOFF = 2    # Both cooperate
DL_CD_PAYOFF = 1    # I cooperate, they defect

# --- Harmony payoffs (cooperation dominant for both) ---
# Ordering: CC > DC > CD > DD
HM_CC_PAYOFF = 4    # Both cooperate (NE)
HM_DC_PAYOFF = 3    # I defect, they cooperate
HM_CD_PAYOFF = 2    # I cooperate, they defect
HM_DD_PAYOFF = 1    # Both defect
