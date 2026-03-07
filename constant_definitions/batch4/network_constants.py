# Security Game -- defender vs attacker resource allocation
SG_DEFEND_SUCCESS = 3
SG_ATTACK_FAIL = -1
SG_DEFEND_FAIL = -2
SG_ATTACK_SUCCESS = 4

# Link Formation -- bilateral consent for network links
LF_MUTUAL_CONNECT = 3
LF_UNILATERAL_COST = -1
LF_MUTUAL_ISOLATE = 0

# Trust with Punishment (3x3: cooperate, defect, punish)
TWP_CC = 3
TWP_CD = 0
TWP_DC = 5
TWP_DD = 1
TWP_CP = -1
TWP_PC = 2
TWP_DP = -2
TWP_PD = 0
TWP_PP = -1

# Dueling Game -- timing under uncertainty
DG_EARLY_EARLY = 1
DG_EARLY_LATE = 3
DG_LATE_EARLY = -1
DG_LATE_LATE = 2
