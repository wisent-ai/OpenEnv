# Cheap Talk PD -- standard PD payoffs (messages are non-binding)
CTPD_REWARD = 3
CTPD_TEMPTATION = 5
CTPD_PUNISHMENT = 1
CTPD_SUCKER = 0

# Binding Commitment -- cost of making a binding promise
COMMIT_COST = 1

# Correlated Equilibrium (traffic light / mediator)
CE_FOLLOW_FOLLOW = 4
CE_FOLLOW_DEVIATE = 2
CE_DEVIATE_FOLLOW = 5
CE_DEVIATE_DEVIATE = 1

# Focal Point (multi-option coordination without communication)
FP_MATCH_PAYOFF = 5
FP_MISMATCH_PAYOFF = 0

# Mediated Game (accept/reject third-party mediation)
MG_ACCEPT_ACCEPT = 4
MG_ACCEPT_REJECT = 2
MG_REJECT_ACCEPT = 5
MG_REJECT_REJECT = 0

# Free-form natural-language cheap talk. The agent emits a message string
# AND a base-game action; the message is rendered verbatim into the
# opponent's next-round prompt. Payoff still depends only on the action.
# Distinct from VARIANT_CHEAP_TALK (which restricts the message to one
# bit by Cartesian-producting it into the action vocabulary).
VARIANT_FREE_CHAT = "free_chat"
FREE_CHAT_MSG_PREFIX = "MESSAGE:"
FREE_CHAT_ACT_PREFIX = "ACTION:"
FREE_CHAT_MAX_MSG_CHARS = 400
