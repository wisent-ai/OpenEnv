"""String constants for game-theoretic property tags."""

# ── Communication ──
NO_COMMUNICATION = "no_communication"
CHEAP_TALK = "cheap_talk"
COSTLY_SIGNALING = "costly_signaling"
BINDING_COMMITMENT = "binding_commitment"
MEDIATED = "mediated"

# ── Information ──
COMPLETE_INFORMATION = "complete_information"
INCOMPLETE_INFORMATION = "incomplete_information"
ASYMMETRIC_INFORMATION = "asymmetric_information"

# ── Structure ──
SIMULTANEOUS = "simultaneous"
SEQUENTIAL = "sequential"
REPEATED = "repeated"
SINGLE_SHOT = "single_shot"

# ── Payoff type ──
ZERO_SUM = "zero_sum"
SYMMETRIC_PAYOFF = "symmetric_payoff"
ASYMMETRIC_PAYOFF = "asymmetric_payoff"
COORDINATION = "coordination"
ANTI_COORDINATION = "anti_coordination"

# ── Domain ──
SOCIAL_DILEMMA = "social_dilemma"
AUCTION = "auction"
BARGAINING = "bargaining"
VOTING = "voting"
MARKET_COMPETITION = "market_competition"
EVOLUTIONARY = "evolutionary"
SECURITY = "security"
NETWORK = "network"

# ── Action space ──
BINARY_CHOICE = "binary_choice"
SMALL_CHOICE = "small_choice"
LARGE_CHOICE = "large_choice"

# ── Grouped by dimension (for programmatic enumeration) ──
CATEGORIES: dict[str, list[str]] = {
    "communication": [
        NO_COMMUNICATION, CHEAP_TALK, COSTLY_SIGNALING,
        BINDING_COMMITMENT, MEDIATED,
    ],
    "information": [
        COMPLETE_INFORMATION, INCOMPLETE_INFORMATION, ASYMMETRIC_INFORMATION,
    ],
    "structure": [
        SIMULTANEOUS, SEQUENTIAL, REPEATED, SINGLE_SHOT,
    ],
    "payoff_type": [
        ZERO_SUM, SYMMETRIC_PAYOFF, ASYMMETRIC_PAYOFF,
        COORDINATION, ANTI_COORDINATION,
    ],
    "domain": [
        SOCIAL_DILEMMA, AUCTION, BARGAINING, VOTING,
        MARKET_COMPETITION, EVOLUTIONARY, SECURITY, NETWORK,
    ],
    "action_space": [
        BINARY_CHOICE, SMALL_CHOICE, LARGE_CHOICE,
    ],
}
