"""Signaling and incomplete information games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.ext.signaling_constants import (
    BQ_TOUGH_BEER_PAYOFF, BQ_TOUGH_QUICHE_PAYOFF,
    BQ_WEAK_BEER_PAYOFF, BQ_WEAK_QUICHE_PAYOFF,
    BQ_CHALLENGE_COST, BQ_NO_CHALLENGE_BONUS,
    SPENCE_HIGH_WAGE, SPENCE_LOW_WAGE,
    SPENCE_EDU_COST_HIGH, SPENCE_EDU_COST_LOW,
    CT_ALIGNED_MATCH, CT_ALIGNED_MISMATCH, CT_BIAS,
    LEMON_GOOD_QUALITY_VALUE, LEMON_BAD_QUALITY_VALUE,
    LEMON_GOOD_SELLER_COST, LEMON_BAD_SELLER_COST, LEMON_MAX_PRICE,
    BP_GOOD_STATE_VALUE, BP_BAD_STATE_PENALTY, BP_SAFE_PAYOFF,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE


# -- Beer-Quiche (simplified as simultaneous signal-response) --
_BQ_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("beer", "challenge"):    (float(BQ_TOUGH_BEER_PAYOFF + BQ_CHALLENGE_COST), float(_TWO)),
    ("beer", "back_down"):    (float(BQ_TOUGH_BEER_PAYOFF + BQ_NO_CHALLENGE_BONUS), float(int())),
    ("quiche", "challenge"):  (float(BQ_WEAK_QUICHE_PAYOFF + BQ_CHALLENGE_COST), float(-_ONE)),
    ("quiche", "back_down"):  (float(BQ_WEAK_QUICHE_PAYOFF + BQ_NO_CHALLENGE_BONUS), float(int())),
}


# -- Spence Signaling (worker picks edu level, firm responds) --
def _spence_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Worker chooses education; firm offers wage based on signal."""
    educated = player_action == "educate"
    high_wage = opponent_action == "high_wage"
    wage = SPENCE_HIGH_WAGE if high_wage else SPENCE_LOW_WAGE
    cost = SPENCE_EDU_COST_HIGH if educated else int()
    worker_pay = float(wage - cost)
    firm_pay = float(SPENCE_HIGH_WAGE - wage) if educated else float(SPENCE_LOW_WAGE - wage)
    return (worker_pay, firm_pay)


# -- Cheap Talk --
_CT_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("signal_left", "act_left"):   (float(CT_ALIGNED_MATCH), float(CT_ALIGNED_MATCH)),
    ("signal_left", "act_right"):  (float(CT_ALIGNED_MISMATCH), float(CT_ALIGNED_MISMATCH)),
    ("signal_right", "act_left"):  (float(CT_ALIGNED_MISMATCH + CT_BIAS), float(CT_ALIGNED_MISMATCH)),
    ("signal_right", "act_right"): (float(CT_ALIGNED_MATCH + CT_BIAS), float(CT_ALIGNED_MATCH)),
}


# -- Lemon Market --
def _lemon_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Seller sets price; buyer decides to buy or pass."""
    price = int(player_action.rsplit("_", _ONE)[_ONE])
    if opponent_action == "pass":
        return (float(int()), float(int()))
    avg_value = (LEMON_GOOD_QUALITY_VALUE + LEMON_BAD_QUALITY_VALUE) // _TWO
    buyer_pay = float(avg_value - price)
    avg_cost = (LEMON_GOOD_SELLER_COST + LEMON_BAD_SELLER_COST) // _TWO
    seller_pay = float(price - avg_cost)
    return (seller_pay, buyer_pay)


_LEMON_ACTIONS = [f"price_{i}" for i in range(LEMON_MAX_PRICE + _ONE)]


# -- Bayesian Persuasion --
_BP_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("reveal", "act"):    (float(BP_GOOD_STATE_VALUE), float(BP_GOOD_STATE_VALUE)),
    ("reveal", "safe"):   (float(BP_SAFE_PAYOFF), float(BP_SAFE_PAYOFF)),
    ("conceal", "act"):   (float(BP_BAD_STATE_PENALTY), float(BP_BAD_STATE_PENALTY)),
    ("conceal", "safe"):  (float(BP_SAFE_PAYOFF), float(BP_SAFE_PAYOFF)),
}


# -- Register --
SIGNALING_GAMES: dict[str, GameConfig] = {
    "beer_quiche": GameConfig(
        name="Beer-Quiche Game",
        description=(
            "A signaling game: the sender chooses a meal (beer or quiche) "
            "to signal their type; the receiver decides whether to challenge. "
            "Tests reasoning about sequential equilibrium and belief refinement."
        ),
        actions=["beer", "quiche"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_BQ_MATRIX),
    ),
    "spence_signaling": GameConfig(
        name="Spence Job Market Signaling",
        description=(
            "A worker chooses whether to acquire education as a signal of "
            "ability; a firm responds with a wage offer. Tests understanding "
            "of separating versus pooling equilibria in labor markets."
        ),
        actions=["educate", "no_educate"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_spence_payoff,
    ),
    "cheap_talk": GameConfig(
        name="Cheap Talk",
        description=(
            "A sender observes a state and sends a costless message; "
            "the receiver chooses an action. Interests are partially "
            "aligned. Tests strategic communication and credibility."
        ),
        actions=["signal_left", "signal_right"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_CT_MATRIX),
    ),
    "lemon_market": GameConfig(
        name="Lemon Market",
        description=(
            "A seller with private quality information sets a price; "
            "the buyer decides whether to purchase. Adverse selection "
            "can cause market unraveling where only low-quality goods trade."
        ),
        actions=_LEMON_ACTIONS,
        game_type="lemon",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_lemon_payoff,
    ),
    "bayesian_persuasion": GameConfig(
        name="Bayesian Persuasion",
        description=(
            "A sender designs an information structure (reveal or conceal "
            "the state); a receiver takes an action based on the signal. "
            "Tests strategic information disclosure and commitment to "
            "information policies."
        ),
        actions=["reveal", "conceal"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_BP_MATRIX),
    ),
}

GAMES.update(SIGNALING_GAMES)
