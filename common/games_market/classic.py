"""Classic dilemma and extended strategic games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.var.classic_constants import (
    TD_MIN_CLAIM, TD_MAX_CLAIM, TD_BONUS,
    DOLLAR_PRIZE, DOLLAR_MAX_BID,
    UD_CHEAP_COST, UD_EXPENSIVE_COST, UD_CHEAP_VALUE, UD_EXPENSIVE_VALUE,
    MINO_WIN_PAYOFF, MINO_TIE_PAYOFF,
    RPSLS_WIN_PAYOFF, RPSLS_LOSE_PAYOFF, RPSLS_DRAW_PAYOFF,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE
_ZERO_F = float()


# -- Traveler's Dilemma --
def _travelers_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Lower claim gets bonus; higher claim gets penalty."""
    claim_p = int(pa.rsplit("_", _ONE)[_ONE])
    claim_o = int(oa.rsplit("_", _ONE)[_ONE])
    if claim_p == claim_o:
        return (float(claim_p), float(claim_o))
    if claim_p < claim_o:
        return (float(claim_p + TD_BONUS), float(claim_p - TD_BONUS))
    return (float(claim_o - TD_BONUS), float(claim_o + TD_BONUS))


_TD_ACTS = [f"claim_{i}" for i in range(TD_MIN_CLAIM, TD_MAX_CLAIM + _ONE)]


# -- Dollar Auction (escalation: both pay, highest wins) --
def _dollar_auction_payoff(pa: str, oa: str) -> tuple[float, float]:
    bid_p = int(pa.rsplit("_", _ONE)[_ONE])
    bid_o = int(oa.rsplit("_", _ONE)[_ONE])
    if bid_p > bid_o:
        return (float(DOLLAR_PRIZE - bid_p), float(-bid_o))
    if bid_o > bid_p:
        return (float(-bid_p), float(DOLLAR_PRIZE - bid_o))
    half = float(DOLLAR_PRIZE) / _TWO
    return (half - float(bid_p), half - float(bid_o))


_DA_ACTS = [f"bid_{i}" for i in range(DOLLAR_MAX_BID + _ONE)]


# -- Unscrupulous Diner's Dilemma (shared bill) --
def _diner_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Each orders cheap or expensive; bill is split equally."""
    costs = {"order_cheap": UD_CHEAP_COST, "order_expensive": UD_EXPENSIVE_COST}
    values = {"order_cheap": UD_CHEAP_VALUE, "order_expensive": UD_EXPENSIVE_VALUE}
    total_bill = float(costs[pa] + costs[oa])
    each_pays = total_bill / _TWO
    p_val = float(values[pa]) - each_pays
    o_val = float(values[oa]) - each_pays
    return (p_val, o_val)


# -- Minority Game (anti-coordination: minority side wins) --
_MINO_ACTS = ["choose_a", "choose_b", "choose_c"]


def _minority_payoff(pa: str, oa: str) -> tuple[float, float]:
    """With two players: matching = both lose; differing = both win."""
    if pa == oa:
        return (float(MINO_TIE_PAYOFF), float(MINO_TIE_PAYOFF))
    return (float(MINO_WIN_PAYOFF), float(MINO_WIN_PAYOFF))


# -- Rock-Paper-Scissors-Lizard-Spock --
_RPSLS_W = float(RPSLS_WIN_PAYOFF)
_RPSLS_L = float(RPSLS_LOSE_PAYOFF)
_RPSLS_D = float(RPSLS_DRAW_PAYOFF)

_RPSLS_BEATS = {
    "rock": ["scissors", "lizard"],
    "paper": ["rock", "spock"],
    "scissors": ["paper", "lizard"],
    "lizard": ["paper", "spock"],
    "spock": ["rock", "scissors"],
}


def _rpsls_payoff(pa: str, oa: str) -> tuple[float, float]:
    if pa == oa:
        return (_RPSLS_D, _RPSLS_D)
    if oa in _RPSLS_BEATS[pa]:
        return (_RPSLS_W, _RPSLS_L)
    return (_RPSLS_L, _RPSLS_W)


# -- Register --
CLASSIC_GAMES: dict[str, GameConfig] = {
    "travelers_dilemma": GameConfig(
        name="Traveler's Dilemma",
        description=(
            "Two travelers submit claims. The lower claim sets the base "
            "payout with a bonus for the lower claimant and a penalty for "
            "the higher. Nash equilibrium is the minimum claim but "
            "experimental subjects often claim high. Tests the rationality "
            "paradox in iterative dominance reasoning."
        ),
        actions=_TD_ACTS,
        game_type="travelers_dilemma",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_travelers_payoff,
    ),
    "dollar_auction": GameConfig(
        name="Dollar Auction",
        description=(
            "An escalation game: both players bid and both pay their bids "
            "but only the highest bidder wins the prize. Ties split the "
            "prize. Models sunk cost escalation and commitment traps. "
            "Tests resistance to escalation bias."
        ),
        actions=_DA_ACTS,
        game_type="dollar_auction",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_dollar_auction_payoff,
    ),
    "unscrupulous_diner": GameConfig(
        name="Unscrupulous Diner's Dilemma",
        description=(
            "Diners at a restaurant independently order cheap or expensive "
            "meals and split the bill equally. Each prefers expensive food "
            "but shared costs create a free-rider problem. A multiplayer "
            "generalization of the Prisoner's Dilemma in social settings."
        ),
        actions=["order_cheap", "order_expensive"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_diner_payoff,
    ),
    "minority_game": GameConfig(
        name="Minority Game",
        description=(
            "Players independently choose from three options. With two "
            "players, matching choices yield a low tie payoff while "
            "different choices yield a high payoff for both. Tests "
            "anti-coordination and contrarian strategic reasoning."
        ),
        actions=_MINO_ACTS,
        game_type="minority",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_minority_payoff,
    ),
    "rpsls": GameConfig(
        name="Rock-Paper-Scissors-Lizard-Spock",
        description=(
            "An extended zero-sum game with five actions. Each action "
            "beats two others and loses to two others. The unique Nash "
            "equilibrium is uniform randomization. Tests strategic "
            "reasoning in larger zero-sum action spaces."
        ),
        actions=["rock", "paper", "scissors", "lizard", "spock"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_rpsls_payoff,
    ),
}

GAMES.update(CLASSIC_GAMES)
