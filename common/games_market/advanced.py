"""Advanced market mechanism games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.batch4.advanced_constants import (
    PRE_EARLY_EARLY, PRE_EARLY_LATE, PRE_LATE_EARLY, PRE_LATE_LATE,
    PRE_OUT_PAYOFF,
    WOG_LARGE_LARGE, WOG_LARGE_SMALL, WOG_LARGE_NONE,
    WOG_SMALL_SMALL, WOG_SMALL_NONE, WOG_NO_GIFT,
    PS_SAVE_PAYOFF, PS_SCORE_PAYOFF, PS_CENTER_BONUS,
)

_ZERO_F = float()
_OUT_F = float(PRE_OUT_PAYOFF)


# -- Preemption Game (enter_early / enter_late / stay_out) --
_PRE: dict[tuple[str, str], tuple[float, float]] = {
    ("enter_early", "enter_early"): (
        float(PRE_EARLY_EARLY), float(PRE_EARLY_EARLY),
    ),
    ("enter_early", "enter_late"): (
        float(PRE_EARLY_LATE), float(PRE_LATE_EARLY),
    ),
    ("enter_early", "stay_out"): (float(PRE_EARLY_LATE), _OUT_F),
    ("enter_late", "enter_early"): (
        float(PRE_LATE_EARLY), float(PRE_EARLY_LATE),
    ),
    ("enter_late", "enter_late"): (
        float(PRE_LATE_LATE), float(PRE_LATE_LATE),
    ),
    ("enter_late", "stay_out"): (float(PRE_LATE_LATE), _OUT_F),
    ("stay_out", "enter_early"): (_OUT_F, float(PRE_EARLY_LATE)),
    ("stay_out", "enter_late"):  (_OUT_F, float(PRE_LATE_LATE)),
    ("stay_out", "stay_out"):    (_OUT_F, _OUT_F),
}


# -- War of Gifts (gift_large / gift_small / no_gift) --
_WOG_LL = float(WOG_LARGE_LARGE)
_WOG_LS = float(WOG_LARGE_SMALL)
_WOG_LN = float(WOG_LARGE_NONE)
_WOG_SS = float(WOG_SMALL_SMALL)
_WOG_SN = float(WOG_SMALL_NONE)
_WOG_NG = float(WOG_NO_GIFT)
_WOG_SL = _ZERO_F  # small loses to large

_WOG: dict[tuple[str, str], tuple[float, float]] = {
    ("gift_large", "gift_large"): (_WOG_LL, _WOG_LL),
    ("gift_large", "gift_small"): (_WOG_LS, _WOG_SL),
    ("gift_large", "no_gift"):    (_WOG_LN, _WOG_NG),
    ("gift_small", "gift_large"): (_WOG_SL, _WOG_LS),
    ("gift_small", "gift_small"): (_WOG_SS, _WOG_SS),
    ("gift_small", "no_gift"):    (_WOG_SN, _WOG_NG),
    ("no_gift", "gift_large"):    (_WOG_NG, _WOG_LN),
    ("no_gift", "gift_small"):    (_WOG_NG, _WOG_SN),
    ("no_gift", "no_gift"):       (_WOG_NG, _WOG_NG),
}


# -- Penalty Shootout (left / center / right, kicker vs keeper) --
_PS_SAVE = float(PS_SAVE_PAYOFF)
_PS_SCORE = float(PS_SCORE_PAYOFF)
_PS_CENTER = float(PS_CENTER_BONUS)


def _penalty_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Kicker (player) vs keeper (opponent). Match means save."""
    if pa == oa:
        return (_PS_SAVE, -_PS_SAVE)
    if pa == "center":
        score = _PS_SCORE + _PS_CENTER
    else:
        score = _PS_SCORE
    return (score, -score)


# -- Register --
ADVANCED_GAMES: dict[str, GameConfig] = {
    "preemption_game": GameConfig(
        name="Preemption Game",
        description=(
            "A timing game with first-mover advantage. Players choose to "
            "enter a market early (risky if both enter) or late (safer but "
            "second-mover disadvantage) or stay out entirely for a safe "
            "payoff. Early entry against a late opponent captures the market. "
            "Tests preemption incentives and entry deterrence."
        ),
        actions=["enter_early", "enter_late", "stay_out"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_PRE),
    ),
    "war_of_gifts": GameConfig(
        name="War of Gifts",
        description=(
            "A competitive generosity game. Players choose to give a large "
            "gift or small gift or no gift. The largest giver wins prestige "
            "but at material cost. Mutual large gifts cancel prestige gains. "
            "No gift is safe but earns no prestige. Tests competitive "
            "signaling through costly generosity."
        ),
        actions=["gift_large", "gift_small", "no_gift"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_WOG),
    ),
    "penalty_shootout": GameConfig(
        name="Penalty Shootout",
        description=(
            "A zero-sum mismatch game modeling penalty kicks. The kicker "
            "chooses left or center or right; the goalkeeper dives. Matching "
            "means a save. Mismatching means a goal. Center kicks score a "
            "bonus when the goalkeeper guesses wrong. Tests mixed-strategy "
            "reasoning in adversarial settings."
        ),
        actions=["left", "center", "right"],
        game_type="penalty_shootout",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_penalty_payoff,
    ),
}

GAMES.update(ADVANCED_GAMES)
