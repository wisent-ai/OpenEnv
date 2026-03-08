"""Built-in N-player game definitions."""

from __future__ import annotations

from common.games import GameConfig
from common.games_meta.nplayer_config import NPLAYER_GAMES
from constant_definitions.nplayer.nplayer_constants import (
    NPLAYER_DEFAULT_ROUNDS,
    NPG_ENDOWMENT,
    NPG_MULTIPLIER_NUMERATOR,
    NPG_MULTIPLIER_DENOMINATOR,
    NVD_BENEFIT,
    NVD_COST,
    NVD_NO_VOLUNTEER,
    NEF_CAPACITY_FRACTION_NUMERATOR,
    NEF_CAPACITY_FRACTION_DENOMINATOR,
    NEF_ATTEND_REWARD,
    NEF_CROWD_PENALTY,
    NEF_STAY_HOME,
)

_ONE = int(bool(True))
_ZERO = int()


# ---------------------------------------------------------------------------
# Public Goods Game (N-player)
# ---------------------------------------------------------------------------

def _public_goods_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    """Each player contributes from an endowment. The pot is multiplied and split."""
    n = len(actions)
    contributions = []
    for a in actions:
        contributions.append(int(a.rsplit("_", _ONE)[_ONE]))
    total = sum(contributions)
    pool = total * NPG_MULTIPLIER_NUMERATOR / NPG_MULTIPLIER_DENOMINATOR
    share = pool / n
    payoffs = tuple(
        float(NPG_ENDOWMENT - c + share) for c in contributions
    )
    return payoffs


_PG_ACTIONS = [f"contribute_{i}" for i in range(NPG_ENDOWMENT + _ONE)]


# ---------------------------------------------------------------------------
# Volunteer's Dilemma (N-player)
# ---------------------------------------------------------------------------

def _volunteer_dilemma_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    """If at least one player volunteers, everyone benefits but volunteers pay a cost."""
    any_volunteer = any(a == "volunteer" for a in actions)
    payoffs: list[float] = []
    for a in actions:
        if not any_volunteer:
            payoffs.append(float(NVD_NO_VOLUNTEER))
        elif a == "volunteer":
            payoffs.append(float(NVD_BENEFIT - NVD_COST))
        else:
            payoffs.append(float(NVD_BENEFIT))
    return tuple(payoffs)


# ---------------------------------------------------------------------------
# El Farol Bar Problem (N-player)
# ---------------------------------------------------------------------------

def _el_farol_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    """Attend a bar that is fun only when not overcrowded."""
    n = len(actions)
    capacity = n * NEF_CAPACITY_FRACTION_NUMERATOR // NEF_CAPACITY_FRACTION_DENOMINATOR
    attendees = sum(_ONE for a in actions if a == "attend")
    crowded = attendees > capacity
    payoffs: list[float] = []
    for a in actions:
        if a == "stay_home":
            payoffs.append(float(NEF_STAY_HOME))
        elif crowded:
            payoffs.append(float(NEF_CROWD_PENALTY))
        else:
            payoffs.append(float(NEF_ATTEND_REWARD))
    return tuple(payoffs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_THREE = _ONE + _ONE + _ONE
_FIVE = _THREE + _ONE + _ONE

_BUILTIN_NPLAYER_GAMES: dict[str, GameConfig] = {
    "nplayer_public_goods": GameConfig(
        name="N-Player Public Goods",
        description=(
            "Each player contributes from an endowment. The total pot is "
            "multiplied and split equally among all players."
        ),
        actions=_PG_ACTIONS,
        game_type="public_goods",
        num_players=_FIVE,
        default_rounds=NPLAYER_DEFAULT_ROUNDS,
        payoff_fn=_public_goods_payoff,
    ),
    "nplayer_volunteer_dilemma": GameConfig(
        name="N-Player Volunteer's Dilemma",
        description=(
            "Players choose to volunteer or abstain. If at least one "
            "volunteers, everyone benefits but volunteers pay a cost. "
            "If nobody volunteers, everyone gets nothing."
        ),
        actions=["volunteer", "abstain"],
        game_type="matrix",
        num_players=_FIVE,
        default_rounds=NPLAYER_DEFAULT_ROUNDS,
        payoff_fn=_volunteer_dilemma_payoff,
    ),
    "nplayer_el_farol": GameConfig(
        name="N-Player El Farol Bar",
        description=(
            "Players decide whether to attend a bar. The bar is fun when "
            "not crowded but unpleasant when too many people show up."
        ),
        actions=["attend", "stay_home"],
        game_type="matrix",
        num_players=_FIVE,
        default_rounds=NPLAYER_DEFAULT_ROUNDS,
        payoff_fn=_el_farol_payoff,
    ),
}

NPLAYER_GAMES.update(_BUILTIN_NPLAYER_GAMES)
