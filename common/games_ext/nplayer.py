"""N-player social dilemma games for KantBench.

Modeled as one agent vs one opponent (representing aggregate of others).
"""
from __future__ import annotations

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.auction_nplayer_constants import (
    COMMONS_RESOURCE_CAPACITY, COMMONS_MAX_EXTRACTION,
    COMMONS_DEPLETION_PENALTY,
    VOLUNTEER_BENEFIT, VOLUNTEER_COST, VOLUNTEER_NO_VOL,
    EL_FAROL_ATTEND_REWARD, EL_FAROL_CROWD_PENALTY, EL_FAROL_STAY_HOME,
    EL_FAROL_CAPACITY,
)

_ONE = int(bool(True))
_ZERO_F = float()


# -- Tragedy of the Commons --

def _commons_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Resource extraction game.

    Each player extracts from a shared resource. If total extraction
    exceeds capacity, both suffer a depletion penalty.
    """
    p_extract = int(player_action.rsplit("_", _ONE)[_ONE])
    o_extract = int(opponent_action.rsplit("_", _ONE)[_ONE])
    total = p_extract + o_extract

    if total > COMMONS_RESOURCE_CAPACITY:
        return (float(COMMONS_DEPLETION_PENALTY), float(COMMONS_DEPLETION_PENALTY))

    return (float(p_extract), float(o_extract))


_COMMONS_ACTIONS = [
    f"extract_{i}" for i in range(COMMONS_MAX_EXTRACTION + _ONE)
]


# -- Volunteer's Dilemma --

def _volunteer_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """At least one must volunteer for everyone to benefit.

    Volunteering costs the volunteer but benefits all.
    If nobody volunteers, everyone gets nothing.
    """
    p_vol = player_action == "volunteer"
    o_vol = opponent_action == "volunteer"

    if not p_vol and not o_vol:
        return (float(VOLUNTEER_NO_VOL), float(VOLUNTEER_NO_VOL))

    p_pay = float(VOLUNTEER_BENEFIT - VOLUNTEER_COST) if p_vol else float(VOLUNTEER_BENEFIT)
    o_pay = float(VOLUNTEER_BENEFIT - VOLUNTEER_COST) if o_vol else float(VOLUNTEER_BENEFIT)
    return (p_pay, o_pay)


# -- El Farol Bar Problem --

def _el_farol_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Bar attendance decision game.

    Going to the bar is fun if few attend (under capacity), but
    unpleasant if crowded. Staying home gives a moderate fixed payoff.
    """
    p_goes = player_action == "attend"
    o_goes = opponent_action == "attend"

    attendees = int(p_goes) + int(o_goes)
    crowded = attendees > _ONE

    if not p_goes:
        p_pay = float(EL_FAROL_STAY_HOME)
    elif crowded:
        p_pay = float(EL_FAROL_CROWD_PENALTY)
    else:
        p_pay = float(EL_FAROL_ATTEND_REWARD)

    if not o_goes:
        o_pay = float(EL_FAROL_STAY_HOME)
    elif crowded:
        o_pay = float(EL_FAROL_CROWD_PENALTY)
    else:
        o_pay = float(EL_FAROL_ATTEND_REWARD)

    return (p_pay, o_pay)


# -- Register --

NPLAYER_GAMES: dict[str, GameConfig] = {
    "tragedy_of_commons": GameConfig(
        name="Tragedy of the Commons",
        description=(
            "Players extract resources from a shared pool. Individual "
            "incentive is to extract more, but if total extraction exceeds "
            "the sustainable capacity, the resource collapses and everyone "
            "suffers. Models environmental and resource management dilemmas."
        ),
        actions=_COMMONS_ACTIONS,
        game_type="commons",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_commons_payoff,
    ),
    "volunteer_dilemma": GameConfig(
        name="Volunteer's Dilemma",
        description=(
            "At least one player must volunteer (at personal cost) for "
            "everyone to receive a benefit. If nobody volunteers, all get "
            "nothing. Models bystander effects and public good provision."
        ),
        actions=["volunteer", "abstain"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_volunteer_payoff,
    ),
    "el_farol": GameConfig(
        name="El Farol Bar Problem",
        description=(
            "Each player decides whether to attend a bar. If attendance "
            "is below capacity, going is better than staying home. If the "
            "bar is crowded, staying home is better. Models minority games "
            "and congestion dynamics."
        ),
        actions=["attend", "stay_home"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_el_farol_payoff,
    ),
}

GAMES.update(NPLAYER_GAMES)
