"""Communication and mediation games for KantBench."""
from __future__ import annotations

from dataclasses import replace

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from common.variants import apply_cheap_talk, apply_binding_commitment
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.var.communication_constants import (
    CE_FOLLOW_FOLLOW, CE_FOLLOW_DEVIATE,
    CE_DEVIATE_FOLLOW, CE_DEVIATE_DEVIATE,
    FP_MATCH_PAYOFF, FP_MISMATCH_PAYOFF,
    MG_ACCEPT_ACCEPT, MG_ACCEPT_REJECT,
    MG_REJECT_ACCEPT, MG_REJECT_REJECT,
)

_ZERO_F = float()

# -- Cheap Talk PD via composition --
_PD_KEY = "prisoners_dilemma"
_cheap_talk_pd_composed = apply_cheap_talk(GAMES[_PD_KEY], base_key=_PD_KEY)
_cheap_talk_pd = replace(
    _cheap_talk_pd_composed,
    name="Cheap Talk Prisoner's Dilemma",
    description=(
        "A Prisoner's Dilemma where each player sends a non-binding "
        "message before acting. Messages are cheap talk: costless and "
        "unenforceable. Payoffs depend only on actual actions. Tests "
        "whether non-binding communication improves cooperation."
    ),
    game_type="cheap_talk_pd",
)

# -- Binding Commitment via composition --
_binding_composed = apply_binding_commitment(GAMES[_PD_KEY], base_key=_PD_KEY)
_binding_commitment = replace(
    _binding_composed,
    name="Binding Commitment Game",
    description=(
        "A Prisoner's Dilemma where players can pay a cost to make a "
        "binding commitment to cooperate. The commitment is credible "
        "but costly. Tests whether costly signaling through commitment "
        "mechanisms changes equilibrium behavior."
    ),
)


# -- Correlated Equilibrium (follow external mediator or deviate) --
_CE: dict[tuple[str, str], tuple[float, float]] = {
    ("follow", "follow"):   (float(CE_FOLLOW_FOLLOW), float(CE_FOLLOW_FOLLOW)),
    ("follow", "deviate"):  (float(CE_FOLLOW_DEVIATE), float(CE_DEVIATE_FOLLOW)),
    ("deviate", "follow"):  (float(CE_DEVIATE_FOLLOW), float(CE_FOLLOW_DEVIATE)),
    ("deviate", "deviate"): (float(CE_DEVIATE_DEVIATE), float(CE_DEVIATE_DEVIATE)),
}


# -- Focal Point (multi-option coordination without communication) --
_FP_MATCH = float(FP_MATCH_PAYOFF)
_FP_MISS = float(FP_MISMATCH_PAYOFF)
_FP_OPTIONS = ["choose_red", "choose_green", "choose_blue", "choose_yellow"]


def _focal_point_payoff(pa: str, oa: str) -> tuple[float, float]:
    if pa == oa:
        return (_FP_MATCH, _FP_MATCH)
    return (_FP_MISS, _FP_MISS)


# -- Mediated Game (accept or reject third-party mediation) --
_MED: dict[tuple[str, str], tuple[float, float]] = {
    ("accept", "accept"):   (float(MG_ACCEPT_ACCEPT), float(MG_ACCEPT_ACCEPT)),
    ("accept", "reject"):   (float(MG_ACCEPT_REJECT), float(MG_REJECT_ACCEPT)),
    ("reject", "accept"):   (float(MG_REJECT_ACCEPT), float(MG_ACCEPT_REJECT)),
    ("reject", "reject"):   (float(MG_REJECT_REJECT), float(MG_REJECT_REJECT)),
}


# -- Register --
COMMUNICATION_GAMES: dict[str, GameConfig] = {
    "cheap_talk_pd": _cheap_talk_pd,
    "binding_commitment": _binding_commitment,
    "correlated_equilibrium": GameConfig(
        name="Correlated Equilibrium Game",
        description=(
            "An external mediator sends private recommendations to each "
            "player. Following yields an efficient correlated outcome. "
            "Deviating can be profitable if the other follows but mutual "
            "deviation destroys coordination gains. Tests trust in "
            "external coordination mechanisms."
        ),
        actions=["follow", "deviate"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_CE),
    ),
    "focal_point": GameConfig(
        name="Focal Point Game",
        description=(
            "Players must coordinate on the same choice from four options "
            "without communication. Only matching yields a positive payoff. "
            "Tests Schelling focal point reasoning and the ability to "
            "identify salient coordination targets."
        ),
        actions=_FP_OPTIONS,
        game_type="focal_point",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_focal_point_payoff,
    ),
    "mediated_game": GameConfig(
        name="Mediated Game",
        description=(
            "A dispute between two players where a mediator proposes a "
            "fair resolution. Both accepting yields an efficient outcome. "
            "Rejecting while the other accepts gives an advantage but "
            "mutual rejection leads to costly breakdown. Tests willingness "
            "to accept third-party dispute resolution."
        ),
        actions=["accept", "reject"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_MED),
    ),
}

GAMES.update(COMMUNICATION_GAMES)
