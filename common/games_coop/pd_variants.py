"""Prisoner's Dilemma variants for KantBench."""
from __future__ import annotations

from dataclasses import replace

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from common.variants import apply_exit
from constant_definitions.game_constants import (
    DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS,
)
from constant_definitions.var.pd_variant_constants import (
    APD_A_TEMPTATION, APD_A_REWARD, APD_A_PUNISHMENT, APD_A_SUCKER,
    APD_B_TEMPTATION, APD_B_REWARD, APD_B_PUNISHMENT, APD_B_SUCKER,
    DONATION_BENEFIT, DONATION_COST,
    FOF_SHARE_PAYOFF, FOF_STEAL_WIN_PAYOFF,
    PW_DISARM_DISARM, PW_DISARM_ARM, PW_ARM_DISARM, PW_ARM_ARM,
)

_ZERO_F = float()


# -- Asymmetric PD (alibi game: different payoffs per player) --
_ASYM_PD: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(APD_A_REWARD), float(APD_B_REWARD)),
    ("cooperate", "defect"):    (float(APD_A_SUCKER), float(APD_B_TEMPTATION)),
    ("defect", "cooperate"):    (float(APD_A_TEMPTATION), float(APD_B_SUCKER)),
    ("defect", "defect"):       (float(APD_A_PUNISHMENT), float(APD_B_PUNISHMENT)),
}


# -- Donation Game (pay cost c to give benefit b to opponent) --
_DG: dict[tuple[str, str], tuple[float, float]] = {
    ("donate", "donate"): (
        float(DONATION_BENEFIT - DONATION_COST),
        float(DONATION_BENEFIT - DONATION_COST),
    ),
    ("donate", "keep"): (float(-DONATION_COST), float(DONATION_BENEFIT)),
    ("keep", "donate"):  (float(DONATION_BENEFIT), float(-DONATION_COST)),
    ("keep", "keep"):    (_ZERO_F, _ZERO_F),
}


# -- Friend or Foe (game show: both defect yields zero) --
_FOF: dict[tuple[str, str], tuple[float, float]] = {
    ("friend", "friend"): (float(FOF_SHARE_PAYOFF), float(FOF_SHARE_PAYOFF)),
    ("friend", "foe"):    (_ZERO_F, float(FOF_STEAL_WIN_PAYOFF)),
    ("foe", "friend"):    (float(FOF_STEAL_WIN_PAYOFF), _ZERO_F),
    ("foe", "foe"):       (_ZERO_F, _ZERO_F),
}


# -- Peace-War Game (arms race framing from international relations) --
_PW: dict[tuple[str, str], tuple[float, float]] = {
    ("disarm", "disarm"): (float(PW_DISARM_DISARM), float(PW_DISARM_DISARM)),
    ("disarm", "arm"):    (float(PW_DISARM_ARM), float(PW_ARM_DISARM)),
    ("arm", "disarm"):    (float(PW_ARM_DISARM), float(PW_DISARM_ARM)),
    ("arm", "arm"):       (float(PW_ARM_ARM), float(PW_ARM_ARM)),
}


# -- Register --
_PD_KEY = "prisoners_dilemma"
_optional_pd_composed = apply_exit(GAMES[_PD_KEY], base_key=_PD_KEY)
_optional_pd = replace(
    _optional_pd_composed,
    name="Optional Prisoner's Dilemma",
    description=(
        "A Prisoner's Dilemma with a third action: exit. Exiting gives "
        "a safe intermediate payoff regardless of the opponent's choice. "
        "Tests whether outside options change cooperation dynamics and "
        "models situations where players can walk away from interactions."
    ),
)

PD_VARIANT_GAMES: dict[str, GameConfig] = {
    "optional_pd": _optional_pd,
    "asymmetric_pd": GameConfig(
        name="Asymmetric Prisoner's Dilemma",
        description=(
            "A Prisoner's Dilemma where players have unequal payoff "
            "structures. The first player has an alibi advantage with a "
            "higher punishment payoff. Tests strategic reasoning under "
            "asymmetric incentive conditions."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_ASYM_PD),
    ),
    "donation_game": GameConfig(
        name="Donation Game",
        description=(
            "A simplified cooperation model: each player independently "
            "decides whether to donate. Donating costs the donor but "
            "gives a larger benefit to the recipient. The dominant "
            "strategy is to keep, but mutual donation is Pareto superior."
        ),
        actions=["donate", "keep"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_DG),
    ),
    "friend_or_foe": GameConfig(
        name="Friend or Foe",
        description=(
            "A game show variant of the Prisoner's Dilemma. If both choose "
            "friend, winnings are shared. If one steals (foe), they take all. "
            "If both choose foe, neither gets anything. Unlike standard PD, "
            "mutual defection yields zero, creating a weak equilibrium."
        ),
        actions=["friend", "foe"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_FOF),
    ),
    "peace_war": GameConfig(
        name="Peace-War Game",
        description=(
            "An international relations framing of the Prisoner's Dilemma. "
            "Players choose to arm or disarm. Mutual disarmament yields the "
            "best joint outcome but unilateral arming dominates. Models "
            "the security dilemma and arms race escalation dynamics."
        ),
        actions=["disarm", "arm"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_PW),
    ),
}

GAMES.update(PD_VARIANT_GAMES)
