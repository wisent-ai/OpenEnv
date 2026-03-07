"""Extended matrix (normal-form) games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.zero_sum_constants import (
    MP_MATCH_PAYOFF, MP_MISMATCH_PAYOFF,
    RPS_WIN_PAYOFF, RPS_LOSE_PAYOFF, RPS_DRAW_PAYOFF,
)
from constant_definitions.coordination_constants import (
    BOS_PREFERRED_PAYOFF, BOS_COMPROMISE_PAYOFF, BOS_MISMATCH_PAYOFF,
    PC_MATCH_PAYOFF, PC_MISMATCH_PAYOFF,
    DL_DC_PAYOFF, DL_DD_PAYOFF, DL_CC_PAYOFF, DL_CD_PAYOFF,
    HM_CC_PAYOFF, HM_DC_PAYOFF, HM_CD_PAYOFF, HM_DD_PAYOFF,
)

# -- Matching Pennies --
_MP_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("heads", "heads"): (float(MP_MATCH_PAYOFF), float(MP_MISMATCH_PAYOFF)),
    ("heads", "tails"): (float(MP_MISMATCH_PAYOFF), float(MP_MATCH_PAYOFF)),
    ("tails", "heads"): (float(MP_MISMATCH_PAYOFF), float(MP_MATCH_PAYOFF)),
    ("tails", "tails"): (float(MP_MATCH_PAYOFF), float(MP_MISMATCH_PAYOFF)),
}

# -- Rock-Paper-Scissors --
_W, _L, _D = float(RPS_WIN_PAYOFF), float(RPS_LOSE_PAYOFF), float(RPS_DRAW_PAYOFF)
_RPS_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("rock", "rock"):         (_D, _D),
    ("rock", "scissors"):     (_W, _L),
    ("rock", "paper"):        (_L, _W),
    ("scissors", "rock"):     (_L, _W),
    ("scissors", "scissors"): (_D, _D),
    ("scissors", "paper"):    (_W, _L),
    ("paper", "rock"):        (_W, _L),
    ("paper", "scissors"):    (_L, _W),
    ("paper", "paper"):       (_D, _D),
}

# -- Battle of the Sexes --
_BOS_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("opera", "opera"):       (float(BOS_PREFERRED_PAYOFF), float(BOS_COMPROMISE_PAYOFF)),
    ("opera", "football"):    (float(BOS_MISMATCH_PAYOFF), float(BOS_MISMATCH_PAYOFF)),
    ("football", "opera"):    (float(BOS_MISMATCH_PAYOFF), float(BOS_MISMATCH_PAYOFF)),
    ("football", "football"): (float(BOS_COMPROMISE_PAYOFF), float(BOS_PREFERRED_PAYOFF)),
}

# -- Pure Coordination --
_PC_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("left", "left"):   (float(PC_MATCH_PAYOFF), float(PC_MATCH_PAYOFF)),
    ("left", "right"):  (float(PC_MISMATCH_PAYOFF), float(PC_MISMATCH_PAYOFF)),
    ("right", "left"):  (float(PC_MISMATCH_PAYOFF), float(PC_MISMATCH_PAYOFF)),
    ("right", "right"): (float(PC_MATCH_PAYOFF), float(PC_MATCH_PAYOFF)),
}

# -- Deadlock --
_DL_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(DL_CC_PAYOFF), float(DL_CC_PAYOFF)),
    ("cooperate", "defect"):    (float(DL_CD_PAYOFF), float(DL_DC_PAYOFF)),
    ("defect", "cooperate"):    (float(DL_DC_PAYOFF), float(DL_CD_PAYOFF)),
    ("defect", "defect"):       (float(DL_DD_PAYOFF), float(DL_DD_PAYOFF)),
}

# -- Harmony --
_HM_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(HM_CC_PAYOFF), float(HM_CC_PAYOFF)),
    ("cooperate", "defect"):    (float(HM_CD_PAYOFF), float(HM_DC_PAYOFF)),
    ("defect", "cooperate"):    (float(HM_DC_PAYOFF), float(HM_CD_PAYOFF)),
    ("defect", "defect"):       (float(HM_DD_PAYOFF), float(HM_DD_PAYOFF)),
}

# -- Register all games --

EXTENDED_MATRIX_GAMES: dict[str, GameConfig] = {
    "matching_pennies": GameConfig(
        name="Matching Pennies",
        description=(
            "A pure zero-sum game. The matcher wins if both choose the same "
            "side; the mismatcher wins if they differ. The only Nash "
            "equilibrium is a mixed strategy of equal randomization."
        ),
        actions=["heads", "tails"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_MP_MATRIX),
    ),
    "rock_paper_scissors": GameConfig(
        name="Rock-Paper-Scissors",
        description=(
            "A three-action zero-sum game: rock beats scissors, scissors "
            "beats paper, paper beats rock. The unique Nash equilibrium "
            "is uniform randomization over all three actions."
        ),
        actions=["rock", "paper", "scissors"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_RPS_MATRIX),
    ),
    "battle_of_the_sexes": GameConfig(
        name="Battle of the Sexes",
        description=(
            "Two players want to coordinate but have different preferences. "
            "The first player prefers opera, the second prefers football. "
            "Both prefer any coordination over miscoordination. Two pure "
            "Nash equilibria exist at (opera, opera) and (football, football)."
        ),
        actions=["opera", "football"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_BOS_MATRIX),
    ),
    "pure_coordination": GameConfig(
        name="Pure Coordination",
        description=(
            "Two players receive a positive payoff only when they choose "
            "the same action. Both (left, left) and (right, right) are "
            "Nash equilibria. Tests whether agents can converge on a focal "
            "point without communication."
        ),
        actions=["left", "right"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_PC_MATRIX),
    ),
    "deadlock": GameConfig(
        name="Deadlock",
        description=(
            "Similar to the Prisoner's Dilemma but with different payoff "
            "ordering: DC > DD > CC > CD. Both players prefer mutual "
            "defection over mutual cooperation. The unique Nash equilibrium "
            "is (defect, defect) and it is also Pareto optimal."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_DL_MATRIX),
    ),
    "harmony": GameConfig(
        name="Harmony",
        description=(
            "The opposite of a social dilemma: cooperation is the dominant "
            "strategy for both players. Payoff ordering CC > DC > CD > DD "
            "means rational self-interest naturally leads to the socially "
            "optimal outcome of mutual cooperation."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_HM_MATRIX),
    ),
}

GAMES.update(EXTENDED_MATRIX_GAMES)
