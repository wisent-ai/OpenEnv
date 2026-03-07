"""Network and security interaction games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.batch4.network_constants import (
    SG_DEFEND_SUCCESS, SG_ATTACK_FAIL, SG_DEFEND_FAIL, SG_ATTACK_SUCCESS,
    LF_MUTUAL_CONNECT, LF_UNILATERAL_COST, LF_MUTUAL_ISOLATE,
    TWP_CC, TWP_CD, TWP_DC, TWP_DD,
    TWP_CP, TWP_PC, TWP_DP, TWP_PD, TWP_PP,
    DG_EARLY_EARLY, DG_EARLY_LATE, DG_LATE_EARLY, DG_LATE_LATE,
)


# -- Security Game (defender allocates, attacker targets) --
_SG: dict[tuple[str, str], tuple[float, float]] = {
    ("target_a", "target_a"): (float(SG_DEFEND_SUCCESS), float(SG_ATTACK_FAIL)),
    ("target_a", "target_b"): (float(SG_DEFEND_FAIL), float(SG_ATTACK_SUCCESS)),
    ("target_b", "target_a"): (float(SG_DEFEND_FAIL), float(SG_ATTACK_SUCCESS)),
    ("target_b", "target_b"): (float(SG_DEFEND_SUCCESS), float(SG_ATTACK_FAIL)),
}


# -- Link Formation (bilateral consent required) --
_LF_CON = float(LF_MUTUAL_CONNECT)
_LF_UNI = float(LF_UNILATERAL_COST)
_LF_ISO = float(LF_MUTUAL_ISOLATE)

_LF: dict[tuple[str, str], tuple[float, float]] = {
    ("connect", "connect"): (_LF_CON, _LF_CON),
    ("connect", "isolate"): (_LF_UNI, _LF_ISO),
    ("isolate", "connect"): (_LF_ISO, _LF_UNI),
    ("isolate", "isolate"): (_LF_ISO, _LF_ISO),
}


# -- Trust with Punishment (3x3: cooperate, defect, punish) --
_TWP: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(TWP_CC), float(TWP_CC)),
    ("cooperate", "defect"):    (float(TWP_CD), float(TWP_DC)),
    ("cooperate", "punish"):    (float(TWP_CP), float(TWP_PC)),
    ("defect", "cooperate"):    (float(TWP_DC), float(TWP_CD)),
    ("defect", "defect"):       (float(TWP_DD), float(TWP_DD)),
    ("defect", "punish"):       (float(TWP_DP), float(TWP_PD)),
    ("punish", "cooperate"):    (float(TWP_PC), float(TWP_CP)),
    ("punish", "defect"):       (float(TWP_PD), float(TWP_DP)),
    ("punish", "punish"):       (float(TWP_PP), float(TWP_PP)),
}


# -- Dueling Game (fire timing) --
_DG: dict[tuple[str, str], tuple[float, float]] = {
    ("fire_early", "fire_early"): (float(DG_EARLY_EARLY), float(DG_EARLY_EARLY)),
    ("fire_early", "fire_late"):  (float(DG_EARLY_LATE), float(DG_LATE_EARLY)),
    ("fire_late", "fire_early"):  (float(DG_LATE_EARLY), float(DG_EARLY_LATE)),
    ("fire_late", "fire_late"):   (float(DG_LATE_LATE), float(DG_LATE_LATE)),
}


# -- Register --
NETWORK_GAMES: dict[str, GameConfig] = {
    "security_game": GameConfig(
        name="Security Game",
        description=(
            "An attacker-defender game where the defender allocates protection "
            "to one of two targets and the attacker simultaneously chooses "
            "which target to attack. Matching the attacker's target means a "
            "successful defense. Misallocation lets the attacker succeed. "
            "Tests strategic resource allocation under adversarial uncertainty."
        ),
        actions=["target_a", "target_b"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_SG),
    ),
    "link_formation": GameConfig(
        name="Link Formation Game",
        description=(
            "A network formation game where two players simultaneously decide "
            "whether to form a connection. A link forms only when both agree. "
            "Mutual connection yields network benefits. Unilateral connection "
            "attempt is costly. Mutual isolation yields nothing. Tests "
            "bilateral consent in network formation."
        ),
        actions=["connect", "isolate"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_LF),
    ),
    "trust_with_punishment": GameConfig(
        name="Trust with Punishment Game",
        description=(
            "An extended trust game where players can cooperate or defect as "
            "in the standard Prisoner's Dilemma plus a costly punishment "
            "action. Punishing reduces the opponent's payoff but also costs "
            "the punisher. Tests whether altruistic punishment enforces "
            "cooperation even at personal cost."
        ),
        actions=["cooperate", "defect", "punish"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_TWP),
    ),
    "dueling_game": GameConfig(
        name="Dueling Game",
        description=(
            "A timing game where two players simultaneously choose when to "
            "fire: early for a safe but moderate payoff or late for higher "
            "accuracy. Firing early against a late opponent is advantageous. "
            "Mutual late firing yields better outcomes than mutual early. "
            "Tests patience versus preemption under uncertainty."
        ),
        actions=["fire_early", "fire_late"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_DG),
    ),
}

GAMES.update(NETWORK_GAMES)
