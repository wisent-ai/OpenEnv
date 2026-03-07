"""Contest, conflict, and fair division games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.ext.conflict_constants import (
    BLOTTO_BATTLEFIELDS, BLOTTO_TOTAL_TROOPS,
    WOA_PRIZE, WOA_COST_PER_ROUND, WOA_MAX_PERSISTENCE,
    TULLOCK_PRIZE, TULLOCK_MAX_EFFORT,
    INSP_VIOLATION_GAIN, INSP_FINE, INSP_INSPECTION_COST,
    INSP_COMPLIANCE_PAYOFF,
    RUB_SURPLUS, RUB_DISCOUNT_NUM, RUB_DISCOUNT_DEN,
    DAC_ENDOWMENT,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE
_ZERO_F = float()


# -- Colonel Blotto (three battlefields, encoded as alloc_X_Y_Z) --
def _blotto_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Each player allocates troops across battlefields. Most wins per field."""
    p_parts = pa.split("_")[_ONE:]
    o_parts = oa.split("_")[_ONE:]
    p_wins = int()
    o_wins = int()
    for pv, ov in zip(p_parts, o_parts):
        pi, oi = int(pv), int(ov)
        if pi > oi:
            p_wins += _ONE
        elif oi > pi:
            o_wins += _ONE
    return (float(p_wins), float(o_wins))


def _generate_blotto_actions() -> list[str]:
    """Generate all valid troop allocations across battlefields."""
    actions = []
    for a in range(BLOTTO_TOTAL_TROOPS + _ONE):
        for b in range(BLOTTO_TOTAL_TROOPS - a + _ONE):
            c = BLOTTO_TOTAL_TROOPS - a - b
            actions.append(f"alloc_{a}_{b}_{c}")
    return actions


_BLOTTO_ACTS = _generate_blotto_actions()


# -- War of Attrition --
def _woa_payoff(pa: str, oa: str) -> tuple[float, float]:
    p_pers = int(pa.rsplit("_", _ONE)[_ONE])
    o_pers = int(oa.rsplit("_", _ONE)[_ONE])
    if p_pers > o_pers:
        return (float(WOA_PRIZE - p_pers * WOA_COST_PER_ROUND),
                float(-o_pers * WOA_COST_PER_ROUND))
    if o_pers > p_pers:
        return (float(-p_pers * WOA_COST_PER_ROUND),
                float(WOA_PRIZE - o_pers * WOA_COST_PER_ROUND))
    half = float(WOA_PRIZE) / _TWO
    cost = float(p_pers * WOA_COST_PER_ROUND)
    return (half - cost, half - cost)


_WOA_ACTS = [f"persist_{i}" for i in range(WOA_MAX_PERSISTENCE + _ONE)]


# -- Tullock Contest --
def _tullock_payoff(pa: str, oa: str) -> tuple[float, float]:
    e_p = int(pa.rsplit("_", _ONE)[_ONE])
    e_o = int(oa.rsplit("_", _ONE)[_ONE])
    total = e_p + e_o
    if total == int():
        half = float(TULLOCK_PRIZE) / _TWO
        return (half, half)
    p_prob = float(e_p) / float(total)
    return (float(p_prob * TULLOCK_PRIZE - e_p),
            float((_ONE - p_prob) * TULLOCK_PRIZE - e_o))


_TULLOCK_ACTS = [f"effort_{i}" for i in range(TULLOCK_MAX_EFFORT + _ONE)]


# -- Inspection Game --
_INSP_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("violate", "inspect"):   (float(-INSP_FINE), float(INSP_FINE - INSP_INSPECTION_COST)),
    ("violate", "no_inspect"): (float(INSP_VIOLATION_GAIN), float(int())),
    ("comply", "inspect"):    (float(INSP_COMPLIANCE_PAYOFF), float(-INSP_INSPECTION_COST)),
    ("comply", "no_inspect"): (float(INSP_COMPLIANCE_PAYOFF), float(int())),
}


# -- Rubinstein Bargaining (modeled as demand with discount) --
def _rubinstein_payoff(pa: str, oa: str) -> tuple[float, float]:
    d_p = int(pa.rsplit("_", _ONE)[_ONE])
    d_o = int(oa.rsplit("_", _ONE)[_ONE])
    if d_p + d_o <= RUB_SURPLUS:
        return (float(d_p), float(d_o))
    disc_p = float(d_p * RUB_DISCOUNT_NUM) / float(RUB_DISCOUNT_DEN)
    disc_o = float(d_o * RUB_DISCOUNT_NUM) / float(RUB_DISCOUNT_DEN)
    if d_p + d_o <= RUB_SURPLUS + _TWO:
        return (disc_p, disc_o)
    return (_ZERO_F, _ZERO_F)


_RUB_ACTS = [f"demand_{i}" for i in range(RUB_SURPLUS + _ONE)]


# -- Divide-and-Choose --
def _dac_payoff(pa: str, oa: str) -> tuple[float, float]:
    split = int(pa.rsplit("_", _ONE)[_ONE])
    choice = oa
    left_piece = split
    right_piece = DAC_ENDOWMENT - split
    if choice == "choose_left":
        return (float(right_piece), float(left_piece))
    return (float(left_piece), float(right_piece))


_DAC_SPLIT_ACTS = [f"split_{i}" for i in range(DAC_ENDOWMENT + _ONE)]

CONTEST_GAMES: dict[str, GameConfig] = {
    "colonel_blotto": GameConfig(
        name="Colonel Blotto",
        description=(
            "Two players allocate limited troops across multiple "
            "battlefields. The player with more troops wins each field. "
            "Tests multi-dimensional strategic resource allocation."
        ),
        actions=_BLOTTO_ACTS, game_type="blotto",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_blotto_payoff,
    ),
    "war_of_attrition": GameConfig(
        name="War of Attrition",
        description=(
            "Both players choose how long to persist. The survivor wins "
            "a prize but both pay costs for duration. Tests endurance "
            "strategy and rent dissipation reasoning."
        ),
        actions=_WOA_ACTS, game_type="war_of_attrition",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_woa_payoff,
    ),
    "tullock_contest": GameConfig(
        name="Tullock Contest",
        description=(
            "Players invest effort to win a prize. Win probability is "
            "proportional to relative effort. Models lobbying, rent-seeking, "
            "and competitive R&D spending."
        ),
        actions=_TULLOCK_ACTS, game_type="tullock",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_tullock_payoff,
    ),
    "inspection_game": GameConfig(
        name="Inspection Game",
        description=(
            "A potential violator chooses to comply or violate; an inspector "
            "chooses whether to inspect. Mixed-strategy equilibrium models "
            "compliance, auditing, and arms control verification."
        ),
        actions=["violate", "comply"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_INSP_MATRIX),
    ),
    "rubinstein_bargaining": GameConfig(
        name="Rubinstein Bargaining",
        description=(
            "Players make simultaneous demands over a surplus. Compatible "
            "demands yield immediate payoff; excessive demands are "
            "discounted. Models alternating-offers bargaining with "
            "time preference."
        ),
        actions=_RUB_ACTS, game_type="rubinstein",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_rubinstein_payoff,
    ),
    "divide_and_choose": GameConfig(
        name="Divide-and-Choose",
        description=(
            "The divider splits a resource into two portions; the "
            "chooser takes their preferred portion. The optimal "
            "strategy for the divider is an even split. Tests "
            "envy-free fair division reasoning."
        ),
        actions=_DAC_SPLIT_ACTS, game_type="divide_choose",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_dac_payoff,
    ),
}

GAMES.update(CONTEST_GAMES)
