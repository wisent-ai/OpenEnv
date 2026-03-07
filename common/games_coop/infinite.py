"""Infinite-horizon and continuous games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
from constant_definitions.var.infinite_constants import (
    CPD_BENEFIT_NUMERATOR, CPD_COST_NUMERATOR, CPD_DENOMINATOR,
    CPD_MAX_LEVEL,
    DPD_TEMPTATION, DPD_REWARD, DPD_PUNISHMENT, DPD_SUCKER,
    DPD_DEFAULT_ROUNDS,
)

_ONE = int(bool(True))


# -- Continuous PD (variable contribution levels) --
def _continuous_pd_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Each player chooses a cooperation level. Higher = costlier but benefits opponent."""
    lvl_p = int(pa.rsplit("_", _ONE)[_ONE])
    lvl_o = int(oa.rsplit("_", _ONE)[_ONE])
    p_pay = float(lvl_o * CPD_BENEFIT_NUMERATOR) / CPD_DENOMINATOR
    p_pay -= float(lvl_p * CPD_COST_NUMERATOR) / CPD_DENOMINATOR
    o_pay = float(lvl_p * CPD_BENEFIT_NUMERATOR) / CPD_DENOMINATOR
    o_pay -= float(lvl_o * CPD_COST_NUMERATOR) / CPD_DENOMINATOR
    return (p_pay, o_pay)


_CPD_ACTS = [f"level_{i}" for i in range(CPD_MAX_LEVEL + _ONE)]


# -- Discounted PD (high-stakes, long-horizon) --
_DPD_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(DPD_REWARD), float(DPD_REWARD)),
    ("cooperate", "defect"):    (float(DPD_SUCKER), float(DPD_TEMPTATION)),
    ("defect", "cooperate"):    (float(DPD_TEMPTATION), float(DPD_SUCKER)),
    ("defect", "defect"):       (float(DPD_PUNISHMENT), float(DPD_PUNISHMENT)),
}


# -- Register --
INFINITE_GAMES: dict[str, GameConfig] = {
    "continuous_pd": GameConfig(
        name="Continuous Prisoner's Dilemma",
        description=(
            "A generalization of the Prisoner's Dilemma with variable "
            "cooperation levels instead of binary choices. Each unit of "
            "cooperation costs the player but benefits the opponent more. "
            "Tests whether agents find intermediate cooperation strategies "
            "in continuous action spaces."
        ),
        actions=_CPD_ACTS,
        game_type="continuous_pd",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_continuous_pd_payoff,
    ),
    "discounted_pd": GameConfig(
        name="Discounted Prisoner's Dilemma",
        description=(
            "A high-stakes Prisoner's Dilemma with many rounds, modeling "
            "an effectively infinite repeated interaction. The shadow of "
            "the future makes cooperation sustainable under folk theorem "
            "conditions. Tests long-horizon strategic reasoning with "
            "higher temptation and reward differentials."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DPD_DEFAULT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_DPD_MATRIX),
    ),
}

GAMES.update(INFINITE_GAMES)
