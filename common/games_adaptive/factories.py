"""Adaptive payoff game factories with history-dependent payoff functions."""
from __future__ import annotations
from typing import Callable
from common.games import GameConfig, GAME_FACTORIES, _PD_MATRIX, _HD_MATRIX
from constant_definitions.game_constants import (
    TRUST_MULTIPLIER, EVAL_ZERO_FLOAT, EVAL_ONE_FLOAT,
)
from constant_definitions.var.meta.adaptive_constants import (
    ADAPTIVE_PD_MULTIPLIER_MIN_NUMERATOR,
    ADAPTIVE_PD_MULTIPLIER_MIN_DENOMINATOR,
    ADAPTIVE_PD_MULTIPLIER_MAX_NUMERATOR,
    ADAPTIVE_PD_MULTIPLIER_MAX_DENOMINATOR,
    ADAPTIVE_PD_MULTIPLIER_STEP_NUMERATOR,
    ADAPTIVE_PD_MULTIPLIER_STEP_DENOMINATOR,
    ARMS_RACE_COST_STEP_NUMERATOR, ARMS_RACE_COST_STEP_DENOMINATOR,
    ARMS_RACE_MAX_COST_NUMERATOR, ARMS_RACE_MAX_COST_DENOMINATOR,
    TRUST_EROSION_DECAY_NUMERATOR, TRUST_EROSION_DECAY_DENOMINATOR,
    TRUST_EROSION_RECOVERY_NUMERATOR, TRUST_EROSION_RECOVERY_DENOMINATOR,
    MARKET_DEMAND_SHIFT_NUMERATOR, MARKET_DEMAND_SHIFT_DENOMINATOR,
    REPUTATION_BONUS_NUMERATOR, REPUTATION_BONUS_DENOMINATOR,
    ADAPTIVE_DEFAULT_ROUNDS, ADAPTIVE_GAME_TYPE,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE

# Market dynamics tables
_MKT_OUT = {"low": _TWO, "medium": _TWO + _TWO, "high": _TWO * _TWO + _TWO}
_MKT_COST = {"low": _ONE, "medium": _TWO + _ONE, "high": _TWO * _TWO + _TWO}
_MKT_INTERCEPT = (_TWO + _TWO) * (_TWO + _ONE)

def _adaptive_pd_factory() -> GameConfig:
    """PD where mutual cooperation increases future payoffs."""
    min_m = ADAPTIVE_PD_MULTIPLIER_MIN_NUMERATOR / ADAPTIVE_PD_MULTIPLIER_MIN_DENOMINATOR
    max_m = ADAPTIVE_PD_MULTIPLIER_MAX_NUMERATOR / ADAPTIVE_PD_MULTIPLIER_MAX_DENOMINATOR
    step = ADAPTIVE_PD_MULTIPLIER_STEP_NUMERATOR / ADAPTIVE_PD_MULTIPLIER_STEP_DENOMINATOR
    _s = [EVAL_ONE_FLOAT]

    def payoff_fn(p_act: str, o_act: str) -> tuple[float, float]:
        mult = _s[_ZERO]
        base = _PD_MATRIX[(p_act, o_act)]
        result = (base[_ZERO] * mult, base[_ONE] * mult)
        if p_act == "cooperate" and o_act == "cooperate":
            _s[_ZERO] = min(max_m, _s[_ZERO] + step)
        elif p_act == "defect" and o_act == "defect":
            _s[_ZERO] = max(min_m, _s[_ZERO] - step)
        return result

    return GameConfig(
        name="Adaptive Prisoner's Dilemma",
        description=(
            "A Prisoner's Dilemma where mutual cooperation increases "
            "future payoffs via a growing multiplier, while mutual "
            "defection decreases it. Mixed outcomes leave it unchanged."
        ),
        actions=["cooperate", "defect"],
        game_type=ADAPTIVE_GAME_TYPE,
        default_rounds=ADAPTIVE_DEFAULT_ROUNDS,
        payoff_fn=payoff_fn,
    )


def _arms_race_factory() -> GameConfig:
    """Hawk-Dove where hawk-hawk conflict costs escalate each round."""
    c_step = ARMS_RACE_COST_STEP_NUMERATOR / ARMS_RACE_COST_STEP_DENOMINATOR
    max_c = ARMS_RACE_MAX_COST_NUMERATOR / ARMS_RACE_MAX_COST_DENOMINATOR
    _s = [EVAL_ZERO_FLOAT]

    def payoff_fn(p_act: str, o_act: str) -> tuple[float, float]:
        cost = _s[_ZERO]
        base = _HD_MATRIX[(p_act, o_act)]
        if p_act == "hawk" and o_act == "hawk":
            result = (base[_ZERO] - cost, base[_ONE] - cost)
            _s[_ZERO] = min(max_c, _s[_ZERO] + c_step)
        else:
            result = base
            _s[_ZERO] = max(EVAL_ZERO_FLOAT, _s[_ZERO] - c_step / _TWO)
        return result

    return GameConfig(
        name="Arms Race",
        description=(
            "A Hawk-Dove game where mutual hawk play incurs "
            "escalating costs each round. Non-hawk rounds "
            "de-escalate the accumulated conflict cost."
        ),
        actions=["hawk", "dove"],
        game_type=ADAPTIVE_GAME_TYPE,
        default_rounds=ADAPTIVE_DEFAULT_ROUNDS,
        payoff_fn=payoff_fn,
    )


def _trust_erosion_factory() -> GameConfig:
    """Trust-like PD where a multiplier decays after mutual defection."""
    decay = TRUST_EROSION_DECAY_NUMERATOR / TRUST_EROSION_DECAY_DENOMINATOR
    recov = TRUST_EROSION_RECOVERY_NUMERATOR / TRUST_EROSION_RECOVERY_DENOMINATOR
    _s = [float(TRUST_MULTIPLIER)]

    def payoff_fn(p_act: str, o_act: str) -> tuple[float, float]:
        mult = _s[_ZERO]
        base = _PD_MATRIX[(p_act, o_act)]
        result = (base[_ZERO] * mult, base[_ONE] * mult)
        if p_act == "defect" and o_act == "defect":
            _s[_ZERO] = _s[_ZERO] * decay
        elif p_act == "cooperate" and o_act == "cooperate":
            _s[_ZERO] = min(float(TRUST_MULTIPLIER), _s[_ZERO] + recov)
        return result

    return GameConfig(
        name="Trust Erosion",
        description=(
            "A Prisoner's Dilemma where a trust multiplier amplifies "
            "all payoffs. Mutual defection erodes trust, while mutual "
            "cooperation slowly rebuilds it."
        ),
        actions=["cooperate", "defect"],
        game_type=ADAPTIVE_GAME_TYPE,
        default_rounds=ADAPTIVE_DEFAULT_ROUNDS,
        payoff_fn=payoff_fn,
    )


def _market_dynamics_factory() -> GameConfig:
    """Cournot-like duopoly where demand shifts based on total output."""
    shift = MARKET_DEMAND_SHIFT_NUMERATOR / MARKET_DEMAND_SHIFT_DENOMINATOR
    _s = [float(_MKT_INTERCEPT)]

    def payoff_fn(p_act: str, o_act: str) -> tuple[float, float]:
        intercept = _s[_ZERO]
        p_out, o_out = _MKT_OUT[p_act], _MKT_OUT[o_act]
        total = p_out + o_out
        price = max(EVAL_ZERO_FLOAT, intercept - total)
        p_rev = price * p_out - _MKT_COST[p_act]
        o_rev = price * o_out - _MKT_COST[o_act]
        if total > (_MKT_INTERCEPT / _TWO):
            _s[_ZERO] = max(float(_TWO), _s[_ZERO] - shift)
        else:
            _s[_ZERO] = min(float(_MKT_INTERCEPT), _s[_ZERO] + shift)
        return (p_rev, o_rev)

    return GameConfig(
        name="Market Dynamics",
        description=(
            "A Cournot-like duopoly where each player chooses output "
            "level. The demand curve shifts based on past total output: "
            "high output depresses future demand, restraint recovers it."
        ),
        actions=["low", "medium", "high"],
        game_type=ADAPTIVE_GAME_TYPE,
        default_rounds=ADAPTIVE_DEFAULT_ROUNDS,
        payoff_fn=payoff_fn,
    )


def _reputation_payoffs_factory() -> GameConfig:
    """Base PD with payoff bonus proportional to cooperation history."""
    bonus_rate = REPUTATION_BONUS_NUMERATOR / REPUTATION_BONUS_DENOMINATOR
    _s = [_ZERO, _ZERO]  # [coop_count, total_rounds]

    def payoff_fn(p_act: str, o_act: str) -> tuple[float, float]:
        base = _PD_MATRIX[(p_act, o_act)]
        total = _s[_ONE]
        coop_rate = _s[_ZERO] / total if total > _ZERO else EVAL_ZERO_FLOAT
        bonus = coop_rate * bonus_rate
        result = (base[_ZERO] + bonus, base[_ONE] + bonus)
        _s[_ONE] += _ONE
        if p_act == "cooperate":
            _s[_ZERO] += _ONE
        return result

    return GameConfig(
        name="Reputation Payoffs",
        description=(
            "A Prisoner's Dilemma where both players receive a bonus "
            "proportional to the player's historical cooperation rate. "
            "Building a cooperative reputation pays future dividends."
        ),
        actions=["cooperate", "defect"],
        game_type=ADAPTIVE_GAME_TYPE,
        default_rounds=ADAPTIVE_DEFAULT_ROUNDS,
        payoff_fn=payoff_fn,
    )


# Register all factories
GAME_FACTORIES["adaptive_prisoners_dilemma"] = _adaptive_pd_factory
GAME_FACTORIES["arms_race"] = _arms_race_factory
GAME_FACTORIES["trust_erosion"] = _trust_erosion_factory
GAME_FACTORIES["market_dynamics"] = _market_dynamics_factory
GAME_FACTORIES["reputation_payoffs"] = _reputation_payoffs_factory
