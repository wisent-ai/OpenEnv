"""Rule catalog and payoff transforms for the meta-gaming variant system.

Each rule is a payoff transform: given base payoffs and actions, return
modified payoffs. The ``apply_rule`` dispatcher looks up a rule by name
and delegates to the corresponding transform function.
"""
from __future__ import annotations

from typing import Callable

from constant_definitions.var.meta.meta_rule_constants import (
    RULE_NONE, RULE_EQUAL_SPLIT, RULE_COOP_BONUS,
    RULE_DEFECT_PENALTY, RULE_MIN_GUARANTEE, RULE_BAN_DEFECT,
    COOP_BONUS_NUMERATOR, COOP_BONUS_DENOMINATOR,
    DEFECT_PENALTY_NUMERATOR, DEFECT_PENALTY_DENOMINATOR,
    MIN_GUARANTEE_NUMERATOR, MIN_GUARANTEE_DENOMINATOR,
    BAN_DEFECT_PENALTY_NUMERATOR, BAN_DEFECT_PENALTY_DENOMINATOR,
    EQUAL_SPLIT_DENOMINATOR,
    META_SEPARATOR, META_SPLIT_LIMIT,
)

RuleTransform = Callable[
    [float, float, str, str], tuple[float, float]
]

_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove"})

_COOP_BONUS = COOP_BONUS_NUMERATOR / COOP_BONUS_DENOMINATOR
_DEFECT_PENALTY = DEFECT_PENALTY_NUMERATOR / DEFECT_PENALTY_DENOMINATOR
_MIN_GUARANTEE = MIN_GUARANTEE_NUMERATOR / MIN_GUARANTEE_DENOMINATOR
_BAN_PENALTY = BAN_DEFECT_PENALTY_NUMERATOR / BAN_DEFECT_PENALTY_DENOMINATOR


def _is_cooperative(action: str) -> bool:
    """Return True if *action* is a cooperative action."""
    return action in _COOPERATIVE_ACTIONS


def _rule_none(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    return (base_p, base_o)


def _rule_equal_split(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    total = base_p + base_o
    share = total / EQUAL_SPLIT_DENOMINATOR
    return (share, share)


def _rule_coop_bonus(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    p_pay = base_p + (_COOP_BONUS if _is_cooperative(p_action) else float())
    o_pay = base_o + (_COOP_BONUS if _is_cooperative(o_action) else float())
    return (p_pay, o_pay)


def _rule_defect_penalty(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    p_pay = base_p - (
        _DEFECT_PENALTY if not _is_cooperative(p_action) else float()
    )
    o_pay = base_o - (
        _DEFECT_PENALTY if not _is_cooperative(o_action) else float()
    )
    return (p_pay, o_pay)


def _rule_min_guarantee(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    p_pay = max(base_p, _MIN_GUARANTEE)
    o_pay = max(base_o, _MIN_GUARANTEE)
    return (p_pay, o_pay)


def _rule_ban_defect(
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    p_pay = base_p - (
        _BAN_PENALTY if not _is_cooperative(p_action) else float()
    )
    o_pay = base_o - (
        _BAN_PENALTY if not _is_cooperative(o_action) else float()
    )
    return (p_pay, o_pay)


RULE_CATALOG: dict[str, RuleTransform] = {
    RULE_NONE: _rule_none,
    RULE_EQUAL_SPLIT: _rule_equal_split,
    RULE_COOP_BONUS: _rule_coop_bonus,
    RULE_DEFECT_PENALTY: _rule_defect_penalty,
    RULE_MIN_GUARANTEE: _rule_min_guarantee,
    RULE_BAN_DEFECT: _rule_ban_defect,
}


def apply_rule(
    rule_name: str,
    base_p: float, base_o: float,
    p_action: str, o_action: str,
) -> tuple[float, float]:
    """Look up *rule_name* in the catalog and apply its transform."""
    return RULE_CATALOG[rule_name](base_p, base_o, p_action, o_action)


_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE


def parse_meta_action(
    action: str,
    split_limit: int = META_SPLIT_LIMIT,
) -> tuple[str, str, str]:
    """Parse an encoded meta-action into (prefix, rule, base_action).

    The action format is ``prefix_rule_baseaction`` where *rule* is a
    single token (no underscores).  Using ``split`` with the configured
    split limit yields exactly three parts.
    """
    parts = action.split(META_SEPARATOR, split_limit)
    return (parts[_ZERO], parts[_ONE], parts[_TWO])
