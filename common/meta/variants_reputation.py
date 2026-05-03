"""Reputation gossip variant transform for the composable variant system.

Follows the ``apply_*`` pattern from ``variants.py`` and ``variants_meta.py``.
Adds ``gossip_<rating>_<base_action>`` actions to any base game.
Payoffs depend only on the base action, like cheap_talk.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from common.games import GameConfig

from constant_definitions.var.meta.reputation_constants import (
    VARIANT_GOSSIP,
    DEFAULT_RATINGS,
    GOSSIP_PREFIX,
    GOSSIP_SEPARATOR,
    GOSSIP_SPLIT_LIMIT,
)

_ONE = int(bool(True))
_ZERO = int()
_TWO = _ONE + _ONE


def apply_gossip(
    base: GameConfig,
    ratings: tuple[str, ...] = DEFAULT_RATINGS,
    base_key: str = "",
) -> GameConfig:
    """Add reputation gossip to a base game.

    For base actions ``[A, B]`` and ratings ``[trustworthy, untrustworthy,
    neutral]``, produces ``[gossip_trustworthy_A, gossip_trustworthy_B,
    gossip_untrustworthy_A, ...]``.  Payoffs depend only on the actual
    action (last segment), like cheap_talk.
    """
    sep = GOSSIP_SEPARATOR
    prefix = GOSSIP_PREFIX
    new_actions = [
        sep.join([prefix, rating, act])
        for rating in ratings
        for act in base.actions
    ]
    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        actual_p = pa.rsplit(sep, _ONE)[_ONE]
        actual_o = oa.rsplit(sep, _ONE)[_ONE]
        return original_payoff(actual_p, actual_o)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_GOSSIP,),
        base_game_key=base_key or base.base_game_key,
    )


def parse_gossip_action(action: str) -> tuple[str, str, str]:
    """Parse ``gossip_<rating>_<base_action>`` into components.

    Returns ``(prefix, rating, base_action)``.
    """
    parts = action.split(GOSSIP_SEPARATOR, GOSSIP_SPLIT_LIMIT)
    return (parts[_ZERO], parts[_ONE], parts[_TWO])


_REPUTATION_VARIANT_REGISTRY: dict[str, Callable[..., GameConfig]] = {
    VARIANT_GOSSIP: apply_gossip,
}
