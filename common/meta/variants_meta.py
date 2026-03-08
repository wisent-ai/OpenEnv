"""Meta-gaming variant transforms for rule proposal, signaling, and negotiation.

Four composable transforms following the ``apply_*`` pattern from
``variants.py``.  Each expands the action space to encode both a rule
proposal and a base-game action in a single string.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from common.games import GameConfig
from common.meta.meta_rules import apply_rule, parse_meta_action

from constant_definitions.var.meta.meta_rule_constants import (
    VARIANT_RULE_PROPOSAL, VARIANT_RULE_SIGNAL,
    VARIANT_CONSTITUTIONAL, VARIANT_PROPOSER_RESPONDER,
    META_PROP_PREFIX, META_SIG_PREFIX, META_CONST_PREFIX,
    META_RPROP_PREFIX, META_RACCEPT_PREFIX, META_RREJECT_PREFIX,
    META_SEPARATOR,
    DEFAULT_RULE_CATALOG, RULE_NONE,
)

_ONE = int(bool(True))
_ZERO = int()


def _build_prefixed_actions(
    prefix: str,
    rules: tuple[str, ...],
    base_actions: list[str],
) -> list[str]:
    """Build action list: prefix_rule_baseaction for each combination."""
    sep = META_SEPARATOR
    return [
        sep.join([prefix, rule, act])
        for rule in rules
        for act in base_actions
    ]


def apply_rule_proposal(
    base: GameConfig,
    rules: tuple[str, ...] = DEFAULT_RULE_CATALOG,
    base_key: str = "",
) -> GameConfig:
    """Simultaneous, binding, per-round rule proposal.

    Both players choose ``prop_<rule>_<action>``.  If both propose the
    same rule the rule's payoff transform is applied; otherwise base
    payoffs are used.
    """
    prefix = META_PROP_PREFIX
    new_actions = _build_prefixed_actions(prefix, rules, base.actions)
    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        _, p_rule, p_act = parse_meta_action(pa)
        _, o_rule, o_act = parse_meta_action(oa)
        base_p, base_o = original_payoff(p_act, o_act)
        if p_rule == o_rule:
            return apply_rule(p_rule, base_p, base_o, p_act, o_act)
        return (base_p, base_o)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_RULE_PROPOSAL,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_rule_signal(
    base: GameConfig,
    rules: tuple[str, ...] = DEFAULT_RULE_CATALOG,
    base_key: str = "",
) -> GameConfig:
    """Simultaneous, non-binding, per-round rule signal.

    Both players choose ``sig_<rule>_<action>``.  Proposals are visible
    in history but never enforced -- payoffs always come from the base game.
    """
    prefix = META_SIG_PREFIX
    new_actions = _build_prefixed_actions(prefix, rules, base.actions)
    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        _, _p_rule, p_act = parse_meta_action(pa)
        _, _o_rule, o_act = parse_meta_action(oa)
        return original_payoff(p_act, o_act)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_RULE_SIGNAL,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_constitutional(
    base: GameConfig,
    rules: tuple[str, ...] = DEFAULT_RULE_CATALOG,
    base_key: str = "",
) -> GameConfig:
    """Multi-round negotiation with binding lock-in once agreed.

    Both players choose ``const_<rule>_<action>``.  The first round
    where both propose the same non-none rule locks that rule in for
    ALL subsequent rounds.  Before agreement, base payoffs apply.

    A fresh mutable closure is created per call so each episode via
    ``compose_game()`` gets clean state.
    """
    prefix = META_CONST_PREFIX
    new_actions = _build_prefixed_actions(prefix, rules, base.actions)
    original_payoff = base.payoff_fn
    adopted_rule: list[str] = []

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        _, p_rule, p_act = parse_meta_action(pa)
        _, o_rule, o_act = parse_meta_action(oa)
        base_p, base_o = original_payoff(p_act, o_act)

        if adopted_rule:
            return apply_rule(adopted_rule[_ZERO], base_p, base_o, p_act, o_act)

        if p_rule == o_rule and p_rule != RULE_NONE:
            adopted_rule.append(p_rule)
            return apply_rule(p_rule, base_p, base_o, p_act, o_act)

        return (base_p, base_o)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_CONSTITUTIONAL,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_proposer_responder(
    base: GameConfig,
    rules: tuple[str, ...] = DEFAULT_RULE_CATALOG,
    base_key: str = "",
) -> GameConfig:
    """Asymmetric: player proposes a rule, opponent accepts or rejects.

    Player actions: ``rprop_<rule>_<action>`` (propose + play).
    Opponent actions: ``raccept_<action>`` or ``rreject_<action>``
    (respond + play).

    Accept -> rule applies to base payoffs.  Reject -> base payoffs.
    """
    sep = META_SEPARATOR
    player_actions = _build_prefixed_actions(
        META_RPROP_PREFIX, rules, base.actions,
    )
    opp_actions: list[str] = []
    for act in base.actions:
        opp_actions.append(sep.join([META_RACCEPT_PREFIX, act]))
        opp_actions.append(sep.join([META_RREJECT_PREFIX, act]))

    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        _, p_rule, p_act = parse_meta_action(pa)
        o_parts = oa.split(sep, _ONE)
        o_prefix = o_parts[_ZERO]
        o_act = o_parts[_ONE]
        base_p, base_o = original_payoff(p_act, o_act)
        if o_prefix == META_RACCEPT_PREFIX:
            return apply_rule(p_rule, base_p, base_o, p_act, o_act)
        return (base_p, base_o)

    return replace(
        base,
        actions=player_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_PROPOSER_RESPONDER,),
        base_game_key=base_key or base.base_game_key,
        opponent_actions=tuple(opp_actions),
    )


_META_VARIANT_REGISTRY: dict[str, Callable[..., GameConfig]] = {
    VARIANT_RULE_PROPOSAL: apply_rule_proposal,
    VARIANT_RULE_SIGNAL: apply_rule_signal,
    VARIANT_CONSTITUTIONAL: apply_constitutional,
    VARIANT_PROPOSER_RESPONDER: apply_proposer_responder,
}
