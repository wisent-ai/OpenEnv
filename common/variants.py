"""Composable game variant transforms for KantBench.

Each ``apply_*`` function takes a :class:`GameConfig` and returns a new
:class:`GameConfig` with modified actions, payoff function, and metadata.
Variants compose: ``apply_exit(apply_cheap_talk(base))`` works.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import DEFAULT_TWO_PLAYERS
from constant_definitions.var.pd_variant_constants import (
    OPD_EXIT_PAYOFF,
    VARIANT_CHEAP_TALK,
    VARIANT_EXIT,
    VARIANT_BINDING_COMMITMENT,
    VARIANT_NOISY_ACTIONS,
    VARIANT_NOISY_PAYOFFS,
    CT_MSG_PREFIX,
    CT_SEPARATOR,
    BC_COMMIT_PREFIX,
    BC_FREE_PREFIX,
    EXIT_ACTION,
    DEFAULT_TREMBLE_PROB_NUMERATOR,
    DEFAULT_TREMBLE_PROB_DENOMINATOR,
    DEFAULT_NOISE_SCALE_NUMERATOR,
    DEFAULT_NOISE_SCALE_DENOMINATOR,
)
from constant_definitions.var.communication_constants import COMMIT_COST

_ONE = int(bool(True))
_ZERO = int()


def apply_cheap_talk(
    base: GameConfig,
    base_key: str = "",
) -> GameConfig:
    """Add a non-binding message phase to a base game.

    For base actions ``[A, B]`` produces ``[msg_A_A, msg_A_B, msg_B_A,
    msg_B_B]``.  Payoffs depend only on the actual action (last segment).
    """
    sep = CT_SEPARATOR
    prefix = CT_MSG_PREFIX
    base_actions = base.actions
    new_actions = [
        sep.join([prefix, msg, act])
        for msg in base_actions
        for act in base_actions
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
        applied_variants=base.applied_variants + (VARIANT_CHEAP_TALK,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_exit(
    base: GameConfig,
    base_key: str = "",
    exit_payoff: int = OPD_EXIT_PAYOFF,
) -> GameConfig:
    """Add an exit option that gives both players a safe payoff.

    Appends ``"exit"`` to the action list.  If either player exits both
    receive *exit_payoff*; otherwise delegates to the base payoff function.
    """
    exit_f = float(exit_payoff)
    exit_act = EXIT_ACTION
    new_actions = list(base.actions) + [exit_act]
    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        if pa == exit_act or oa == exit_act:
            return (exit_f, exit_f)
        return original_payoff(pa, oa)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_EXIT,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_binding_commitment(
    base: GameConfig,
    base_key: str = "",
    commit_cost: int = COMMIT_COST,
) -> GameConfig:
    """Add a costly binding commitment mechanism.

    For base actions ``[A, B, ...]`` the first action *A* gets a
    ``commit_A`` variant (player locked to *A*, pays *commit_cost*).
    All actions get a ``free_X`` variant (no cost, free choice).
    """
    sep = CT_SEPARATOR
    commit_pfx = BC_COMMIT_PREFIX
    free_pfx = BC_FREE_PREFIX
    cost_f = float(commit_cost)
    base_actions = base.actions
    commit_action = base_actions[_ZERO]

    new_actions = [sep.join([commit_pfx, commit_action])]
    for act in base_actions:
        new_actions.append(sep.join([free_pfx, act]))

    original_payoff = base.payoff_fn

    def _parse(action: str) -> tuple[str, bool]:
        """Return (actual_action, is_committed)."""
        parts = action.split(sep, _ONE)
        return parts[_ONE], parts[_ZERO] == commit_pfx

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        p_act, p_committed = _parse(pa)
        o_act, o_committed = _parse(oa)
        p_pay, o_pay = original_payoff(p_act, o_act)
        if p_committed:
            p_pay = p_pay - cost_f
        if o_committed:
            o_pay = o_pay - cost_f
        return (p_pay, o_pay)

    return replace(
        base,
        actions=new_actions,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_BINDING_COMMITMENT,),
        base_game_key=base_key or base.base_game_key,
    )


_DEFAULT_TREMBLE = DEFAULT_TREMBLE_PROB_NUMERATOR / DEFAULT_TREMBLE_PROB_DENOMINATOR
_DEFAULT_NOISE = DEFAULT_NOISE_SCALE_NUMERATOR / DEFAULT_NOISE_SCALE_DENOMINATOR
_NOISY_ONLY_TWO_PLAYER = "apply_noisy variant only supports two-player games"


def apply_noisy_actions(
    base: GameConfig,
    base_key: str = "",
    tremble_prob: float = _DEFAULT_TREMBLE,
) -> GameConfig:
    """With probability *tremble_prob* each player's action is replaced by a random one."""
    if base.num_players != DEFAULT_TWO_PLAYERS:
        raise ValueError(_NOISY_ONLY_TWO_PLAYER)
    import random as _rng_mod
    original_payoff = base.payoff_fn
    actions = base.actions

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        actual_p = _rng_mod.choice(actions) if _rng_mod.random() < tremble_prob else pa
        actual_o = _rng_mod.choice(actions) if _rng_mod.random() < tremble_prob else oa
        return original_payoff(actual_p, actual_o)

    return replace(
        base,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_NOISY_ACTIONS,),
        base_game_key=base_key or base.base_game_key,
    )


def apply_noisy_payoffs(
    base: GameConfig,
    base_key: str = "",
    noise_scale: float = _DEFAULT_NOISE,
) -> GameConfig:
    """Add Gaussian noise N(zero, noise_scale) to each payoff independently."""
    if base.num_players != DEFAULT_TWO_PLAYERS:
        raise ValueError(_NOISY_ONLY_TWO_PLAYER)
    import random as _rng_mod
    original_payoff = base.payoff_fn

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        p, o = original_payoff(pa, oa)
        return (p + _rng_mod.gauss(float(_ZERO), noise_scale),
                o + _rng_mod.gauss(float(_ZERO), noise_scale))

    return replace(
        base,
        payoff_fn=_payoff,
        applied_variants=base.applied_variants + (VARIANT_NOISY_PAYOFFS,),
        base_game_key=base_key or base.base_game_key,
    )


_VARIANT_REGISTRY: dict[str, Callable[..., GameConfig]] = {
    VARIANT_CHEAP_TALK: apply_cheap_talk,
    VARIANT_EXIT: apply_exit,
    VARIANT_BINDING_COMMITMENT: apply_binding_commitment,
    VARIANT_NOISY_ACTIONS: apply_noisy_actions,
    VARIANT_NOISY_PAYOFFS: apply_noisy_payoffs,
}


def compose_game(base_key: str, *variant_names: str) -> GameConfig:
    """Build a game by applying named variants to a base game.

    Example::

        compose_game("stag_hunt", "cheap_talk", "exit")
    """
    game = GAMES[base_key]
    for vname in variant_names:
        apply_fn = _VARIANT_REGISTRY[vname]
        game = apply_fn(game, base_key=base_key)
    return game
