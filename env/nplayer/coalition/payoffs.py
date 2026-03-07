"""Pure functions for computing coalition-adjusted payoffs."""

from __future__ import annotations

from constant_definitions.nplayer.coalition_constants import (
    ENFORCEMENT_CHEAP_TALK,
    ENFORCEMENT_PENALTY,
    ENFORCEMENT_BINDING,
)
from env.nplayer.coalition.models import ActiveCoalition

_ONE = int(bool(True))
_ZERO = int()
_ZERO_F = float()


def compute_coalition_payoffs(
    base_payoffs: tuple[float, ...],
    actions: tuple[str, ...],
    active_coalitions: list[ActiveCoalition],
    enforcement: str,
    penalty_numerator: int,
    penalty_denominator: int,
) -> tuple[tuple[float, ...], list[int], list[float], list[float]]:
    """Compute payoffs adjusted for coalition agreements.

    Returns
    -------
    adjusted_payoffs : tuple[float, ...]
        Payoffs after penalties and side payments.
    defectors : list[int]
        Player indices who broke a coalition agreement.
    penalties : list[float]
        Penalty amount per player (zero for non-defectors).
    side_payments : list[float]
        Net side-payment transfer per player.
    """
    n = len(base_payoffs)
    adjusted = list(base_payoffs)
    penalties = [_ZERO_F] * n
    side_pmts = [_ZERO_F] * n
    defectors: list[int] = []

    # Identify defectors: coalition members who did not play the agreed action
    for coalition in active_coalitions:
        for member in coalition.members:
            if member < n and actions[member] != coalition.agreed_action:
                if member not in defectors:
                    defectors.append(member)

    # Apply enforcement
    if enforcement == ENFORCEMENT_PENALTY:
        for d in defectors:
            penalty = base_payoffs[d] * penalty_numerator / penalty_denominator
            penalties[d] = penalty
            adjusted[d] = base_payoffs[d] - penalty

    # Under cheap_talk, no payoff modification.
    # Under binding, actions were already overridden so defectors list should
    # be empty unless something external bypassed the override.

    # Apply side payments
    for coalition in active_coalitions:
        if coalition.side_payment > _ZERO_F:
            proposer = coalition.members[_ZERO]
            other_members = coalition.members[_ONE:]
            total_paid = coalition.side_payment * len(other_members)
            side_pmts[proposer] -= total_paid
            adjusted[proposer] -= total_paid
            for m in other_members:
                if m < n:
                    side_pmts[m] += coalition.side_payment
                    adjusted[m] += coalition.side_payment

    return tuple(adjusted), defectors, penalties, side_pmts
