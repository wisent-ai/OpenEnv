"""Governance-aware opponent strategies."""

from __future__ import annotations

import random
from typing import Protocol

from env.nplayer.governance.models import GovernanceProposal, GovernanceVote

_ZERO = int()
_ONE = int(bool(True))


class GovernanceStrategy(Protocol):
    """Interface for governance opponent behaviour."""

    def propose_governance(self, player_index: int) -> list[GovernanceProposal]: ...

    def vote_on_governance(
        self, player_index: int, proposals: list[GovernanceProposal],
    ) -> list[GovernanceVote]: ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class GovernancePassiveStrategy:
    """No proposals, no votes."""

    def propose_governance(self, player_index: int) -> list[GovernanceProposal]:
        return []

    def vote_on_governance(
        self, player_index: int, proposals: list[GovernanceProposal],
    ) -> list[GovernanceVote]:
        return []


class GovernanceRandomStrategy:
    """No proposals, random votes."""

    def propose_governance(self, player_index: int) -> list[GovernanceProposal]:
        return []

    def vote_on_governance(
        self, player_index: int, proposals: list[GovernanceProposal],
    ) -> list[GovernanceVote]:
        return [
            GovernanceVote(voter=player_index, proposal_index=idx, approve=random.choice([True, False]))
            for idx in range(len(proposals))
        ]


class GovernanceConservativeStrategy:
    """No proposals, rejects all."""

    def propose_governance(self, player_index: int) -> list[GovernanceProposal]:
        return []

    def vote_on_governance(
        self, player_index: int, proposals: list[GovernanceProposal],
    ) -> list[GovernanceVote]:
        return [
            GovernanceVote(voter=player_index, proposal_index=idx, approve=False)
            for idx in range(len(proposals))
        ]


class GovernanceProgressiveStrategy:
    """No proposals, approves all."""

    def propose_governance(self, player_index: int) -> list[GovernanceProposal]:
        return []

    def vote_on_governance(
        self, player_index: int, proposals: list[GovernanceProposal],
    ) -> list[GovernanceVote]:
        return [
            GovernanceVote(voter=player_index, proposal_index=idx, approve=True)
            for idx in range(len(proposals))
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GOVERNANCE_STRATEGIES: dict[str, GovernanceStrategy] = {
    "governance_passive": GovernancePassiveStrategy(),
    "governance_random": GovernanceRandomStrategy(),
    "governance_conservative": GovernanceConservativeStrategy(),
    "governance_progressive": GovernanceProgressiveStrategy(),
}


def get_governance_strategy(name: str) -> GovernanceStrategy:
    """Look up a governance strategy by name. Raises KeyError if not found."""
    return GOVERNANCE_STRATEGIES[name]
