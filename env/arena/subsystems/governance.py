"""ArenaGovernance — proposal/vote/tally for the metagame arena."""
from __future__ import annotations

from typing import Any, Optional

from common.games_meta.dynamic import create_matrix_game
from constant_definitions.arena.arena_constants import (
    PROPOSAL_BAN,
    PROPOSAL_ADD,
    PROPOSAL_RULE,
    PROPOSAL_NEW_GAME,
    PROPOSAL_TYPES,
    MAX_PROPOSALS_PER_ROUND,
    BAN_THRESHOLD_NUMERATOR,
    BAN_THRESHOLD_DENOMINATOR,
    RULE_THRESHOLD_NUMERATOR,
    RULE_THRESHOLD_DENOMINATOR,
)
from env.arena.models import ArenaProposal, ArenaVote

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()


class ArenaGovernance:
    """Manages governance proposals, voting, and resolution.

    Mirrors the ``GovernanceEngine`` from ``env.nplayer.governance.engine``
    but uses reputation-weighted voting and arena-specific proposal types.
    """

    def __init__(self) -> None:
        self._pending: list[ArenaProposal] = []
        self._history: list[dict[str, Any]] = []

    @property
    def pending_proposals(self) -> list[ArenaProposal]:
        return list(self._pending)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def submit_proposals(
        self,
        proposals: list[ArenaProposal],
        active_models: list[str],
    ) -> list[ArenaProposal]:
        """Validate and queue proposals. Returns accepted proposals."""
        accepted: list[ArenaProposal] = []
        for prop in proposals:
            if len(self._pending) >= MAX_PROPOSALS_PER_ROUND:
                break
            if prop.proposer not in active_models:
                continue
            if prop.proposal_type not in PROPOSAL_TYPES:
                continue
            if not self._validate(prop, active_models):
                continue
            self._pending.append(prop)
            accepted.append(prop)
        return accepted

    def tally_votes(
        self,
        votes: list[ArenaVote],
        active_models: list[str],
    ) -> tuple[list[int], list[int]]:
        """Count weighted votes. Returns (adopted_indices, rejected_indices)."""
        total_weight = _ZERO_F
        for v in votes:
            if v.voter in active_models:
                total_weight += v.weight
        approve_weights: dict[int, float] = {}
        for v in votes:
            if v.voter not in active_models:
                continue
            if v.approve:
                approve_weights[v.proposal_index] = (
                    approve_weights.get(v.proposal_index, _ZERO_F) + v.weight
                )
        adopted: list[int] = []
        rejected: list[int] = []
        for idx, prop in enumerate(self._pending):
            threshold = self._threshold_for(prop, total_weight)
            if approve_weights.get(idx, _ZERO_F) >= threshold:
                adopted.append(idx)
            else:
                rejected.append(idx)
        result = {
            "proposals": [p.model_dump() for p in self._pending],
            "votes": [v.model_dump() for v in votes],
            "adopted": adopted,
            "rejected": rejected,
        }
        self._history.append(result)
        proposals_snapshot = list(self._pending)
        self._pending = []
        return adopted, rejected

    def apply_adopted(
        self, adopted_indices: list[int], proposals: list[ArenaProposal],
    ) -> list[dict[str, Any]]:
        """Return a list of actions to perform for adopted proposals."""
        actions: list[dict[str, Any]] = []
        for idx in adopted_indices:
            if idx >= len(proposals):
                continue
            prop = proposals[idx]
            actions.append({
                "type": prop.proposal_type,
                "target_model": prop.target_model,
                "rule_description": prop.rule_description,
                "game_definition": prop.game_definition,
            })
        return actions

    def create_proposed_game(
        self, game_def: dict[str, Any],
    ) -> Optional[str]:
        """Try to create a game from a proposal's game_definition."""
        try:
            name = game_def.get("name", "custom")
            actions = game_def.get("actions", [])
            matrix = {}
            for key_str, val in game_def.get("payoff_matrix", {}).items():
                parts = key_str.split(",") if isinstance(key_str, str) else key_str
                matrix[(parts[_ZERO].strip(), parts[_ONE].strip())] = tuple(val)
            create_matrix_game(
                name=name, actions=actions,
                payoff_matrix=matrix, register=True,
            )
            return f"dynamic_{name}"
        except (ValueError, KeyError, IndexError):
            return None

    def _validate(self, prop: ArenaProposal, active_models: list[str]) -> bool:
        if prop.proposal_type == PROPOSAL_BAN:
            return prop.target_model is not None
        if prop.proposal_type == PROPOSAL_ADD:
            return prop.target_model is not None
        if prop.proposal_type == PROPOSAL_RULE:
            return prop.rule_description is not None
        if prop.proposal_type == PROPOSAL_NEW_GAME:
            return prop.game_definition is not None
        return False

    def _threshold_for(self, prop: ArenaProposal, total_weight: float) -> float:
        if prop.proposal_type == PROPOSAL_BAN:
            return total_weight * BAN_THRESHOLD_NUMERATOR / BAN_THRESHOLD_DENOMINATOR
        return total_weight * RULE_THRESHOLD_NUMERATOR / RULE_THRESHOLD_DENOMINATOR
