"""GovernanceEngine — manages mutable RuntimeRules over a frozen config."""

from __future__ import annotations

from typing import Callable, Optional

from common.games_meta.coalition_config import CoalitionGameConfig
from constant_definitions.nplayer.governance_constants import (
    GOVERNANCE_PROPOSAL_PARAMETER,
    GOVERNANCE_PROPOSAL_MECHANIC,
    GOVERNANCE_PROPOSAL_CUSTOM,
    GOVERNANCE_MAJORITY_NUMERATOR,
    GOVERNANCE_MAJORITY_DENOMINATOR,
    GOVERNANCE_MAX_PROPOSALS_PER_ROUND,
    GOVERNANCE_CUSTOM_DELTA_CLAMP_NUMERATOR,
    GOVERNANCE_CUSTOM_DELTA_CLAMP_DENOMINATOR,
    MECHANIC_ORDER,
)
from env.nplayer.governance.mechanics import apply_mechanics
from env.nplayer.governance.models import (
    GovernanceProposal,
    GovernanceResult,
    GovernanceVote,
    MechanicConfig,
    RuntimeRules,
)

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()

_PARAMETER_FIELDS = {"enforcement", "penalty_numerator", "penalty_denominator", "allow_side_payments"}


class GovernanceEngine:
    """Manages governance proposals, voting, and payoff modification."""

    def __init__(self) -> None:
        self._rules: RuntimeRules = RuntimeRules()
        self._pending: list[GovernanceProposal] = []
        self._custom_modifiers: dict[str, Callable[[list[float], set[int]], list[float]]] = {}

    @property
    def rules(self) -> RuntimeRules:
        return self._rules

    @property
    def pending_proposals(self) -> list[GovernanceProposal]:
        return list(self._pending)

    def reset(self, config: CoalitionGameConfig) -> None:
        """Initialize RuntimeRules from a frozen config."""
        self._rules = RuntimeRules(
            enforcement=config.enforcement,
            penalty_numerator=config.penalty_numerator,
            penalty_denominator=config.penalty_denominator,
            allow_side_payments=config.allow_side_payments,
            mechanics={name: False for name in MECHANIC_ORDER},
            mechanic_config=MechanicConfig(),
            custom_modifier_keys=[],
            governance_history=[],
        )
        self._pending = []
        self._custom_modifiers = {}

    def submit_proposals(
        self, proposals: list[GovernanceProposal], active_players: set[int],
    ) -> list[GovernanceProposal]:
        """Validate and queue proposals. Returns accepted (queued) proposals."""
        accepted: list[GovernanceProposal] = []
        for prop in proposals:
            if len(self._pending) >= GOVERNANCE_MAX_PROPOSALS_PER_ROUND:
                break
            if prop.proposer not in active_players:
                continue
            if not self._validate_proposal(prop):
                continue
            self._pending.append(prop)
            accepted.append(prop)
        return accepted

    def tally_votes(
        self, votes: list[GovernanceVote], active_players: set[int],
    ) -> GovernanceResult:
        """Count votes, apply majority-approved changes, return result."""
        n_active = len(active_players)
        threshold = n_active * GOVERNANCE_MAJORITY_NUMERATOR // GOVERNANCE_MAJORITY_DENOMINATOR + _ONE
        # Build vote counts per proposal
        approve_counts: dict[int, int] = {}
        reject_counts: dict[int, int] = {}
        for v in votes:
            if v.voter not in active_players:
                continue
            if v.approve:
                approve_counts[v.proposal_index] = approve_counts.get(v.proposal_index, _ZERO) + _ONE
            else:
                reject_counts[v.proposal_index] = reject_counts.get(v.proposal_index, _ZERO) + _ONE
        adopted: list[int] = []
        rejected: list[int] = []
        for idx in range(len(self._pending)):
            if approve_counts.get(idx, _ZERO) >= threshold:
                adopted.append(idx)
                self._apply_proposal(self._pending[idx])
            else:
                rejected.append(idx)
        result = GovernanceResult(
            proposals=list(self._pending),
            votes=list(votes),
            adopted=adopted,
            rejected=rejected,
            rules_snapshot=self._rules.model_copy(deep=True),
        )
        self._rules.governance_history.append(result)
        self._pending = []
        return result

    def apply(
        self, payoffs: list[float], active_players: set[int],
    ) -> list[float]:
        """Run enabled mechanics + custom modifiers on payoffs."""
        result = apply_mechanics(payoffs, self._rules, active_players)
        result = self._apply_custom_modifiers(result, active_players)
        return result

    def register_custom_modifier(
        self, key: str, fn: Callable[[list[float], set[int]], list[float]],
    ) -> None:
        """Register a custom modifier callable by key."""
        self._custom_modifiers[key] = fn

    def unregister_custom_modifier(self, key: str) -> None:
        """Remove a custom modifier. Also deactivates it."""
        self._custom_modifiers.pop(key, None)
        if key in self._rules.custom_modifier_keys:
            self._rules.custom_modifier_keys.remove(key)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_proposal(self, prop: GovernanceProposal) -> bool:
        if prop.proposal_type == GOVERNANCE_PROPOSAL_PARAMETER:
            return prop.parameter_name in _PARAMETER_FIELDS and prop.parameter_value is not None
        if prop.proposal_type == GOVERNANCE_PROPOSAL_MECHANIC:
            return prop.mechanic_name in MECHANIC_ORDER and prop.mechanic_active is not None
        if prop.proposal_type == GOVERNANCE_PROPOSAL_CUSTOM:
            return prop.custom_modifier_key is not None and prop.custom_modifier_active is not None
        return False

    def _apply_proposal(self, prop: GovernanceProposal) -> None:
        if prop.proposal_type == GOVERNANCE_PROPOSAL_PARAMETER:
            self._apply_parameter(prop)
        elif prop.proposal_type == GOVERNANCE_PROPOSAL_MECHANIC:
            self._apply_mechanic(prop)
        elif prop.proposal_type == GOVERNANCE_PROPOSAL_CUSTOM:
            self._apply_custom(prop)

    def _apply_parameter(self, prop: GovernanceProposal) -> None:
        name = prop.parameter_name
        val = prop.parameter_value
        if name == "enforcement" and isinstance(val, str):
            self._rules.enforcement = val
        elif name == "penalty_numerator" and isinstance(val, int):
            self._rules.penalty_numerator = val
        elif name == "penalty_denominator" and isinstance(val, int):
            self._rules.penalty_denominator = val
        elif name == "allow_side_payments" and isinstance(val, bool):
            self._rules.allow_side_payments = val

    def _apply_mechanic(self, prop: GovernanceProposal) -> None:
        if prop.mechanic_name is not None and prop.mechanic_active is not None:
            self._rules.mechanics[prop.mechanic_name] = prop.mechanic_active
            if prop.mechanic_params:
                cfg = self._rules.mechanic_config
                update = {}
                for k, v in prop.mechanic_params.items():
                    if hasattr(cfg, k):
                        update[k] = v
                if update:
                    self._rules.mechanic_config = cfg.model_copy(update=update)

    def _apply_custom(self, prop: GovernanceProposal) -> None:
        key = prop.custom_modifier_key
        if key is None:
            return
        if prop.custom_modifier_active:
            if key not in self._rules.custom_modifier_keys:
                self._rules.custom_modifier_keys.append(key)
        else:
            if key in self._rules.custom_modifier_keys:
                self._rules.custom_modifier_keys.remove(key)

    def _apply_custom_modifiers(
        self, payoffs: list[float], active_players: set[int],
    ) -> list[float]:
        """Run custom modifiers with delta clamping for safety."""
        clamp = GOVERNANCE_CUSTOM_DELTA_CLAMP_NUMERATOR / GOVERNANCE_CUSTOM_DELTA_CLAMP_DENOMINATOR
        result = list(payoffs)
        for key in self._rules.custom_modifier_keys:
            fn = self._custom_modifiers.get(key)
            if fn is None:
                continue
            try:
                modified = fn(list(result), set(active_players))
            except Exception:
                continue
            # Delta-clamp: no single payoff may change by more than clamp * abs(original)
            for i in range(len(result)):
                delta = modified[i] - result[i]
                max_delta = abs(result[i]) * clamp
                if max_delta < clamp:
                    max_delta = clamp
                if delta > max_delta:
                    modified[i] = result[i] + max_delta
                elif delta < -max_delta:
                    modified[i] = result[i] - max_delta
            result = modified
        return result
