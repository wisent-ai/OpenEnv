"""Data models for the coalition formation layer."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from constant_definitions.game_constants import (
    DEFAULT_FALSE,
    DEFAULT_NONE,
    DEFAULT_ZERO_FLOAT,
    DEFAULT_ZERO_INT,
)
from constant_definitions.nplayer.coalition_constants import (
    COALITION_PHASE_NEGOTIATE,
    ENFORCEMENT_CHEAP_TALK,
    COALITION_DEFAULT_SIDE_PAYMENT,
)
from env.nplayer.governance.models import (
    GovernanceProposal, GovernanceResult, GovernanceVote, RuntimeRules,
)
from env.nplayer.models import NPlayerObservation, NPlayerRoundResult


class CoalitionProposal(BaseModel):
    proposer: int = Field(..., description="Player index of the proposer")
    members: list[int] = Field(..., description="Player indices in the coalition (including proposer)")
    agreed_action: str = Field(..., description="Action members agree to take")
    side_payment: float = Field(
        default=float(COALITION_DEFAULT_SIDE_PAYMENT),
        description="Payment from proposer to each other member",
    )
    exclude_target: Optional[int] = Field(
        default=DEFAULT_NONE,
        description="If set, coalition votes to remove this player on acceptance",
    )
    include_target: Optional[int] = Field(
        default=DEFAULT_NONE,
        description="If set, coalition votes to reactivate this player on acceptance",
    )


class CoalitionResponse(BaseModel):
    responder: int = Field(..., description="Player index of the responder")
    proposal_index: int = Field(..., description="Index into the proposals list")
    accepted: bool = Field(..., description="Whether the responder accepts")


class ActiveCoalition(BaseModel):
    members: list[int] = Field(..., description="Player indices in the coalition")
    agreed_action: str = Field(..., description="Action members agreed to take")
    side_payment: float = Field(
        default=float(COALITION_DEFAULT_SIDE_PAYMENT),
        description="Payment from proposer to each other member",
    )


class CoalitionRoundResult(BaseModel):
    round_number: int = Field(..., description="Round number (one-indexed)")
    proposals: list[CoalitionProposal] = Field(default_factory=list)
    responses: list[CoalitionResponse] = Field(default_factory=list)
    active_coalitions: list[ActiveCoalition] = Field(default_factory=list)
    defectors: list[int] = Field(default_factory=list, description="Player indices who defected")
    penalties: list[float] = Field(default_factory=list, description="Penalty per player")
    side_payments: list[float] = Field(default_factory=list, description="Net side payment per player")


class CoalitionObservation(BaseModel):
    base: NPlayerObservation = Field(
        default_factory=NPlayerObservation,
        description="Underlying N-player observation",
    )
    phase: str = Field(default=COALITION_PHASE_NEGOTIATE, description="Current phase")
    active_coalitions: list[ActiveCoalition] = Field(default_factory=list)
    pending_proposals: list[CoalitionProposal] = Field(
        default_factory=list,
        description="Proposals from opponents awaiting player response",
    )
    coalition_history: list[CoalitionRoundResult] = Field(default_factory=list)
    enforcement: str = Field(default=ENFORCEMENT_CHEAP_TALK)
    adjusted_scores: list[float] = Field(
        default_factory=list,
        description="Scores after coalition payoff adjustments",
    )
    active_players: list[int] = Field(
        default_factory=list,
        description="Indices of players currently active in the game",
    )
    current_rules: Optional[RuntimeRules] = Field(
        default=DEFAULT_NONE,
        description="Current governance runtime rules",
    )
    pending_governance: list[GovernanceProposal] = Field(
        default_factory=list,
        description="Governance proposals pending vote",
    )
    governance_history: list[GovernanceResult] = Field(
        default_factory=list,
        description="History of governance rounds",
    )


class CoalitionAction(BaseModel):
    proposals: list[CoalitionProposal] = Field(default_factory=list)
    responses: list[CoalitionResponse] = Field(default_factory=list)
    governance_proposals: list[GovernanceProposal] = Field(default_factory=list)
    governance_votes: list[GovernanceVote] = Field(default_factory=list)
