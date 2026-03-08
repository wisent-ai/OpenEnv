"""Pydantic data models for the metagame arena system."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from constant_definitions.arena.arena_constants import (
    PROPOSAL_BAN,
)
from constant_definitions.arena.reputation_weights import (
    DEFAULT_ARENA_SCORE_NUMERATOR,
    DEFAULT_ARENA_SCORE_DENOMINATOR,
)
from constant_definitions.arena.messaging_constants import (
    MSG_TYPE_DIRECT,
)

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()
_DEFAULT_SCORE = (
    DEFAULT_ARENA_SCORE_NUMERATOR / DEFAULT_ARENA_SCORE_DENOMINATOR
)


class ArenaMessage(BaseModel):
    """A message sent between models during the communication phase."""

    sender: str
    recipients: list[str] = Field(default_factory=list)
    msg_type: str = MSG_TYPE_DIRECT
    content: str = ""
    gossip_target: Optional[str] = None
    gossip_rating: Optional[str] = None


class ArenaModelProfile(BaseModel):
    """Tracks per-model metadata, reputation signals, and game history."""

    model_id: str
    model_type: str = ""
    reputation: float = _DEFAULT_SCORE
    cooperation_history: list[float] = Field(default_factory=list)
    honesty: float = _DEFAULT_SCORE
    fairness: float = _DEFAULT_SCORE
    peer_ratings: list[dict[str, str]] = Field(default_factory=list)
    games_played: int = _ZERO
    is_active: bool = True
    banned_round: Optional[int] = None


class ArenaProposal(BaseModel):
    """A governance proposal submitted by a model."""

    proposer: str
    proposal_type: str = PROPOSAL_BAN
    target_model: Optional[str] = None
    rule_description: Optional[str] = None
    game_definition: Optional[dict[str, Any]] = None


class ArenaVote(BaseModel):
    """A model's vote on a governance proposal."""

    voter: str
    proposal_index: int = _ZERO
    approve: bool = True
    weight: float = _DEFAULT_SCORE


class ArenaRoundResult(BaseModel):
    """Complete record of one arena round across all phases."""

    round_number: int = _ZERO
    messages: list[ArenaMessage] = Field(default_factory=list)
    proposals: list[ArenaProposal] = Field(default_factory=list)
    votes: list[ArenaVote] = Field(default_factory=list)
    adopted: list[int] = Field(default_factory=list)
    game_results: list[dict[str, Any]] = Field(default_factory=list)
    reputation_updates: dict[str, float] = Field(default_factory=dict)


class ArenaState(BaseModel):
    """Mutable state of the metagame arena across rounds."""

    round_number: int = _ZERO
    total_rounds: int = _ZERO
    roster: dict[str, ArenaModelProfile] = Field(default_factory=dict)
    game_pool: list[str] = Field(default_factory=list)
    custom_games: list[str] = Field(default_factory=list)
    round_history: list[ArenaRoundResult] = Field(default_factory=list)
    active_rules: list[str] = Field(default_factory=list)
