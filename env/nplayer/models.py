"""Data models for the N-player environment."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from constant_definitions.game_constants import (
    DEFAULT_FALSE,
    DEFAULT_NONE,
    DEFAULT_ZERO_FLOAT,
    DEFAULT_ZERO_INT,
    MIN_STEP_COUNT,
)


class NPlayerRoundResult(BaseModel):
    round_number: int = Field(..., description="Round number (one-indexed)")
    actions: list[str] = Field(..., description="Actions taken by all players")
    payoffs: list[float] = Field(..., description="Payoffs received by all players")


class NPlayerAction(BaseModel):
    action: str = Field(..., description="The action to take this round")
    metadata: dict = Field(default_factory=dict)


class NPlayerObservation(BaseModel):
    done: bool = Field(default=DEFAULT_FALSE, description="Whether the episode is over")
    reward: float = Field(default=DEFAULT_ZERO_FLOAT, description="Reward for this step")
    game_name: str = Field(default="", description="Name of the current game")
    game_description: str = Field(default="", description="Description of the game rules")
    available_actions: list[str] = Field(default_factory=list, description="Valid actions")
    current_round: int = Field(default=DEFAULT_ZERO_INT, description="Current round number")
    total_rounds: int = Field(default=DEFAULT_ZERO_INT, description="Total rounds in episode")
    history: list[NPlayerRoundResult] = Field(default_factory=list, description="Round history")
    scores: list[float] = Field(default_factory=list, description="Cumulative scores for all players")
    num_players: int = Field(default=DEFAULT_ZERO_INT, description="Number of players")
    player_index: int = Field(default=DEFAULT_ZERO_INT, description="This player's index")
    last_round: Optional[NPlayerRoundResult] = Field(default=DEFAULT_NONE, description="Most recent round")
    metadata: dict = Field(default_factory=dict)


class NPlayerGameState(BaseModel):
    episode_id: Optional[str] = Field(default=DEFAULT_NONE, description="Episode identifier")
    step_count: int = Field(default=DEFAULT_ZERO_INT, ge=MIN_STEP_COUNT, description="Steps taken")
    game_name: str = Field(default="", description="Current game name")
    current_round: int = Field(default=DEFAULT_ZERO_INT, description="Current round")
    total_rounds: int = Field(default=DEFAULT_ZERO_INT, description="Total rounds")
    num_players: int = Field(default=DEFAULT_ZERO_INT, description="Number of players")
    scores: list[float] = Field(default_factory=list, description="Cumulative scores for all players")
    history: list[NPlayerRoundResult] = Field(default_factory=list, description="Round history")
    is_done: bool = Field(default=DEFAULT_FALSE, description="Whether episode has ended")
