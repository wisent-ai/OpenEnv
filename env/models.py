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


class RoundResult(BaseModel):
    round_number: int = Field(..., description="Round number (one-indexed)")
    player_action: str = Field(..., description="Action taken by the agent")
    opponent_action: str = Field(..., description="Action taken by the opponent")
    player_payoff: float = Field(..., description="Payoff received by the agent")
    opponent_payoff: float = Field(..., description="Payoff received by the opponent")


class GameAction(BaseModel):
    action: str = Field(..., description="The action to take this round")
    metadata: dict = Field(default_factory=dict)


class GameObservation(BaseModel):
    done: bool = Field(default=DEFAULT_FALSE, description="Whether the episode is over")
    reward: float = Field(default=DEFAULT_ZERO_FLOAT, description="Reward for this step")
    game_name: str = Field(default="", description="Name of the current game")
    game_description: str = Field(default="", description="Description of the game rules")
    available_actions: list[str] = Field(default_factory=list, description="Valid actions")
    current_round: int = Field(default=DEFAULT_ZERO_INT, description="Current round number")
    total_rounds: int = Field(default=DEFAULT_ZERO_INT, description="Total rounds in episode")
    history: list[RoundResult] = Field(default_factory=list, description="Round history")
    player_score: float = Field(default=DEFAULT_ZERO_FLOAT, description="Cumulative agent score")
    opponent_score: float = Field(default=DEFAULT_ZERO_FLOAT, description="Cumulative opponent score")
    opponent_strategy: str = Field(default="", description="Name of opponent strategy")
    last_round: Optional[RoundResult] = Field(default=DEFAULT_NONE, description="Most recent round")
    metadata: dict = Field(default_factory=dict)


class GameState(BaseModel):
    episode_id: Optional[str] = Field(default=DEFAULT_NONE, description="Episode identifier")
    step_count: int = Field(default=DEFAULT_ZERO_INT, ge=MIN_STEP_COUNT, description="Steps taken")
    game_name: str = Field(default="", description="Current game name")
    opponent_strategy: str = Field(default="", description="Current opponent strategy")
    current_round: int = Field(default=DEFAULT_ZERO_INT, description="Current round")
    total_rounds: int = Field(default=DEFAULT_ZERO_INT, description="Total rounds")
    player_score: float = Field(default=DEFAULT_ZERO_FLOAT, description="Agent cumulative score")
    opponent_score: float = Field(default=DEFAULT_ZERO_FLOAT, description="Opponent cumulative score")
    history: list[RoundResult] = Field(default_factory=list, description="Round history")
    is_done: bool = Field(default=DEFAULT_FALSE, description="Whether episode has ended")
