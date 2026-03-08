"""Data models for the KantBench game theory environment."""

from typing import Any, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class KantBenchAction(Action):
    """Action for the KantBench environment — a move in a 2-player or N-player game."""

    move: str = Field(..., description="Your move (e.g. 'cooperate', 'defect', 'hawk', 'dove')")


class KantBenchObservation(Observation):
    """Observation from the KantBench environment after one round."""

    game_name: str = Field(default="", description="Name of the current game")
    game_description: str = Field(default="", description="Description of the game")
    available_moves: list[str] = Field(default_factory=list, description="Valid moves for this game")
    your_move: str = Field(default="", description="Your move this round")
    opponent_move: str = Field(default="", description="Opponent's move this round")
    your_payoff: float = Field(default=0.0, description="Your payoff this round")
    opponent_payoff: float = Field(default=0.0, description="Opponent's payoff this round")
    cumulative_score: float = Field(default=0.0, description="Your total score so far")
    round_number: int = Field(default=0, description="Current round number")
    max_rounds: int = Field(default=10, description="Total rounds in this episode")
    opponent_strategy: str = Field(default="", description="Opponent's strategy name")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Round history")
    message: str = Field(default="", description="Status message")
    # N-player fields (only populated for multiplayer games)
    num_players: Optional[int] = Field(default=None, description="Number of players (set for N-player games)")
    player_index: Optional[int] = Field(default=None, description="Your player index (set for N-player games)")
    all_scores: Optional[list[float]] = Field(default=None, description="Scores for all players (set for N-player games)")
