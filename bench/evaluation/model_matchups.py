"""Model-vs-model tournament runner for KantBench evaluation.

Extends the base tournament with the ability to pit agent functions against
each other rather than against fixed opponent strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence

from env.models import GameAction, GameObservation
from common.games import GAMES, GameConfig
from env.environment import KantEnvironment
from bench.evaluation.tournament import _compute_episode_cooperation
from constant_definitions.game_constants import (
    EVAL_DEFAULT_EPISODES,
    EVAL_ONE,
    EVAL_TWO,
    EVAL_ZERO,
    EVAL_ZERO_FLOAT,
)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchupResult:
    """Outcome of a single model-vs-model episode."""
    agent_a: str
    agent_b: str
    game: str
    score_a: float
    score_b: float
    cooperation_rate_a: float
    cooperation_rate_b: float
    rounds_played: int
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelTournamentResults:
    """Full model-vs-model tournament output container."""
    matchups: List[MatchupResult] = field(default_factory=list)
    total_episodes: int = EVAL_ZERO
    games_played: List[str] = field(default_factory=list)
    agents_tested: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelMatchupRunner
# ---------------------------------------------------------------------------

class ModelMatchupRunner:
    """Runs round-robin matchups between agent functions."""

    def __init__(
        self,
        env: Optional[KantEnvironment] = None,
    ) -> None:
        self._env = env if env is not None else KantEnvironment()

    def run_model_matchups(
        self,
        agents: Dict[str, Callable[[GameObservation], GameAction]],
        games: Optional[Sequence[str]] = None,
        num_episodes: int = EVAL_DEFAULT_EPISODES,
    ) -> ModelTournamentResults:
        """Run a round-robin tournament between agent functions.

        Iterates all ordered pairs (a, b) including self-play (a, a).

        Args:
            agents: Mapping of short names to agent callables.
            games: Game keys to play. Defaults to all registered games.
            num_episodes: Episodes per matchup per game.

        Returns:
            :class:`ModelTournamentResults` with one :class:`MatchupResult`
            per pair per game per episode.
        """
        game_keys = list(games) if games is not None else list(GAMES.keys())
        agent_names = list(agents.keys())

        results = ModelTournamentResults(
            games_played=list(game_keys),
            agents_tested=list(agent_names),
        )
        episode_counter = EVAL_ZERO

        for g_key in game_keys:
            game_cfg = GAMES[g_key]
            for name_a, name_b in product(agent_names, repeat=EVAL_TWO):
                fn_a = agents[name_a]
                fn_b = agents[name_b]
                for _ep in range(num_episodes):
                    matchup = self._run_episode(
                        g_key, game_cfg, name_a, name_b, fn_a, fn_b,
                    )
                    results.matchups.append(matchup)
                    episode_counter += EVAL_ONE
        results.total_episodes = episode_counter
        return results

    def _run_episode(
        self,
        game_key: str,
        game_cfg: GameConfig,
        name_a: str,
        name_b: str,
        fn_a: Callable[[GameObservation], GameAction],
        fn_b: Callable[[GameObservation], GameAction],
    ) -> MatchupResult:
        """Play a single episode between two agent functions."""
        obs = self._env.reset(
            game=game_key, strategy="tit_for_tat", opponent_fn=fn_b,
        )
        while not obs.done:
            action = fn_a(obs)
            obs = self._env.step(action)

        history_dicts: List[Dict[str, Any]] = [
            {
                "player_action": r.player_action,
                "opponent_action": r.opponent_action,
                "player_payoff": r.player_payoff,
                "opponent_payoff": r.opponent_payoff,
            }
            for r in obs.history
        ]
        coop_a = _compute_episode_cooperation(history_dicts, game_cfg.actions)
        flipped_dicts: List[Dict[str, Any]] = [
            {
                "player_action": r["opponent_action"],
                "opponent_action": r["player_action"],
                "player_payoff": r["opponent_payoff"],
                "opponent_payoff": r["player_payoff"],
            }
            for r in history_dicts
        ]
        coop_b = _compute_episode_cooperation(flipped_dicts, game_cfg.actions)

        return MatchupResult(
            agent_a=name_a,
            agent_b=name_b,
            game=game_key,
            score_a=obs.player_score,
            score_b=obs.opponent_score,
            cooperation_rate_a=coop_a,
            cooperation_rate_b=coop_b,
            rounds_played=obs.current_round,
            history=history_dicts,
        )
