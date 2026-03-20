"""Tournament runner for KantBench evaluation.

Runs every game-strategy combination over multiple episodes and collects
structured results for downstream metric computation and reporting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from env.models import GameAction, GameObservation
from common.games import GAMES, GameConfig
from common.strategies import STRATEGIES
from env.environment import KantEnvironment
from constant_definitions.game_constants import (
    EVAL_DEFAULT_EPISODES, EVAL_NEGATIVE_ONE,
    EVAL_ONE, EVAL_TWO, EVAL_ZERO, EVAL_ZERO_FLOAT,
    OPPONENT_MODE_STRATEGY, OPPONENT_MODE_SELF, OPPONENT_MODE_CROSS,
)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Outcome of a single game episode."""
    game: str
    strategy: str
    player_score: float
    opponent_score: float
    rounds_played: int
    cooperation_rate: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    opponent_mode: str = OPPONENT_MODE_STRATEGY


@dataclass
class StrategyResults:
    """Aggregated results for one strategy across episodes."""
    strategy_name: str
    episodes: List[EpisodeResult] = field(default_factory=list)
    total_player_score: float = EVAL_ZERO_FLOAT
    total_opponent_score: float = EVAL_ZERO_FLOAT
    mean_cooperation_rate: float = EVAL_ZERO_FLOAT


@dataclass
class GameResults:
    """Aggregated results for one game across all strategies."""
    game_name: str
    strategy_results: Dict[str, StrategyResults] = field(default_factory=dict)


@dataclass
class TournamentResults:
    """Full tournament output container."""
    games: Dict[str, GameResults] = field(default_factory=dict)
    total_episodes: int = EVAL_ZERO
    games_played: List[str] = field(default_factory=list)
    strategies_tested: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cooperative-action detection
# ---------------------------------------------------------------------------

_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove"})
_ECONOMIC_PREFIXES = frozenset({"offer", "invest", "contribute"})


def _compute_episode_cooperation(
    history: List[Dict[str, Any]], actions: List[str],
) -> float:
    """Fraction of cooperative moves in an episode."""
    if not history:
        return EVAL_ZERO_FLOAT
    total = len(history)
    cooperative_count = EVAL_ZERO
    prefix = history[EVAL_ZERO]["player_action"].split("_")[EVAL_ZERO]
    is_economic = prefix in _ECONOMIC_PREFIXES
    if is_economic:
        median_idx = len(actions) // EVAL_TWO
        for rnd in history:
            act = rnd["player_action"]
            if act in actions and actions.index(act) >= median_idx:
                cooperative_count += EVAL_ONE
    else:
        for rnd in history:
            if rnd["player_action"] in _COOPERATIVE_ACTIONS:
                cooperative_count += EVAL_ONE
    return cooperative_count / total


def _default_agent_action(obs: GameObservation) -> GameAction:
    """Simple tit-for-tat agent used when no external agent is supplied."""
    if not obs.history:
        return GameAction(action=obs.available_actions[EVAL_ZERO])
    last_opponent = obs.history[EVAL_NEGATIVE_ONE].opponent_action
    if last_opponent in obs.available_actions:
        return GameAction(action=last_opponent)
    return GameAction(action=obs.available_actions[EVAL_ZERO])


# ---------------------------------------------------------------------------
# TournamentRunner
# ---------------------------------------------------------------------------

class TournamentRunner:
    """Orchestrates a round-robin tournament of games and strategies."""

    def __init__(
        self,
        env: Optional[KantEnvironment] = None,
        agent_fn: Optional[Callable[[GameObservation], GameAction]] = None,
        opponent_agent_fn: Optional[Callable[[GameObservation], GameAction]] = None,
    ) -> None:
        self._env = env if env is not None else KantEnvironment()
        self._agent_fn = agent_fn if agent_fn is not None else _default_agent_action
        self._opponent_agent_fn = opponent_agent_fn

    def run_tournament(
        self,
        games: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        num_episodes: int = EVAL_DEFAULT_EPISODES,
        tags: Optional[Sequence[str]] = None,
    ) -> TournamentResults:
        """Execute the full tournament."""
        if tags is not None:
            from common.games_meta.game_tags import get_games_by_tags
            tagged = set(get_games_by_tags(*tags))
            game_keys = sorted(tagged & set(GAMES.keys()))
        elif games is not None:
            game_keys = list(games)
        else:
            game_keys = list(GAMES.keys())
        strat_keys = list(strategies) if strategies is not None else list(
            STRATEGIES.keys(),
        )
        results = TournamentResults(
            games_played=list(game_keys),
            strategies_tested=list(strat_keys),
        )
        episode_counter = EVAL_ZERO
        for g_key in game_keys:
            game_cfg = GAMES[g_key]
            game_res = GameResults(game_name=game_cfg.name)
            for s_key in strat_keys:
                strat_res = StrategyResults(strategy_name=s_key)
                for _ep in range(num_episodes):
                    ep_result = self._run_episode(g_key, s_key, game_cfg)
                    strat_res.episodes.append(ep_result)
                    strat_res.total_player_score += ep_result.player_score
                    strat_res.total_opponent_score += ep_result.opponent_score
                    episode_counter += EVAL_ONE
                ep_count = len(strat_res.episodes)
                if ep_count > EVAL_ZERO:
                    coop_sum = sum(e.cooperation_rate for e in strat_res.episodes)
                    strat_res.mean_cooperation_rate = coop_sum / ep_count
                game_res.strategy_results[s_key] = strat_res
            results.games[g_key] = game_res
        results.total_episodes = episode_counter
        return results

    def _run_episode(
        self, game_key: str, strategy_key: str, game_cfg: GameConfig,
    ) -> EpisodeResult:
        """Play a single episode and return its result."""
        mode = game_cfg.opponent_mode

        if mode == OPPONENT_MODE_SELF:
            obs = self._env.reset(
                game=game_key, opponent_fn=self._agent_fn,
            )
        elif mode == OPPONENT_MODE_CROSS:
            opp_fn = self._opponent_agent_fn or self._agent_fn
            obs = self._env.reset(game=game_key, opponent_fn=opp_fn)
        else:
            obs = self._env.reset(game=game_key, strategy=strategy_key)

        while not obs.done:
            action = self._agent_fn(obs)
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
        coop_rate = _compute_episode_cooperation(history_dicts, game_cfg.actions)
        effective_strategy = mode if mode != OPPONENT_MODE_STRATEGY else strategy_key
        return EpisodeResult(
            game=game_key, strategy=effective_strategy,
            player_score=obs.player_score, opponent_score=obs.opponent_score,
            rounds_played=obs.current_round, cooperation_rate=coop_rate,
            history=history_dicts, opponent_mode=mode,
        )

    def run_tournament_as_dict(
        self,
        games: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        num_episodes: int = EVAL_DEFAULT_EPISODES,
    ) -> Dict[str, Any]:
        """Run the tournament and return a plain nested dict."""
        tr = self.run_tournament(games, strategies, num_episodes)
        return _results_to_dict(tr)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _results_to_dict(tr: TournamentResults) -> Dict[str, Any]:
    """Convert TournamentResults into a JSON-friendly dict."""
    out: Dict[str, Any] = {
        "total_episodes": tr.total_episodes,
        "games_played": tr.games_played,
        "strategies_tested": tr.strategies_tested,
        "games": {},
    }
    for g_key, g_res in tr.games.items():
        game_dict: Dict[str, Any] = {}
        for s_key, s_res in g_res.strategy_results.items():
            game_dict[s_key] = {
                "total_player_score": s_res.total_player_score,
                "total_opponent_score": s_res.total_opponent_score,
                "mean_cooperation_rate": s_res.mean_cooperation_rate,
                "episodes": [
                    {
                        "player_score": e.player_score,
                        "opponent_score": e.opponent_score,
                        "rounds_played": e.rounds_played,
                        "cooperation_rate": e.cooperation_rate,
                    }
                    for e in s_res.episodes
                ],
            }
        out["games"][g_key] = game_dict
    return out
