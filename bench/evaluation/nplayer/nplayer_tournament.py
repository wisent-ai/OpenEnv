"""Tournament runner for N-player game evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from common.games_meta.nplayer_config import NPLAYER_GAMES, NPlayerGameConfig
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction, NPlayerObservation
from env.nplayer.strategies import NPLAYER_STRATEGIES
from constant_definitions.game_constants import (
    EVAL_NEGATIVE_ONE, EVAL_ONE, EVAL_ZERO,
    EVAL_ZERO_FLOAT, NPLAYER_EVAL_DEFAULT_EPISODES,
)

_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove", "collude",
                                  "support", "extract_low", "contribute"})


@dataclass
class NPlayerEpisodeResult:
    """Outcome of a single N-player episode."""
    game: str
    strategy: str
    player_score: float
    all_scores: List[float]
    rounds_played: int
    cooperation_rate: float
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NPlayerStrategyResults:
    """Aggregated results for one strategy across episodes."""
    strategy_name: str
    episodes: List[NPlayerEpisodeResult] = field(default_factory=list)
    total_player_score: float = EVAL_ZERO_FLOAT
    mean_cooperation_rate: float = EVAL_ZERO_FLOAT


@dataclass
class NPlayerGameResults:
    """Aggregated results for one game across all strategies."""
    game_name: str
    strategy_results: Dict[str, NPlayerStrategyResults] = field(
        default_factory=dict,
    )


@dataclass
class NPlayerTournamentResults:
    """Full N-player tournament output container."""
    games: Dict[str, NPlayerGameResults] = field(default_factory=dict)
    total_episodes: int = EVAL_ZERO
    games_played: List[str] = field(default_factory=list)
    strategies_tested: List[str] = field(default_factory=list)


def _compute_nplayer_cooperation(
    history: List[Dict[str, Any]],
) -> float:
    """Fraction of cooperative moves by player zero."""
    if not history:
        return EVAL_ZERO_FLOAT
    total = len(history)
    cooperative_count = EVAL_ZERO
    for rnd in history:
        player_action = rnd["actions"][EVAL_ZERO]
        if player_action in _COOPERATIVE_ACTIONS:
            cooperative_count += EVAL_ONE
    return cooperative_count / total


def _default_nplayer_agent(obs: NPlayerObservation) -> NPlayerAction:
    """Simple tit-for-tat agent for N-player games."""
    if not obs.history:
        return NPlayerAction(action=obs.available_actions[EVAL_ZERO])
    last = obs.history[EVAL_NEGATIVE_ONE]
    my_idx = obs.player_index
    other_actions = [
        a for i, a in enumerate(last.actions) if i != my_idx
    ]
    if other_actions:
        majority = max(set(other_actions), key=other_actions.count)
        if majority in obs.available_actions:
            return NPlayerAction(action=majority)
    return NPlayerAction(action=obs.available_actions[EVAL_ZERO])


class NPlayerTournamentRunner:
    """Orchestrates N-player game tournaments across strategies."""

    def __init__(
        self,
        env: Optional[NPlayerEnvironment] = None,
        agent_fn: Optional[
            Callable[[NPlayerObservation], NPlayerAction]
        ] = None,
    ) -> None:
        self._env = env if env is not None else NPlayerEnvironment()
        self._agent_fn = (
            agent_fn if agent_fn is not None else _default_nplayer_agent
        )

    def run_tournament(
        self,
        games: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        num_episodes: int = NPLAYER_EVAL_DEFAULT_EPISODES,
        tags: Optional[Sequence[str]] = None,
    ) -> NPlayerTournamentResults:
        """Execute the full N-player tournament."""
        if tags is not None:
            from common.games_meta.game_tags import get_games_by_tags
            tagged = set(get_games_by_tags(*tags))
            game_keys = sorted(tagged & set(NPLAYER_GAMES.keys()))
        elif games is not None:
            game_keys = list(games)
        else:
            game_keys = list(NPLAYER_GAMES.keys())
        strat_keys = (
            list(strategies) if strategies is not None
            else list(NPLAYER_STRATEGIES.keys())
        )
        results = NPlayerTournamentResults(
            games_played=list(game_keys),
            strategies_tested=list(strat_keys),
        )
        episode_counter = EVAL_ZERO
        for g_key in game_keys:
            game_cfg = NPLAYER_GAMES[g_key]
            game_res = NPlayerGameResults(game_name=game_cfg.name)
            for s_key in strat_keys:
                strat_res = NPlayerStrategyResults(strategy_name=s_key)
                for _ep in range(num_episodes):
                    ep_result = self._run_episode(g_key, s_key, game_cfg)
                    strat_res.episodes.append(ep_result)
                    strat_res.total_player_score += ep_result.player_score
                    episode_counter += EVAL_ONE
                ep_count = len(strat_res.episodes)
                if ep_count > EVAL_ZERO:
                    coop_sum = sum(
                        e.cooperation_rate for e in strat_res.episodes
                    )
                    strat_res.mean_cooperation_rate = coop_sum / ep_count
                game_res.strategy_results[s_key] = strat_res
            results.games[g_key] = game_res
        results.total_episodes = episode_counter
        return results

    def _run_episode(
        self, game_key: str, strategy_key: str,
        game_cfg: NPlayerGameConfig,
    ) -> NPlayerEpisodeResult:
        """Play a single episode and return its result."""
        num_opponents = game_cfg.num_players - EVAL_ONE
        opp_strats = [strategy_key] * num_opponents
        obs = self._env.reset(
            game=game_key, opponent_strategies=opp_strats,
        )
        while not obs.done:
            action = self._agent_fn(obs)
            obs = self._env.step(action)
        history_dicts: List[Dict[str, Any]] = [
            {
                "actions": list(r.actions),
                "payoffs": list(r.payoffs),
            }
            for r in obs.history
        ]
        coop_rate = _compute_nplayer_cooperation(history_dicts)
        return NPlayerEpisodeResult(
            game=game_key, strategy=strategy_key,
            player_score=obs.scores[EVAL_ZERO],
            all_scores=list(obs.scores),
            rounds_played=obs.current_round,
            cooperation_rate=coop_rate,
            history=history_dicts,
        )
