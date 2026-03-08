"""KantBench environment adapter for the HF Space.

Thin wrapper that delegates to the real KantEnvironment (90+ 2-player games,
17 strategies) and NPlayerEnvironment (3 N-player games) instead of a
standalone reimplementation.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import KantBenchAction, KantBenchObservation
from env.environment import KantEnvironment
from env.models import GameAction
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction, NPlayerObservation

# Register built-in N-player games into the registry
import common.games_meta.nplayer_games  # noqa: F401
from common.games_meta.nplayer_config import NPLAYER_GAMES


class KantbenchEnvironment(Environment):
    """Game theory environment exposing 90+ two-player and N-player games.

    Wraps the real KantEnvironment and NPlayerEnvironment, routing
    automatically based on the requested game name.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._env_2p = KantEnvironment()
        self._env_np = NPlayerEnvironment()
        self._is_nplayer: bool = False

    def reset(self, **kwargs: Any) -> KantBenchObservation:
        game_name: str = kwargs.get("game", "prisoners_dilemma")

        if game_name in NPLAYER_GAMES:
            self._is_nplayer = True
            # Map Space kwargs to NPlayerEnvironment.reset signature
            opponent_strategies: Optional[list[str]] = None
            strategy = kwargs.get("strategy")
            if strategy:
                opponent_strategies = [strategy]
            obs = self._env_np.reset(
                game_name,
                num_rounds=kwargs.get("num_rounds"),
                opponent_strategies=opponent_strategies,
            )
            return _nplayer_to_space_obs(obs)
        else:
            self._is_nplayer = False
            obs = self._env_2p.reset(**kwargs)
            return _to_space_obs(obs)

    def step(self, action: KantBenchAction, **kwargs: Any) -> KantBenchObservation:
        if self._is_nplayer:
            internal_action = NPlayerAction(action=action.move)
            obs = self._env_np.step(internal_action)
            return _nplayer_to_space_obs(obs)
        else:
            internal_action = GameAction(action=action.move)
            obs = self._env_2p.step(internal_action, **kwargs)
            return _to_space_obs(obs)

    @property
    def state(self) -> State:
        if self._is_nplayer:
            s = self._env_np.state
        else:
            s = self._env_2p.state
        return State(
            episode_id=s.episode_id or "",
            step_count=s.step_count,
        )


def _to_space_obs(obs) -> KantBenchObservation:
    """Convert internal GameObservation to Space-facing KantBenchObservation."""
    last = obs.last_round
    history = [
        {
            "round": r.round_number,
            "your_move": r.player_action,
            "opponent_move": r.opponent_action,
            "your_payoff": r.player_payoff,
            "opponent_payoff": r.opponent_payoff,
        }
        for r in obs.history
    ]
    return KantBenchObservation(
        game_name=obs.game_name,
        game_description=obs.game_description,
        available_moves=list(obs.available_actions),
        your_move=last.player_action if last else "",
        opponent_move=last.opponent_action if last else "",
        your_payoff=last.player_payoff if last else 0.0,
        opponent_payoff=last.opponent_payoff if last else 0.0,
        cumulative_score=obs.player_score,
        round_number=obs.current_round,
        max_rounds=obs.total_rounds,
        opponent_strategy=obs.opponent_strategy,
        history=history,
        done=obs.done,
        reward=obs.reward,
        message="Game over — call reset() to start a new episode." if obs.done else "",
    )


def _nplayer_to_space_obs(obs: NPlayerObservation) -> KantBenchObservation:
    """Convert NPlayerObservation to Space-facing KantBenchObservation."""
    last = obs.last_round
    history = [
        {
            "round": r.round_number,
            "actions": r.actions,
            "payoffs": r.payoffs,
        }
        for r in obs.history
    ]
    return KantBenchObservation(
        game_name=obs.game_name,
        game_description=obs.game_description,
        available_moves=list(obs.available_actions),
        your_move=last.actions[0] if last else "",
        opponent_move="",  # N-player: see history for all actions
        your_payoff=last.payoffs[0] if last else 0.0,
        opponent_payoff=0.0,  # N-player: see history for all payoffs
        cumulative_score=obs.scores[0] if obs.scores else 0.0,
        round_number=obs.current_round,
        max_rounds=obs.total_rounds,
        opponent_strategy="",
        history=history,
        done=obs.done,
        reward=obs.reward,
        message="Game over — call reset() to start a new episode." if obs.done else "",
        num_players=obs.num_players,
        player_index=obs.player_index,
        all_scores=list(obs.scores),
    )
