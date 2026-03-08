"""KantBench environment adapter for the HF Space.

Thin wrapper that delegates to the real KantEnvironment (90+ games,
17 strategies) instead of a standalone reimplementation.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import KantBenchAction, KantBenchObservation
from env.environment import KantEnvironment
from env.models import GameAction


class KantbenchEnvironment(Environment):
    """Game theory environment exposing 90+ games via the OpenEnv interface.

    Wraps the real KantEnvironment and translates between the Space's
    model types (KantBenchAction/Observation) and the internal types.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._env = KantEnvironment()

    def reset(self, **kwargs: Any) -> KantBenchObservation:
        obs = self._env.reset(**kwargs)
        return _to_space_obs(obs)

    def step(self, action: KantBenchAction, **kwargs: Any) -> KantBenchObservation:
        internal_action = GameAction(action=action.move)
        obs = self._env.step(internal_action, **kwargs)
        return _to_space_obs(obs)

    @property
    def state(self) -> State:
        s = self._env.state
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
