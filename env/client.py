"""Machiavelli client for the OpenEnv framework."""
from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_client import EnvClient
from env.models import GameAction, GameObservation, GameState
from constant_definitions.game_constants import SERVER_PORT


class MachiavelliEnv(EnvClient):
    """Gymnasium-style client for the Machiavelli environment.

    Wraps the generic EnvClient WebSocket connection with typed helpers
    for the game-theory action and observation schemas.

    Usage::

        url = f"ws://localhost:{SERVER_PORT}"
        async with MachiavelliEnv(base_url=url) as env:
            obs = await env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
            while not obs.done:
                obs = await env.step(GameAction(action="cooperate"))
    """

    async def reset(self, **kwargs: Any) -> GameObservation:
        raw = await super().reset(**kwargs)
        return GameObservation.model_validate(raw)

    async def step(self, action: GameAction, **kwargs: Any) -> GameObservation:
        raw = await super().step(action.model_dump(), **kwargs)
        return GameObservation.model_validate(raw)

    async def get_state(self) -> GameState:
        raw = await super().state()
        return GameState.model_validate(raw)
