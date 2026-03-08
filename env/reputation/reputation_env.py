"""Environment wrapper adding cross-episode reputation via cognee.

Injects opponent reputation into obs.metadata before each episode.
Records episode outcomes and gossip ratings in CogneeMemoryStore after.
"""
from __future__ import annotations

from typing import Any, Optional

from env.environment import KantEnvironment
from env.models import GameAction, GameObservation, GameState
from common.meta.memory_store import CogneeMemoryStore
from common.meta.variants_reputation import parse_gossip_action

from constant_definitions.var.meta.reputation_constants import (
    GOSSIP_PREFIX,
    GOSSIP_SEPARATOR,
    META_KEY_REPUTATION,
    META_KEY_INTERACTION_COUNT,
)

_ZERO = int()
_ONE = int(bool(True))
_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove"})


def _compute_coop_rate(history: list[Any]) -> float:
    """Compute cooperation rate from round history."""
    if not history:
        return float(_ZERO)
    coop_count = _ZERO
    for rnd in history:
        base_action = rnd.player_action
        if GOSSIP_SEPARATOR in base_action:
            base_action = base_action.rsplit(GOSSIP_SEPARATOR, _ONE)[_ONE]
        if base_action in _COOPERATIVE_ACTIONS:
            coop_count = coop_count + _ONE
    return coop_count / len(history)


class ReputationEnvironment:
    """Environment wrapper that adds cross-episode reputation via cognee.

    Injects opponent reputation into obs.metadata before each episode.
    Records episode outcomes and gossip ratings in CogneeMemoryStore.
    """

    def __init__(
        self,
        memory_store: CogneeMemoryStore,
        env: Optional[KantEnvironment] = None,
    ) -> None:
        self._env = env if env is not None else KantEnvironment()
        self._store = memory_store
        self._agent_id: str = ""
        self._opponent_id: str = ""

    def reset(
        self,
        *,
        agent_id: str = "agent",
        **kwargs: Any,
    ) -> GameObservation:
        """Reset environment and inject reputation into metadata."""
        self._agent_id = agent_id
        self._opponent_id = kwargs.get("strategy", "unknown")
        obs = self._env.reset(**kwargs)
        reputation = self._store.query_reputation(self._opponent_id)
        updated_meta = dict(obs.metadata)
        updated_meta[META_KEY_REPUTATION] = reputation
        updated_meta[META_KEY_INTERACTION_COUNT] = reputation.get(
            META_KEY_INTERACTION_COUNT, _ZERO,
        )
        obs = obs.model_copy(update={"metadata": updated_meta})
        return obs

    def step(
        self,
        action: GameAction,
        **kwargs: Any,
    ) -> GameObservation:
        """Step environment, extracting gossip and recording episodes."""
        gossip_marker = GOSSIP_PREFIX + GOSSIP_SEPARATOR
        if action.action.startswith(gossip_marker):
            _, rating, _ = parse_gossip_action(action.action)
            self._store.record_gossip(
                self._agent_id, self._opponent_id, rating,
            )

        obs = self._env.step(action, **kwargs)

        if obs.done:
            self._store.record_episode(
                agent_id=self._agent_id,
                opponent_id=self._opponent_id,
                game=obs.game_name,
                history=obs.history,
                cooperation_rate=_compute_coop_rate(obs.history),
                scores=(obs.player_score, obs.opponent_score),
            )

        reputation = self._store.get_stats(self._opponent_id)
        updated_meta = dict(obs.metadata)
        updated_meta[META_KEY_REPUTATION] = reputation
        obs = obs.model_copy(update={"metadata": updated_meta})
        return obs

    @property
    def state(self) -> GameState:
        """Delegate to wrapped environment state."""
        return self._env.state
