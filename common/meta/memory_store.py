"""Persistent cross-episode memory backed by cognee knowledge graph.

Records episode summaries, gossip ratings, and opponent statistics.
Uses in-memory stats when cognee is not installed.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from constant_definitions.var.meta.reputation_constants import (
    COGNEE_DATASET_NAME,
    COGNEE_SEARCH_TYPE,
    DEFAULT_REPUTATION_SCORE_NUMERATOR,
    DEFAULT_REPUTATION_SCORE_DENOMINATOR,
    REPUTATION_DECAY_NUMERATOR,
    REPUTATION_DECAY_DENOMINATOR,
    META_KEY_COOPERATION_RATE,
    META_KEY_INTERACTION_COUNT,
    META_KEY_GOSSIP_HISTORY,
)

_ZERO = int()
_ONE = int(bool(True))
_DEFAULT_SCORE = (
    DEFAULT_REPUTATION_SCORE_NUMERATOR / DEFAULT_REPUTATION_SCORE_DENOMINATOR
)
_DECAY = REPUTATION_DECAY_NUMERATOR / REPUTATION_DECAY_DENOMINATOR

try:
    import cognee as _cognee  # type: ignore[import-untyped]
    _HAS_COGNEE = True
except ImportError:
    _cognee = None
    _HAS_COGNEE = False


class AsyncBridge:
    """Runs async coroutines from sync code via a dedicated thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._thread.start()

    def run(self, coro: Any) -> Any:
        """Submit *coro* to the background loop and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


def _default_reputation() -> dict[str, Any]:
    """Return a neutral default reputation dict."""
    return {
        "score": _DEFAULT_SCORE,
        META_KEY_COOPERATION_RATE: _DEFAULT_SCORE,
        META_KEY_INTERACTION_COUNT: _ZERO,
        META_KEY_GOSSIP_HISTORY: [],
    }


def _format_episode_text(
    agent_id: str,
    opponent_id: str,
    game: str,
    history: list[Any],
    cooperation_rate: float,
    scores: tuple[float, float],
) -> str:
    """Format an episode summary for cognee ingestion."""
    rounds = len(history)
    p_score, o_score = scores
    actions = "; ".join(
        f"R{r.round_number}: {r.player_action} vs {r.opponent_action}"
        for r in history
    )
    return (
        f"Game Interaction Report\n"
        f"Agent: {agent_id} | Opponent: {opponent_id} | Game: {game}\n"
        f"Rounds: {rounds} | Agent Score: {p_score} | "
        f"Opponent Score: {o_score}\n"
        f"Cooperation Rate: {cooperation_rate}\n"
        f"Actions: {actions}\n"
    )


def _parse_reputation(
    results: Any, stats: dict[str, Any],
) -> dict[str, Any]:
    """Merge cognee search results with in-memory stats."""
    rep = dict(stats) if stats else _default_reputation()
    if results:
        rep["cognee_context"] = str(results)
    return rep


class CogneeMemoryStore:
    """Persistent memory backed by cognee knowledge graph."""

    def __init__(self) -> None:
        self._bridge = AsyncBridge() if _HAS_COGNEE else None
        self._stats: dict[str, dict[str, Any]] = {}

    def record_episode(
        self,
        agent_id: str,
        opponent_id: str,
        game: str,
        history: list[Any],
        cooperation_rate: float,
        scores: tuple[float, float],
    ) -> None:
        """Format episode as text and add to cognee, then cognify."""
        text = _format_episode_text(
            agent_id, opponent_id, game, history,
            cooperation_rate, scores,
        )
        if self._bridge is not None and _HAS_COGNEE:
            try:
                self._bridge.run(
                    _cognee.add(text, dataset_name=COGNEE_DATASET_NAME),
                )
                self._bridge.run(_cognee.cognify())
            except Exception:
                pass
        self._update_stats(opponent_id, cooperation_rate, scores)

    def query_reputation(self, opponent_id: str) -> dict[str, Any]:
        """Query cognee for opponent reputation. Uses stats if unavailable."""
        stats = self._stats.get(opponent_id, _default_reputation())
        if self._bridge is None or not _HAS_COGNEE:
            return stats
        try:
            results = self._bridge.run(
                _cognee.search(
                    f"reputation and behavior of {opponent_id}",
                    search_type=COGNEE_SEARCH_TYPE,
                ),
            )
            return _parse_reputation(results, stats)
        except Exception:
            return stats

    def record_gossip(
        self, rater_id: str, target_id: str, rating: str,
    ) -> None:
        """Record a gossip rating in cognee."""
        text = f"{rater_id} rated {target_id} as {rating}."
        if self._bridge is not None and _HAS_COGNEE:
            try:
                self._bridge.run(
                    _cognee.add(text, dataset_name=COGNEE_DATASET_NAME),
                )
            except Exception:
                pass
        target_stats = self._stats.setdefault(
            target_id, _default_reputation(),
        )
        gossip_list = target_stats.setdefault(META_KEY_GOSSIP_HISTORY, [])
        gossip_list.append({"rater": rater_id, "rating": rating})

    def get_stats(self, opponent_id: str) -> dict[str, Any]:
        """Fast in-memory stats (no LLM call)."""
        return self._stats.get(opponent_id, _default_reputation())

    def _update_stats(
        self,
        opponent_id: str,
        coop_rate: float,
        scores: tuple[float, float],
    ) -> None:
        """Update running statistics for an opponent."""
        current = self._stats.get(opponent_id, _default_reputation())
        count = current.get(META_KEY_INTERACTION_COUNT, _ZERO) + _ONE
        old_coop = current.get(META_KEY_COOPERATION_RATE, _DEFAULT_SCORE)
        blended = old_coop * _DECAY + coop_rate * (_ONE - _DECAY)
        current["score"] = blended
        current[META_KEY_COOPERATION_RATE] = blended
        current[META_KEY_INTERACTION_COUNT] = count
        self._stats[opponent_id] = current
