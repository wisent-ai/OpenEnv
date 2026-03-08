"""ArenaGamePool — manages available games for the metagame arena."""
from __future__ import annotations

import random
from typing import Any, Optional

from common.games import GAMES
from common.games_meta.dynamic import create_matrix_game
from constant_definitions.arena.arena_constants import (
    DEFAULT_GAMES_PER_ROUND,
    DEFAULT_POOL_SIZE,
)

_ZERO = int()
_ONE = int(bool(True))

_DEFAULT_GAME_KEYS = (
    "prisoners_dilemma",
    "stag_hunt",
    "hawk_dove",
    "trust",
    "public_goods",
)


class ArenaGamePool:
    """Manages the set of games available for arena play.

    Maintains a default pool of classic games plus any model-proposed
    custom games created via ``create_matrix_game``.
    """

    def __init__(self) -> None:
        self._games: list[str] = [
            key for key in _DEFAULT_GAME_KEYS if key in GAMES
        ]
        self._custom_games: list[str] = []
        self._play_counts: dict[str, int] = {}

    @property
    def available_games(self) -> list[str]:
        """All games currently in the pool."""
        return list(self._games)

    @property
    def custom_games(self) -> list[str]:
        """Model-proposed custom games."""
        return list(self._custom_games)

    def register_model_game(
        self,
        name: str,
        actions: list[str],
        payoff_matrix: dict[tuple[str, str], tuple[float, float]],
        description: str = "",
    ) -> Optional[str]:
        """Parse an LLM-proposed game definition and register it.

        Returns the registry key on success, None on failure.
        """
        try:
            config = create_matrix_game(
                name=name,
                actions=actions,
                payoff_matrix=payoff_matrix,
                description=description,
                register=True,
            )
            key = f"dynamic_{name}"
            if key not in self._games:
                self._games.append(key)
                self._custom_games.append(key)
            return key
        except (ValueError, KeyError):
            return None

    def select_games(
        self,
        count: int = DEFAULT_GAMES_PER_ROUND,
    ) -> list[str]:
        """Pick games for this round, weighted by inverse play frequency."""
        pool = self._games
        if not pool:
            return []
        actual_count = min(count, len(pool))
        max_count = max(
            (self._play_counts.get(g, _ZERO) for g in pool),
            default=_ONE,
        )
        weights = [
            max_count - self._play_counts.get(g, _ZERO) + _ONE
            for g in pool
        ]
        selected = []
        remaining = list(zip(pool, weights))
        for _ in range(actual_count):
            if not remaining:
                break
            games_only = [r[_ZERO] for r in remaining]
            w_only = [r[_ONE] for r in remaining]
            choice = random.choices(games_only, weights=w_only, k=_ONE)[_ZERO]
            selected.append(choice)
            remaining = [r for r in remaining if r[_ZERO] != choice]
        return selected

    def record_play(self, game_key: str) -> None:
        """Increment the play count for a game."""
        self._play_counts[game_key] = (
            self._play_counts.get(game_key, _ZERO) + _ONE
        )

    def remove_game(self, game_key: str) -> bool:
        """Remove a game from the pool."""
        if game_key in self._games:
            self._games.remove(game_key)
            if game_key in self._custom_games:
                self._custom_games.remove(game_key)
            return True
        return False
