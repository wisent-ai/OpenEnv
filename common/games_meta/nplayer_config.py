"""N-player game configuration dataclass and registry."""

from __future__ import annotations

from common.games import GameConfig

NPlayerGameConfig = GameConfig

NPLAYER_GAMES: dict[str, GameConfig] = {}


def get_nplayer_game(name: str) -> GameConfig:
    """Look up an N-player game by name. Raises KeyError if not found."""
    return NPLAYER_GAMES[name]
