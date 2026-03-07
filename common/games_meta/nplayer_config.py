"""N-player game configuration dataclass and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from constant_definitions.nplayer.nplayer_constants import (
    NPLAYER_DEFAULT_ROUNDS,
)


@dataclass(frozen=True)
class NPlayerGameConfig:
    """Immutable specification for an N-player game type."""

    name: str
    description: str
    actions: list[str]
    num_players: int
    default_rounds: int
    payoff_fn: Callable[[tuple[str, ...]], tuple[float, ...]]


NPLAYER_GAMES: dict[str, NPlayerGameConfig] = {}


def get_nplayer_game(name: str) -> NPlayerGameConfig:
    """Look up an N-player game by name. Raises KeyError if not found."""
    return NPLAYER_GAMES[name]
