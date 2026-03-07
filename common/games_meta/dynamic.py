"""Dynamic game creation API for building games at runtime."""

from __future__ import annotations

from typing import Callable

from common.games import GameConfig, GAMES, _matrix_payoff_fn
from constant_definitions.nplayer.dynamic_constants import (
    MIN_ACTIONS,
    MAX_ACTIONS,
    DYNAMIC_DEFAULT_ROUNDS,
    REGISTRY_PREFIX,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE


def _validate_actions(actions: list[str]) -> None:
    """Raise ValueError if action list is invalid."""
    if len(actions) < MIN_ACTIONS:
        raise ValueError(
            f"Need at least {MIN_ACTIONS} actions, got {len(actions)}"
        )
    if len(actions) > MAX_ACTIONS:
        raise ValueError(
            f"At most {MAX_ACTIONS} actions allowed, got {len(actions)}"
        )
    if len(actions) != len(set(actions)):
        raise ValueError("Duplicate actions are not allowed")


def _validate_matrix(
    actions: list[str],
    payoff_matrix: dict[tuple[str, str], tuple[float, float]],
) -> None:
    """Raise ValueError if the matrix is incomplete or has invalid keys."""
    expected = {(a, b) for a in actions for b in actions}
    actual = set(payoff_matrix.keys())
    missing = expected - actual
    if missing:
        raise ValueError(f"Payoff matrix is missing entries: {missing}")
    extra = actual - expected
    if extra:
        raise ValueError(f"Payoff matrix has unknown action pairs: {extra}")


def create_matrix_game(
    name: str,
    actions: list[str],
    payoff_matrix: dict[tuple[str, str], tuple[float, float]],
    *,
    description: str = "",
    default_rounds: int = DYNAMIC_DEFAULT_ROUNDS,
    register: bool = False,
) -> GameConfig:
    """Create a GameConfig backed by an explicit payoff matrix.

    Parameters
    ----------
    name:
        Display name for the game.
    actions:
        List of action strings available to both players.
    payoff_matrix:
        ``{(player_action, opponent_action): (player_pay, opponent_pay)}``.
    description:
        Human-readable description of the game rules.
    default_rounds:
        Number of rounds when the caller does not specify.
    register:
        If ``True``, add the game to the global ``GAMES`` registry using the
        key ``dynamic_<name>``.

    Returns
    -------
    GameConfig
    """
    _validate_actions(actions)
    _validate_matrix(actions, payoff_matrix)
    config = GameConfig(
        name=name,
        description=description or f"Dynamic matrix game: {name}",
        actions=list(actions),
        game_type="matrix",
        default_rounds=default_rounds,
        payoff_fn=_matrix_payoff_fn(dict(payoff_matrix)),
    )
    if register:
        key = REGISTRY_PREFIX + name
        GAMES[key] = config
    return config


def create_symmetric_game(
    name: str,
    actions: list[str],
    payoffs: dict[tuple[str, str], float],
    *,
    description: str = "",
    default_rounds: int = DYNAMIC_DEFAULT_ROUNDS,
    register: bool = False,
) -> GameConfig:
    """Create a symmetric GameConfig from single-value payoffs.

    In a symmetric game, ``payoff(A, B)`` for the row player equals
    ``payoff(B, A)`` for the column player. You only specify the row-player
    payoff for each cell and the full matrix is derived.

    Parameters
    ----------
    name:
        Display name.
    actions:
        List of action strings.
    payoffs:
        ``{(my_action, their_action): my_payoff}``.
    description:
        Human-readable description.
    default_rounds:
        Number of rounds.
    register:
        If ``True``, register as ``dynamic_<name>``.

    Returns
    -------
    GameConfig
    """
    _validate_actions(actions)
    expected = {(a, b) for a in actions for b in actions}
    actual = set(payoffs.keys())
    missing = expected - actual
    if missing:
        raise ValueError(f"Symmetric payoff table is missing entries: {missing}")

    full_matrix: dict[tuple[str, str], tuple[float, float]] = {}
    for a in actions:
        for b in actions:
            full_matrix[(a, b)] = (payoffs[(a, b)], payoffs[(b, a)])

    return create_matrix_game(
        name,
        actions,
        full_matrix,
        description=description,
        default_rounds=default_rounds,
        register=register,
    )


def create_custom_game(
    name: str,
    actions: list[str],
    payoff_fn: Callable[[str, str], tuple[float, float]],
    *,
    game_type: str = "matrix",
    description: str = "",
    default_rounds: int = DYNAMIC_DEFAULT_ROUNDS,
    register: bool = False,
) -> GameConfig:
    """Create a GameConfig with an arbitrary payoff function.

    Parameters
    ----------
    name:
        Display name.
    actions:
        List of action strings.
    payoff_fn:
        ``(player_action, opponent_action) -> (player_pay, opponent_pay)``.
    game_type:
        Game type tag (default ``"matrix"``).
    description:
        Human-readable description.
    default_rounds:
        Number of rounds.
    register:
        If ``True``, register as ``dynamic_<name>``.

    Returns
    -------
    GameConfig
    """
    _validate_actions(actions)
    config = GameConfig(
        name=name,
        description=description or f"Dynamic custom game: {name}",
        actions=list(actions),
        game_type=game_type,
        default_rounds=default_rounds,
        payoff_fn=payoff_fn,
    )
    if register:
        key = REGISTRY_PREFIX + name
        GAMES[key] = config
    return config


def unregister_game(key: str) -> None:
    """Remove a game from the global ``GAMES`` registry.

    Raises ``KeyError`` if the key is not found.
    """
    del GAMES[key]
