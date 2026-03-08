"""Game configuration registry and payoff computation for KantBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from constant_definitions.game_constants import (
    DEFAULT_ZERO_FLOAT,
    DEFAULT_ZERO_INT,
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
    SH_SS_PAYOFF, SH_SH_PAYOFF, SH_HS_PAYOFF, SH_HH_PAYOFF,
    HD_HH_PAYOFF, HD_HD_PAYOFF, HD_DH_PAYOFF, HD_DD_PAYOFF,
    ULTIMATUM_POT,
    TRUST_MULTIPLIER, TRUST_ENDOWMENT,
    PG_MULTIPLIER_NUMERATOR, PG_MULTIPLIER_DENOMINATOR,
    PG_ENDOWMENT, PG_DEFAULT_NUM_PLAYERS,
    DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS, DEFAULT_TWO_PLAYERS,
    OPPONENT_MODE_STRATEGY,
)

# ---------------------------------------------------------------------------
# GameConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    """Immutable specification for a single game type."""

    name: str
    description: str
    actions: list[str]
    game_type: str
    default_rounds: int
    payoff_fn: Callable
    num_players: int = DEFAULT_TWO_PLAYERS
    applied_variants: tuple[str, ...] = ()
    base_game_key: str = ""
    enforcement: str = ""
    penalty_numerator: int = DEFAULT_ZERO_INT
    penalty_denominator: int = SINGLE_SHOT_ROUNDS
    allow_side_payments: bool = False
    opponent_mode: str = OPPONENT_MODE_STRATEGY


# ---------------------------------------------------------------------------
# Matrix-game payoff helpers
# ---------------------------------------------------------------------------

_PD_MATRIX = {
    ("cooperate", "cooperate"): (float(PD_CC_PAYOFF), float(PD_CC_PAYOFF)),
    ("cooperate", "defect"):    (float(PD_CD_PAYOFF), float(PD_DC_PAYOFF)),
    ("defect", "cooperate"):    (float(PD_DC_PAYOFF), float(PD_CD_PAYOFF)),
    ("defect", "defect"):       (float(PD_DD_PAYOFF), float(PD_DD_PAYOFF)),
}

_SH_MATRIX = {
    ("stag", "stag"): (float(SH_SS_PAYOFF), float(SH_SS_PAYOFF)),
    ("stag", "hare"): (float(SH_SH_PAYOFF), float(SH_HS_PAYOFF)),
    ("hare", "stag"): (float(SH_HS_PAYOFF), float(SH_SH_PAYOFF)),
    ("hare", "hare"): (float(SH_HH_PAYOFF), float(SH_HH_PAYOFF)),
}

_HD_MATRIX = {
    ("hawk", "hawk"): (float(HD_HH_PAYOFF), float(HD_HH_PAYOFF)),
    ("hawk", "dove"): (float(HD_HD_PAYOFF), float(HD_DH_PAYOFF)),
    ("dove", "hawk"): (float(HD_DH_PAYOFF), float(HD_HD_PAYOFF)),
    ("dove", "dove"): (float(HD_DD_PAYOFF), float(HD_DD_PAYOFF)),
}


def _matrix_payoff_fn(matrix: dict) -> Callable:
    """Return a payoff function backed by a pre-built matrix dict."""

    def _payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
        return matrix[(player_action, opponent_action)]

    return _payoff


# ---------------------------------------------------------------------------
# Computed payoff functions
# ---------------------------------------------------------------------------


def _parse_action_amount(action: str) -> int:
    """Extract the integer suffix from an action string like 'offer_5'."""
    parts = action.rsplit("_", maxsplit=SINGLE_SHOT_ROUNDS)
    return int(parts[SINGLE_SHOT_ROUNDS])


def _ultimatum_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Compute Ultimatum Game payoffs.

    The player chooses an offer amount; the opponent accepts or rejects.
    """
    offer = _parse_action_amount(player_action)

    if opponent_action == "reject":
        return (DEFAULT_ZERO_FLOAT, DEFAULT_ZERO_FLOAT)

    # accepted
    player_payoff = float(ULTIMATUM_POT - offer)
    opponent_payoff = float(offer)
    return (player_payoff, opponent_payoff)


def _trust_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Compute Trust Game payoffs.

    The player invests X from their endowment. The opponent receives
    X * multiplier and returns Y of that amount.
    """
    investment = _parse_action_amount(player_action)
    returned = _parse_action_amount(opponent_action)

    player_payoff = float(TRUST_ENDOWMENT - investment + returned)
    opponent_payoff = float(investment * TRUST_MULTIPLIER - returned)
    return (player_payoff, opponent_payoff)


def _public_goods_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Compute Public Goods Game payoffs.

    Each participant contributes from their endowment. The total pot is
    multiplied by (numerator / denominator) then split equally among all
    participants.
    """
    player_contrib = _parse_action_amount(player_action)
    opponent_contrib = _parse_action_amount(opponent_action)

    total_contributions = player_contrib + opponent_contrib
    multiplied_pot = (
        total_contributions * PG_MULTIPLIER_NUMERATOR / PG_MULTIPLIER_DENOMINATOR
    )
    share = multiplied_pot / PG_DEFAULT_NUM_PLAYERS

    player_payoff = float(PG_ENDOWMENT - player_contrib) + share
    opponent_payoff = float(PG_ENDOWMENT - opponent_contrib) + share
    return (player_payoff, opponent_payoff)


# ---------------------------------------------------------------------------
# Action lists for computed games
# ---------------------------------------------------------------------------

_ULTIMATUM_OFFERS: list[str] = [
    f"offer_{i}" for i in range(ULTIMATUM_POT + SINGLE_SHOT_ROUNDS)
]

_TRUST_INVESTMENTS: list[str] = [
    f"invest_{i}" for i in range(TRUST_ENDOWMENT + SINGLE_SHOT_ROUNDS)
]

_PG_CONTRIBUTIONS: list[str] = [
    f"contribute_{i}" for i in range(PG_ENDOWMENT + SINGLE_SHOT_ROUNDS)
]


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------

GAMES: dict[str, GameConfig] = {
    "prisoners_dilemma": GameConfig(
        name="Prisoner's Dilemma",
        description=(
            "Two players simultaneously choose to cooperate or defect. "
            "Mutual cooperation yields a moderate reward, mutual defection "
            "yields a low reward, and unilateral defection tempts with the "
            "highest individual payoff at the other player's expense."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_PD_MATRIX),
    ),
    "stag_hunt": GameConfig(
        name="Stag Hunt",
        description=(
            "Two players choose between hunting stag (risky but rewarding "
            "if both participate) or hunting hare (safe but less rewarding). "
            "Coordination on stag yields the highest joint payoff."
        ),
        actions=["stag", "hare"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_SH_MATRIX),
    ),
    "hawk_dove": GameConfig(
        name="Hawk-Dove",
        description=(
            "Two players choose between aggressive (hawk) and passive (dove) "
            "strategies over a shared resource. Two hawks suffer mutual harm; "
            "a hawk facing a dove claims the resource; two doves share it."
        ),
        actions=["hawk", "dove"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_HD_MATRIX),
    ),
    "ultimatum": GameConfig(
        name="Ultimatum Game",
        description=(
            "The proposer offers a split of a fixed pot. The responder "
            "either accepts (both receive their shares) or rejects "
            "(both receive nothing)."
        ),
        actions=_ULTIMATUM_OFFERS,
        game_type="ultimatum",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_ultimatum_payoff,
    ),
    "trust": GameConfig(
        name="Trust Game",
        description=(
            "The investor sends part of an endowment; the amount is "
            "multiplied and given to the trustee, who then decides how "
            "much to return."
        ),
        actions=_TRUST_INVESTMENTS,
        game_type="trust",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_trust_payoff,
    ),
    "public_goods": GameConfig(
        name="Public Goods Game",
        description=(
            "Each participant decides how much of their endowment to "
            "contribute to a common pool. The pool is multiplied and "
            "distributed equally, creating tension between individual "
            "free-riding and collective benefit."
        ),
        actions=_PG_CONTRIBUTIONS,
        game_type="public_goods",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_public_goods_payoff,
    ),
}


def get_game(name: str) -> GameConfig:
    """Retrieve a GameConfig by its registry key.

    Args:
        name: Key in the GAMES registry (e.g. ``"prisoners_dilemma"``).

    Returns:
        The corresponding :class:`GameConfig` instance.

    Raises:
        KeyError: If *name* is not present in the registry.
    """
    return GAMES[name]


def _load_extensions() -> None:
    """Import extension modules that register additional games."""
    import importlib
    for mod in [
        "common.games_ext.matrix_games", "common.games_ext.sequential",
        "common.games_ext.auction", "common.games_ext.nplayer",
        "common.games_ext.generated", "common.games_info.signaling",
        "common.games_info.contracts", "common.games_info.communication",
        "common.games_info.bayesian", "common.games_info.network",
        "common.games_market.oligopoly", "common.games_market.contests",
        "common.games_market.classic", "common.games_market.generated_v2",
        "common.games_market.advanced", "common.games_coop.cooperative",
        "common.games_coop.dynamic", "common.games_coop.pd_variants",
        "common.games_coop.infinite", "common.games_coop.stochastic",
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass


_load_extensions()

from common.games_meta.dynamic import (  # noqa: E402,F401
    create_matrix_game, create_symmetric_game, create_custom_game,
)
