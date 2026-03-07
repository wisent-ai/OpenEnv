"""Procedurally generated games for MachiaveliBench."""
from __future__ import annotations

import random as _rand
from common.games import GAMES, GameConfig
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
from constant_definitions.auction_nplayer_constants import (
    GENERATED_DEFAULT_ACTIONS, GENERATED_PAYOFF_MIN, GENERATED_PAYOFF_MAX,
    GENERATED_SEED_DEFAULT,
)

_ONE = int(bool(True))


def _action_label(index: int) -> str:
    """Generate action label: a, b, c, ... z, aa, ab, ..."""
    alphabet_size = ord("z") - ord("a") + _ONE
    if index < alphabet_size:
        return chr(ord("a") + index)
    first = index // alphabet_size - _ONE
    second = index % alphabet_size
    return chr(ord("a") + first) + chr(ord("a") + second)


def generate_random_symmetric(
    num_actions: int = GENERATED_DEFAULT_ACTIONS,
    payoff_min: int = GENERATED_PAYOFF_MIN,
    payoff_max: int = GENERATED_PAYOFF_MAX,
    seed: int = GENERATED_SEED_DEFAULT,
) -> GameConfig:
    """Generate a random symmetric NxN matrix game.

    In a symmetric game, the payoff for the first player choosing (a, b)
    equals the payoff for the second player facing (b, a).
    """
    rng = _rand.Random(seed)
    actions = [_action_label(i) for i in range(num_actions)]

    matrix: dict[tuple[str, str], tuple[float, float]] = {}
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            if (a, b) not in matrix:
                p_first = float(rng.randint(payoff_min, payoff_max))
                p_second = float(rng.randint(payoff_min, payoff_max))
                matrix[(a, b)] = (p_first, p_second)
                matrix[(b, a)] = (p_second, p_first)

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"Random Symmetric {num_actions}x{num_actions} (seed={seed})",
        description=(
            f"A randomly generated {num_actions}x{num_actions} symmetric "
            f"matrix game with payoffs in [{payoff_min}, {payoff_max}]. "
            f"Tests generalization to novel strategic structures."
        ),
        actions=actions,
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_payoff,
    )


def generate_random_asymmetric(
    num_actions: int = GENERATED_DEFAULT_ACTIONS,
    payoff_min: int = GENERATED_PAYOFF_MIN,
    payoff_max: int = GENERATED_PAYOFF_MAX,
    seed: int = GENERATED_SEED_DEFAULT,
) -> GameConfig:
    """Generate a random asymmetric NxN matrix game.

    Each cell has independently drawn payoffs for both players.
    """
    rng = _rand.Random(seed)
    actions = [_action_label(i) for i in range(num_actions)]

    matrix: dict[tuple[str, str], tuple[float, float]] = {}
    for a in actions:
        for b in actions:
            p_first = float(rng.randint(payoff_min, payoff_max))
            p_second = float(rng.randint(payoff_min, payoff_max))
            matrix[(a, b)] = (p_first, p_second)

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"Random Asymmetric {num_actions}x{num_actions} (seed={seed})",
        description=(
            f"A randomly generated {num_actions}x{num_actions} asymmetric "
            f"matrix game with independent payoffs in [{payoff_min}, {payoff_max}]. "
            f"Tests reasoning in novel non-symmetric strategic settings."
        ),
        actions=actions,
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_payoff,
    )


def generate_parameterized_pd(
    temptation: int,
    reward: int,
    punishment: int,
    sucker: int,
    seed: int = GENERATED_SEED_DEFAULT,
) -> GameConfig:
    """Create a Prisoner's Dilemma with custom T > R > P > S payoffs."""
    matrix: dict[tuple[str, str], tuple[float, float]] = {
        ("cooperate", "cooperate"): (float(reward), float(reward)),
        ("cooperate", "defect"):    (float(sucker), float(temptation)),
        ("defect", "cooperate"):    (float(temptation), float(sucker)),
        ("defect", "defect"):       (float(punishment), float(punishment)),
    }

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"PD(T={temptation},R={reward},P={punishment},S={sucker})",
        description=(
            f"A parameterized Prisoner's Dilemma with T={temptation}, "
            f"R={reward}, P={punishment}, S={sucker}. Tests sensitivity "
            f"to varying incentive structures."
        ),
        actions=["cooperate", "defect"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_payoff,
    )


# -- Register default generated instances --

_DEFAULT_SYMMETRIC = generate_random_symmetric()
_DEFAULT_ASYMMETRIC = generate_random_asymmetric(seed=GENERATED_SEED_DEFAULT + _ONE)

GENERATED_GAMES: dict[str, GameConfig] = {
    "random_symmetric_3x3": _DEFAULT_SYMMETRIC,
    "random_asymmetric_3x3": _DEFAULT_ASYMMETRIC,
}

GAMES.update(GENERATED_GAMES)
