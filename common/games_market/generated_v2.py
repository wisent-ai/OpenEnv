"""Extended procedurally generated games for KantBench."""
from __future__ import annotations

import random as _rand

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
from constant_definitions.var.generated_ext_constants import (
    RZS_SEED, RZS_MAX_PAYOFF, RZS_DEFAULT_ACTIONS,
    RC_SEED, RC_MATCH_BONUS, RC_MISMATCH_MAX, RC_DEFAULT_ACTIONS,
    PCHK_RESOURCE, PCHK_FIGHT_COST,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE


def _action_label(index: int) -> str:
    return chr(ord("a") + index)


def generate_random_zero_sum(
    num_actions: int = RZS_DEFAULT_ACTIONS,
    max_payoff: int = RZS_MAX_PAYOFF,
    seed: int = RZS_SEED,
) -> GameConfig:
    """Generate a random NxN zero-sum game."""
    rng = _rand.Random(seed)
    actions = [_action_label(i) for i in range(num_actions)]
    matrix: dict[tuple[str, str], tuple[float, float]] = {}
    for a in actions:
        for b in actions:
            val = float(rng.randint(-max_payoff, max_payoff))
            matrix[(a, b)] = (val, -val)

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"Random Zero-Sum {num_actions}x{num_actions} (seed={seed})",
        description=(
            f"A randomly generated {num_actions}x{num_actions} zero-sum "
            f"game. Every outcome sums to zero. Tests minimax reasoning "
            f"in adversarial strategic settings."
        ),
        actions=actions, game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_payoff,
    )


def generate_random_coordination(
    num_actions: int = RC_DEFAULT_ACTIONS,
    match_bonus: int = RC_MATCH_BONUS,
    mismatch_max: int = RC_MISMATCH_MAX,
    seed: int = RC_SEED,
) -> GameConfig:
    """Generate a random NxN coordination game with diagonal bonus."""
    rng = _rand.Random(seed)
    actions = [_action_label(i) for i in range(num_actions)]
    matrix: dict[tuple[str, str], tuple[float, float]] = {}
    for a in actions:
        for b in actions:
            if a == b:
                val = float(match_bonus + rng.randint(int(), mismatch_max))
                matrix[(a, b)] = (val, val)
            else:
                val = float(rng.randint(int(), mismatch_max))
                matrix[(a, b)] = (val, val)

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"Random Coordination {num_actions}x{num_actions} (seed={seed})",
        description=(
            f"A randomly generated {num_actions}x{num_actions} coordination "
            f"game. Matching actions receive a bonus payoff. Tests focal "
            f"point identification in novel coordination structures."
        ),
        actions=actions, game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_payoff,
    )


def generate_parameterized_chicken(
    resource: int = PCHK_RESOURCE,
    fight_cost: int = PCHK_FIGHT_COST,
) -> GameConfig:
    """Create a Hawk-Dove / Chicken game with custom parameters."""
    half_v = float(resource) / _TWO
    fight_pay = (float(resource) - float(fight_cost)) / _TWO
    matrix: dict[tuple[str, str], tuple[float, float]] = {
        ("hawk", "hawk"):   (fight_pay, fight_pay),
        ("hawk", "dove"):   (float(resource), float(int())),
        ("dove", "hawk"):   (float(int()), float(resource)),
        ("dove", "dove"):   (half_v, half_v),
    }

    def _payoff(pa: str, oa: str) -> tuple[float, float]:
        return matrix[(pa, oa)]

    return GameConfig(
        name=f"Chicken(V={resource},C={fight_cost})",
        description=(
            f"A parameterized Chicken / Hawk-Dove game with resource value "
            f"{resource} and fight cost {fight_cost}. Tests anti-coordination "
            f"behavior under varied incentive parameters."
        ),
        actions=["hawk", "dove"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_payoff,
    )


# -- Register default instances --
_ZS = generate_random_zero_sum()
_CO = generate_random_coordination()
_CH = generate_parameterized_chicken()

GENERATED_V2: dict[str, GameConfig] = {
    "random_zero_sum_3x3": _ZS,
    "random_coordination_3x3": _CO,
    "parameterized_chicken": _CH,
}

GAMES.update(GENERATED_V2)
