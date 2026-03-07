"""Sequential (extensive-form) games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import SINGLE_SHOT_ROUNDS, DEFAULT_NUM_ROUNDS
from constant_definitions.sequential_constants import (
    DICTATOR_ENDOWMENT,
    CENTIPEDE_INITIAL_POT, CENTIPEDE_GROWTH_MULTIPLIER, CENTIPEDE_MAX_STAGES,
    CENTIPEDE_LARGE_SHARE_NUMERATOR, CENTIPEDE_LARGE_SHARE_DENOMINATOR,
    CENTIPEDE_SMALL_SHARE_NUMERATOR, CENTIPEDE_SMALL_SHARE_DENOMINATOR,
    STACKELBERG_DEMAND_INTERCEPT, STACKELBERG_DEMAND_SLOPE,
    STACKELBERG_MARGINAL_COST, STACKELBERG_MAX_QUANTITY,
)

_ONE = int(bool(True))


# -- Dictator Game --

def _dictator_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Dictator allocates from endowment; recipient has no choice."""
    amount = int(player_action.rsplit("_", _ONE)[_ONE])
    dictator_keeps = float(DICTATOR_ENDOWMENT - amount)
    recipient_gets = float(amount)
    return (dictator_keeps, recipient_gets)


_DICTATOR_ACTIONS = [
    f"give_{i}" for i in range(DICTATOR_ENDOWMENT + _ONE)
]


# -- Centipede Game --

def _centipede_payoff(player_action: str, opponent_action: str) -> tuple[float, float]:
    """Alternating pass/take game with growing pot.

    Actions encode the stage: 'take_N' means take at stage N,
    'pass_all' means pass through all stages.
    The opponent strategy similarly responds with take or pass.
    """
    if player_action == "pass_all":
        player_stage = CENTIPEDE_MAX_STAGES + _ONE
    else:
        player_stage = int(player_action.rsplit("_", _ONE)[_ONE])

    if opponent_action == "pass_all":
        opp_stage = CENTIPEDE_MAX_STAGES + _ONE
    else:
        opp_stage = int(opponent_action.rsplit("_", _ONE)[_ONE])

    take_stage = min(player_stage, opp_stage)

    pot = CENTIPEDE_INITIAL_POT
    for _ in range(take_stage):
        pot = pot * CENTIPEDE_GROWTH_MULTIPLIER

    large = pot * CENTIPEDE_LARGE_SHARE_NUMERATOR // CENTIPEDE_LARGE_SHARE_DENOMINATOR
    small = pot * CENTIPEDE_SMALL_SHARE_NUMERATOR // CENTIPEDE_SMALL_SHARE_DENOMINATOR

    if player_stage <= opp_stage:
        return (float(large), float(small))
    return (float(small), float(large))


_CENTIPEDE_ACTIONS = [
    f"take_{i}" for i in range(CENTIPEDE_MAX_STAGES + _ONE)
] + ["pass_all"]


# -- Stackelberg Competition --

def _stackelberg_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Stackelberg duopoly: leader (player) and follower (opponent).

    Profit = (demand_intercept - slope * (q_leader + q_follower) - cost) * q
    """
    q_leader = int(player_action.rsplit("_", _ONE)[_ONE])
    q_follower = int(opponent_action.rsplit("_", _ONE)[_ONE])

    total_q = q_leader + q_follower
    price = STACKELBERG_DEMAND_INTERCEPT - STACKELBERG_DEMAND_SLOPE * total_q

    leader_profit = float((price - STACKELBERG_MARGINAL_COST) * q_leader)
    follower_profit = float((price - STACKELBERG_MARGINAL_COST) * q_follower)
    return (leader_profit, follower_profit)


_STACKELBERG_ACTIONS = [
    f"produce_{i}" for i in range(STACKELBERG_MAX_QUANTITY + _ONE)
]


# -- Register --

SEQUENTIAL_GAMES: dict[str, GameConfig] = {
    "dictator": GameConfig(
        name="Dictator Game",
        description=(
            "One player (the dictator) decides how to split an endowment "
            "with a passive recipient who has no say. Tests fairness "
            "preferences and altruistic behavior when there is no strategic "
            "incentive to share."
        ),
        actions=_DICTATOR_ACTIONS,
        game_type="dictator",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_dictator_payoff,
    ),
    "centipede": GameConfig(
        name="Centipede Game",
        description=(
            "Players alternate deciding to take or pass. Each pass doubles "
            "the pot. The taker gets the larger share while the other gets "
            "the smaller share. Backward induction predicts immediate taking, "
            "but cooperation through passing yields higher joint payoffs."
        ),
        actions=_CENTIPEDE_ACTIONS,
        game_type="centipede",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_centipede_payoff,
    ),
    "stackelberg": GameConfig(
        name="Stackelberg Competition",
        description=(
            "A quantity-setting duopoly where the leader commits to a "
            "production quantity first, and the follower observes and "
            "responds. The leader can exploit first-mover advantage. "
            "Price is determined by total market quantity."
        ),
        actions=_STACKELBERG_ACTIONS,
        game_type="stackelberg",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_stackelberg_payoff,
    ),
}

GAMES.update(SEQUENTIAL_GAMES)
