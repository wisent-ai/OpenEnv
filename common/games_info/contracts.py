"""Principal-agent and contract theory games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import SINGLE_SHOT_ROUNDS
from constant_definitions.ext.dynamic_constants import (
    MH_BASE_OUTPUT, MH_EFFORT_BOOST, MH_EFFORT_COST, MH_MAX_BONUS,
    SCR_HIGH_TYPE_VALUE, SCR_LOW_TYPE_VALUE,
    SCR_PREMIUM_PRICE, SCR_BASIC_PRICE,
    GE_MAX_WAGE, GE_MAX_EFFORT,
    GE_EFFORT_COST_PER_UNIT, GE_PRODUCTIVITY_PER_EFFORT,
)

_ONE = int(bool(True))
_ZERO = int()


# -- Moral Hazard --
def _moral_hazard_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Principal sets bonus; agent chooses effort.

    Principal: output - bonus if agent works.
    Agent: bonus - effort_cost if working, base if shirking.
    """
    bonus = int(player_action.rsplit("_", _ONE)[_ONE])
    works = opponent_action == "work"
    output = MH_BASE_OUTPUT + MH_EFFORT_BOOST if works else MH_BASE_OUTPUT
    principal_pay = float(output - bonus)
    agent_pay = float(bonus - MH_EFFORT_COST) if works else float(bonus)
    return (principal_pay, agent_pay)


_MH_BONUS_ACTIONS = [f"bonus_{i}" for i in range(MH_MAX_BONUS + _ONE)]


# -- Screening --
def _screening_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Principal offers contract menu; agent self-selects.

    Agent picks premium or basic contract based on private type.
    """
    if player_action == "offer_premium":
        price = SCR_PREMIUM_PRICE
    else:
        price = SCR_BASIC_PRICE

    if opponent_action == "choose_premium":
        buyer_value = SCR_HIGH_TYPE_VALUE
        seller_pay = float(SCR_PREMIUM_PRICE)
        buyer_pay = float(buyer_value - SCR_PREMIUM_PRICE)
    else:
        buyer_value = SCR_LOW_TYPE_VALUE
        seller_pay = float(SCR_BASIC_PRICE)
        buyer_pay = float(buyer_value - SCR_BASIC_PRICE)

    return (seller_pay, buyer_pay)


# -- Gift Exchange --
def _gift_exchange_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Employer offers wage; worker chooses effort.

    Employer profit = productivity * effort - wage.
    Worker payoff = wage - effort_cost * effort.
    """
    wage = int(player_action.rsplit("_", _ONE)[_ONE])
    effort = int(opponent_action.rsplit("_", _ONE)[_ONE])
    employer_pay = float(GE_PRODUCTIVITY_PER_EFFORT * effort - wage)
    worker_pay = float(wage - GE_EFFORT_COST_PER_UNIT * effort)
    return (employer_pay, worker_pay)


_GE_WAGE_ACTIONS = [f"wage_{i}" for i in range(GE_MAX_WAGE + _ONE)]


# -- Register --
CONTRACT_GAMES: dict[str, GameConfig] = {
    "moral_hazard": GameConfig(
        name="Moral Hazard (Principal-Agent)",
        description=(
            "A principal offers a bonus contract; an agent with "
            "unobservable effort decides whether to work or shirk. "
            "Tests optimal incentive design and the tradeoff between "
            "motivation and rent extraction."
        ),
        actions=_MH_BONUS_ACTIONS,
        game_type="moral_hazard",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_moral_hazard_payoff,
    ),
    "screening": GameConfig(
        name="Screening Game",
        description=(
            "An uninformed principal offers a menu of contracts; "
            "agents of different types self-select. Tests understanding "
            "of incentive compatibility and separating mechanisms "
            "as in Rothschild-Stiglitz insurance models."
        ),
        actions=["offer_premium", "offer_basic"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_screening_payoff,
    ),
    "gift_exchange": GameConfig(
        name="Gift Exchange Game",
        description=(
            "An employer offers a wage; a worker chooses effort. "
            "Nash prediction is minimal effort regardless of wage, "
            "but reciprocity often leads to higher wages eliciting "
            "higher effort. Tests fairness-driven behavior."
        ),
        actions=_GE_WAGE_ACTIONS,
        game_type="gift_exchange",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_gift_exchange_payoff,
    ),
}

GAMES.update(CONTRACT_GAMES)
