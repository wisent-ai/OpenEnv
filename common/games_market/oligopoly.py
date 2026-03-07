"""Market competition and bargaining games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.ext.market_constants import (
    COURNOT_DEMAND_INTERCEPT, COURNOT_DEMAND_SLOPE, COURNOT_MARGINAL_COST,
    COURNOT_MAX_QUANTITY,
    BERTRAND_MAX_PRICE, BERTRAND_MARGINAL_COST, BERTRAND_MARKET_SIZE,
    HOTELLING_LINE_LENGTH, HOTELLING_TRANSPORT_COST, HOTELLING_MARKET_VALUE,
    ED_MONOPOLY_PROFIT, ED_DUOPOLY_PROFIT, ED_FIGHT_COST,
    ED_ENTRANT_FIGHT_LOSS, ED_STAY_OUT_PAYOFF,
    ND_SURPLUS, DA_BUYER_VALUE, DA_SELLER_COST, DA_MAX_PRICE,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE
_ZERO_F = float()


def _cournot_payoff(pa: str, oa: str) -> tuple[float, float]:
    q_p = int(pa.rsplit("_", _ONE)[_ONE])
    q_o = int(oa.rsplit("_", _ONE)[_ONE])
    total = q_p + q_o
    price = COURNOT_DEMAND_INTERCEPT - COURNOT_DEMAND_SLOPE * total
    return (float((price - COURNOT_MARGINAL_COST) * q_p),
            float((price - COURNOT_MARGINAL_COST) * q_o))


def _bertrand_payoff(pa: str, oa: str) -> tuple[float, float]:
    p_p = int(pa.rsplit("_", _ONE)[_ONE])
    p_o = int(oa.rsplit("_", _ONE)[_ONE])
    if p_p < p_o:
        demand = max(BERTRAND_MARKET_SIZE - p_p, int())
        return (float((p_p - BERTRAND_MARGINAL_COST) * demand), _ZERO_F)
    if p_o < p_p:
        demand = max(BERTRAND_MARKET_SIZE - p_o, int())
        return (_ZERO_F, float((p_o - BERTRAND_MARGINAL_COST) * demand))
    demand = max(BERTRAND_MARKET_SIZE - p_p, int())
    half_profit = float((p_p - BERTRAND_MARGINAL_COST) * demand) / _TWO
    return (half_profit, half_profit)


def _hotelling_payoff(pa: str, oa: str) -> tuple[float, float]:
    loc_p = int(pa.rsplit("_", _ONE)[_ONE])
    loc_o = int(oa.rsplit("_", _ONE)[_ONE])
    if loc_p == loc_o:
        share = float(HOTELLING_MARKET_VALUE) / _TWO
        return (share, share)
    mid = (loc_p + loc_o) / _TWO
    p_share = mid if loc_p < loc_o else float(HOTELLING_LINE_LENGTH) - mid
    o_share = float(HOTELLING_LINE_LENGTH) - p_share
    return (float(p_share * HOTELLING_TRANSPORT_COST),
            float(o_share * HOTELLING_TRANSPORT_COST))


_ED_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("enter", "accommodate"): (float(ED_DUOPOLY_PROFIT), float(ED_DUOPOLY_PROFIT)),
    ("enter", "fight"):       (float(ED_ENTRANT_FIGHT_LOSS), float(ED_FIGHT_COST)),
    ("stay_out", "accommodate"): (float(ED_STAY_OUT_PAYOFF), float(ED_MONOPOLY_PROFIT)),
    ("stay_out", "fight"):    (float(ED_STAY_OUT_PAYOFF), float(ED_MONOPOLY_PROFIT)),
}


def _nash_demand_payoff(pa: str, oa: str) -> tuple[float, float]:
    d_p = int(pa.rsplit("_", _ONE)[_ONE])
    d_o = int(oa.rsplit("_", _ONE)[_ONE])
    if d_p + d_o <= ND_SURPLUS:
        return (float(d_p), float(d_o))
    return (_ZERO_F, _ZERO_F)


def _double_auction_payoff(pa: str, oa: str) -> tuple[float, float]:
    bid = int(pa.rsplit("_", _ONE)[_ONE])
    ask = int(oa.rsplit("_", _ONE)[_ONE])
    if bid >= ask:
        price = (bid + ask) // _TWO
        return (float(DA_BUYER_VALUE - price), float(price - DA_SELLER_COST))
    return (_ZERO_F, _ZERO_F)


_COURNOT_ACTS = [f"produce_{i}" for i in range(COURNOT_MAX_QUANTITY + _ONE)]
_BERTRAND_ACTS = [f"price_{i}" for i in range(BERTRAND_MAX_PRICE + _ONE)]
_HOTELLING_ACTS = [f"locate_{i}" for i in range(HOTELLING_LINE_LENGTH + _ONE)]
_ND_ACTS = [f"demand_{i}" for i in range(ND_SURPLUS + _ONE)]
_DA_ACTS = [f"bid_{i}" for i in range(DA_MAX_PRICE + _ONE)]

OLIGOPOLY_GAMES: dict[str, GameConfig] = {
    "cournot": GameConfig(
        name="Cournot Duopoly",
        description=(
            "Two firms simultaneously choose production quantities. "
            "Market price decreases with total output. Tests Nash "
            "equilibrium reasoning in quantity competition."
        ),
        actions=_COURNOT_ACTS, game_type="cournot",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_cournot_payoff,
    ),
    "bertrand": GameConfig(
        name="Bertrand Competition",
        description=(
            "Two firms simultaneously set prices. The lower-price firm "
            "captures the market. The Bertrand paradox predicts pricing "
            "at marginal cost even with only two competitors."
        ),
        actions=_BERTRAND_ACTS, game_type="bertrand",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_bertrand_payoff,
    ),
    "hotelling": GameConfig(
        name="Hotelling Location Game",
        description=(
            "Two firms choose locations on a line. Consumers visit the "
            "nearest firm. Tests the principle of minimum differentiation "
            "and spatial competition dynamics."
        ),
        actions=_HOTELLING_ACTS, game_type="hotelling",
        default_rounds=DEFAULT_NUM_ROUNDS, payoff_fn=_hotelling_payoff,
    ),
    "entry_deterrence": GameConfig(
        name="Entry Deterrence",
        description=(
            "A potential entrant decides whether to enter a market; "
            "the incumbent decides whether to fight or accommodate. "
            "Tests credible commitment and limit pricing reasoning."
        ),
        actions=["enter", "stay_out"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_ED_MATRIX),
    ),
    "nash_demand": GameConfig(
        name="Nash Demand Game",
        description=(
            "Two players simultaneously demand shares of a surplus. "
            "If demands are compatible (sum within surplus), both "
            "receive their demand; otherwise both get nothing."
        ),
        actions=_ND_ACTS, game_type="nash_demand",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_nash_demand_payoff,
    ),
    "double_auction": GameConfig(
        name="Double Auction",
        description=(
            "A buyer submits a bid and a seller submits an ask. Trade "
            "occurs at the midpoint if bid exceeds ask. Tests price "
            "discovery and competitive market behavior."
        ),
        actions=_DA_ACTS, game_type="double_auction",
        default_rounds=SINGLE_SHOT_ROUNDS, payoff_fn=_double_auction_payoff,
    ),
}

GAMES.update(OLIGOPOLY_GAMES)
