"""Auction mechanism games for KantBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig
from constant_definitions.game_constants import SINGLE_SHOT_ROUNDS
from constant_definitions.auction_nplayer_constants import (
    AUCTION_ITEM_VALUE, AUCTION_MAX_BID, AUCTION_BID_INCREMENT,
)

_ONE = int(bool(True))
_ZERO = int()
_ZERO_F = float()


def _parse_bid(action: str) -> int:
    """Extract bid amount from action string like 'bid_5'."""
    return int(action.rsplit("_", _ONE)[_ONE])


# -- First-Price Sealed Bid Auction --

def _first_price_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Highest bidder wins and pays their own bid."""
    p_bid = _parse_bid(player_action)
    o_bid = _parse_bid(opponent_action)

    if p_bid > o_bid:
        p_pay = float(AUCTION_ITEM_VALUE - p_bid)
        o_pay = _ZERO_F
    elif o_bid > p_bid:
        p_pay = _ZERO_F
        o_pay = float(AUCTION_ITEM_VALUE - o_bid)
    else:
        half_surplus = float(AUCTION_ITEM_VALUE - p_bid) / (_ONE + _ONE)
        p_pay = half_surplus
        o_pay = half_surplus
    return (p_pay, o_pay)


# -- Second-Price (Vickrey) Auction --

def _vickrey_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Highest bidder wins but pays the second-highest bid."""
    p_bid = _parse_bid(player_action)
    o_bid = _parse_bid(opponent_action)

    if p_bid > o_bid:
        p_pay = float(AUCTION_ITEM_VALUE - o_bid)
        o_pay = _ZERO_F
    elif o_bid > p_bid:
        p_pay = _ZERO_F
        o_pay = float(AUCTION_ITEM_VALUE - p_bid)
    else:
        half_surplus = float(AUCTION_ITEM_VALUE - p_bid) / (_ONE + _ONE)
        p_pay = half_surplus
        o_pay = half_surplus
    return (p_pay, o_pay)


# -- All-Pay Auction --

def _allpay_payoff(
    player_action: str, opponent_action: str,
) -> tuple[float, float]:
    """Both bidders pay their bids; only the winner gets the item."""
    p_bid = _parse_bid(player_action)
    o_bid = _parse_bid(opponent_action)

    if p_bid > o_bid:
        p_pay = float(AUCTION_ITEM_VALUE - p_bid)
        o_pay = float(-o_bid)
    elif o_bid > p_bid:
        p_pay = float(-p_bid)
        o_pay = float(AUCTION_ITEM_VALUE - o_bid)
    else:
        half_value = float(AUCTION_ITEM_VALUE) / (_ONE + _ONE)
        p_pay = half_value - float(p_bid)
        o_pay = half_value - float(o_bid)
    return (p_pay, o_pay)


# -- Action lists --

_BID_ACTIONS = [
    f"bid_{i}" for i in range(
        _ZERO, AUCTION_MAX_BID + AUCTION_BID_INCREMENT, AUCTION_BID_INCREMENT,
    )
]


# -- Register --

AUCTION_GAMES: dict[str, GameConfig] = {
    "first_price_auction": GameConfig(
        name="First-Price Sealed-Bid Auction",
        description=(
            "Two bidders simultaneously submit sealed bids for an item. "
            "The highest bidder wins and pays their own bid. Strategic "
            "bidding requires shading below true value to maximize surplus "
            "while still winning."
        ),
        actions=_BID_ACTIONS,
        game_type="auction",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_first_price_payoff,
    ),
    "vickrey_auction": GameConfig(
        name="Second-Price (Vickrey) Auction",
        description=(
            "Two bidders submit sealed bids. The highest bidder wins but "
            "pays the second-highest bid. The dominant strategy is to bid "
            "one's true valuation, making this a strategy-proof mechanism."
        ),
        actions=_BID_ACTIONS,
        game_type="auction",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_vickrey_payoff,
    ),
    "allpay_auction": GameConfig(
        name="All-Pay Auction",
        description=(
            "Two bidders submit sealed bids. Both pay their bids regardless "
            "of outcome, but only the highest bidder receives the item. "
            "Models contests, lobbying, and rent-seeking where effort is "
            "spent whether or not you win."
        ),
        actions=_BID_ACTIONS,
        game_type="auction",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_allpay_payoff,
    ),
}

GAMES.update(AUCTION_GAMES)
