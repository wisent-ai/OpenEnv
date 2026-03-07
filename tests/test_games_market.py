"""Tests for market competition and contest games."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.ext.market_constants import (
    COURNOT_DEMAND_INTERCEPT, COURNOT_MARGINAL_COST,
    BERTRAND_MARGINAL_COST,
    ED_MONOPOLY_PROFIT, ED_DUOPOLY_PROFIT, ED_STAY_OUT_PAYOFF,
    ND_SURPLUS,
    DA_BUYER_VALUE, DA_SELLER_COST,
)
from constant_definitions.ext.conflict_constants import (
    BLOTTO_TOTAL_TROOPS,
    WOA_PRIZE, WOA_COST_PER_ROUND,
    TULLOCK_PRIZE,
    INSP_VIOLATION_GAIN, INSP_FINE,
    RUB_SURPLUS,
    DAC_ENDOWMENT,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_SIX = _FIVE + _ONE

_MARKET_KEYS = [
    "cournot", "bertrand", "hotelling", "entry_deterrence",
    "nash_demand", "double_auction",
    "colonel_blotto", "war_of_attrition", "tullock_contest",
    "inspection_game", "rubinstein_bargaining", "divide_and_choose",
]


class TestMarketRegistry:
    @pytest.mark.parametrize("key", _MARKET_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _MARKET_KEYS)
    def test_game_has_payoff(self, key: str) -> None:
        game = get_game(key)
        assert game.payoff_fn is not None


class TestCournot:
    _game = get_game("cournot")

    def test_zero_production(self) -> None:
        p, o = self._game.payoff_fn("produce_0", "produce_0")
        assert p == float(_ZERO)

    def test_symmetric_output(self) -> None:
        p, o = self._game.payoff_fn("produce_3", "produce_3")
        assert p == o


class TestBertrand:
    _game = get_game("bertrand")

    def test_undercut_wins(self) -> None:
        p, o = self._game.payoff_fn("price_4", "price_5")
        assert p > _ZERO
        assert o == float(_ZERO)


class TestEntryDeterrence:
    _game = get_game("entry_deterrence")

    def test_stay_out(self) -> None:
        p, o = self._game.payoff_fn("stay_out", "accommodate")
        assert p == float(ED_STAY_OUT_PAYOFF)
        assert o == float(ED_MONOPOLY_PROFIT)

    def test_enter_accommodate(self) -> None:
        p, o = self._game.payoff_fn("enter", "accommodate")
        assert p == float(ED_DUOPOLY_PROFIT)


class TestNashDemand:
    _game = get_game("nash_demand")

    def test_compatible_demands(self) -> None:
        p, o = self._game.payoff_fn("demand_4", "demand_6")
        assert p == float(_FIVE - _ONE)
        assert o == float(_SIX)

    def test_excessive_demands(self) -> None:
        p, o = self._game.payoff_fn("demand_6", "demand_6")
        assert p == float(_ZERO)


class TestDoubleAuction:
    _game = get_game("double_auction")

    def test_trade_occurs(self) -> None:
        p, o = self._game.payoff_fn("bid_6", "bid_4")
        price = (_SIX + _FIVE - _ONE) // _TWO
        assert p == float(DA_BUYER_VALUE - price)

    def test_no_trade(self) -> None:
        p, o = self._game.payoff_fn("bid_3", "bid_5")
        assert p == float(_ZERO)


class TestColonelBlotto:
    _game = get_game("colonel_blotto")

    def test_win_all_battlefields(self) -> None:
        p, _ = self._game.payoff_fn("alloc_6_0_0", "alloc_0_3_3")
        assert p >= _ONE

    def test_action_count(self) -> None:
        assert len(self._game.actions) > _FIVE


class TestWarOfAttrition:
    _game = get_game("war_of_attrition")

    def test_higher_persistence_wins(self) -> None:
        p, o = self._game.payoff_fn("persist_5", "persist_3")
        assert p > _ZERO
        assert o <= _ZERO


class TestTullockContest:
    _game = get_game("tullock_contest")

    def test_zero_effort_split(self) -> None:
        p, o = self._game.payoff_fn("effort_0", "effort_0")
        assert p == float(TULLOCK_PRIZE) / _TWO


class TestInspectionGame:
    _game = get_game("inspection_game")

    def test_violate_caught(self) -> None:
        p, _ = self._game.payoff_fn("violate", "inspect")
        assert p == float(-INSP_FINE)

    def test_violate_uncaught(self) -> None:
        p, _ = self._game.payoff_fn("violate", "no_inspect")
        assert p == float(INSP_VIOLATION_GAIN)


class TestRubinsteinBargaining:
    _game = get_game("rubinstein_bargaining")

    def test_compatible_demands(self) -> None:
        p, o = self._game.payoff_fn("demand_4", "demand_5")
        assert p == float(_FIVE - _ONE)
        assert o == float(_FIVE)


class TestDivideAndChoose:
    _game = get_game("divide_and_choose")

    def test_even_split(self) -> None:
        mid = DAC_ENDOWMENT // _TWO
        p, o = self._game.payoff_fn(f"split_{mid}", "choose_left")
        assert p == float(DAC_ENDOWMENT - mid)
        assert o == float(mid)
