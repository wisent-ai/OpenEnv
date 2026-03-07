"""Tests for signaling and contract theory games."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.ext.signaling_constants import (
    BQ_TOUGH_BEER_PAYOFF, BQ_NO_CHALLENGE_BONUS,
    SPENCE_HIGH_WAGE, SPENCE_EDU_COST_HIGH,
    CT_ALIGNED_MATCH,
    LEMON_MAX_PRICE,
    BP_GOOD_STATE_VALUE, BP_SAFE_PAYOFF,
)
from constant_definitions.ext.dynamic_constants import (
    MH_BASE_OUTPUT, MH_EFFORT_BOOST, MH_MAX_BONUS,
    SCR_HIGH_TYPE_VALUE, SCR_PREMIUM_PRICE,
    GE_PRODUCTIVITY_PER_EFFORT, GE_MAX_WAGE, GE_MAX_EFFORT,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_EIGHT = _FIVE + _THREE

_INFO_KEYS = [
    "beer_quiche", "spence_signaling", "cheap_talk",
    "lemon_market", "bayesian_persuasion",
    "moral_hazard", "screening", "gift_exchange",
]


class TestInfoRegistry:
    @pytest.mark.parametrize("key", _INFO_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _INFO_KEYS)
    def test_game_has_actions(self, key: str) -> None:
        game = get_game(key)
        assert len(game.actions) >= _TWO


class TestBeerQuiche:
    _game = get_game("beer_quiche")

    def test_beer_no_challenge(self) -> None:
        p, _ = self._game.payoff_fn("beer", "back_down")
        assert p == float(BQ_TOUGH_BEER_PAYOFF + BQ_NO_CHALLENGE_BONUS)

    def test_actions(self) -> None:
        assert "beer" in self._game.actions
        assert "quiche" in self._game.actions


class TestSpenceSignaling:
    _game = get_game("spence_signaling")

    def test_educate_high_wage(self) -> None:
        p, _ = self._game.payoff_fn("educate", "high_wage")
        assert p == float(SPENCE_HIGH_WAGE - SPENCE_EDU_COST_HIGH)


class TestCheapTalk:
    _game = get_game("cheap_talk")

    def test_aligned_communication(self) -> None:
        p, o = self._game.payoff_fn("signal_left", "act_left")
        assert p == float(CT_ALIGNED_MATCH)
        assert o == float(CT_ALIGNED_MATCH)


class TestLemonMarket:
    _game = get_game("lemon_market")

    def test_pass_yields_zero(self) -> None:
        p, o = self._game.payoff_fn("price_5", "pass")
        assert p == float(_ZERO)
        assert o == float(_ZERO)

    def test_action_count(self) -> None:
        assert len(self._game.actions) == LEMON_MAX_PRICE + _ONE


class TestBayesianPersuasion:
    _game = get_game("bayesian_persuasion")

    def test_reveal_and_act(self) -> None:
        p, o = self._game.payoff_fn("reveal", "act")
        assert p == float(BP_GOOD_STATE_VALUE)

    def test_conceal_and_safe(self) -> None:
        p, _ = self._game.payoff_fn("conceal", "safe")
        assert p == float(BP_SAFE_PAYOFF)


class TestMoralHazard:
    _game = get_game("moral_hazard")

    def test_bonus_zero_shirk(self) -> None:
        p, o = self._game.payoff_fn("bonus_0", "shirk")
        assert p == float(MH_BASE_OUTPUT)
        assert o == float(_ZERO)

    def test_bonus_with_work(self) -> None:
        p, o = self._game.payoff_fn(f"bonus_{_FIVE}", "work")
        expected_output = MH_BASE_OUTPUT + MH_EFFORT_BOOST
        assert p == float(expected_output - _FIVE)


class TestScreening:
    _game = get_game("screening")

    def test_premium_chosen(self) -> None:
        p, o = self._game.payoff_fn("offer_premium", "choose_premium")
        assert p == float(SCR_PREMIUM_PRICE)
        assert o == float(SCR_HIGH_TYPE_VALUE - SCR_PREMIUM_PRICE)


class TestGiftExchange:
    _game = get_game("gift_exchange")

    def test_zero_wage_zero_effort(self) -> None:
        p, o = self._game.payoff_fn("wage_0", "effort_0")
        assert p == float(_ZERO)
        assert o == float(_ZERO)

    def test_high_wage_high_effort(self) -> None:
        p, o = self._game.payoff_fn(f"wage_{_FIVE}", f"effort_{_FIVE}")
        assert p == float(GE_PRODUCTIVITY_PER_EFFORT * _FIVE - _FIVE)
