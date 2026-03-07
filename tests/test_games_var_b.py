"""Tests for classic dilemmas and extended generated games."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.var.classic_constants import (
    TD_BONUS, DOLLAR_PRIZE,
    MINO_WIN_PAYOFF, MINO_TIE_PAYOFF,
    RPSLS_WIN_PAYOFF, RPSLS_DRAW_PAYOFF,
)
from constant_definitions.var.generated_ext_constants import PCHK_RESOURCE

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_EIGHT = _FIVE + _THREE

_VAR_B_KEYS = [
    "travelers_dilemma", "dollar_auction", "unscrupulous_diner",
    "minority_game", "rpsls",
    "random_zero_sum_3x3", "random_coordination_3x3", "parameterized_chicken",
]


class TestVarBRegistry:
    @pytest.mark.parametrize("key", _VAR_B_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _VAR_B_KEYS)
    def test_game_callable(self, key: str) -> None:
        game = get_game(key)
        p, _ = game.payoff_fn(game.actions[_ZERO], game.actions[_ZERO])
        assert isinstance(p, float)


class TestTravelersDilemma:
    _game = get_game("travelers_dilemma")

    def test_equal_claims(self) -> None:
        p, o = self._game.payoff_fn("claim_5", "claim_5")
        assert p == float(_FIVE) and o == float(_FIVE)

    def test_lower_gets_bonus(self) -> None:
        p, o = self._game.payoff_fn("claim_3", "claim_5")
        assert p == float(_THREE + TD_BONUS)
        assert o == float(_THREE - TD_BONUS)


class TestDollarAuction:
    _game = get_game("dollar_auction")

    def test_higher_bid_wins(self) -> None:
        p, o = self._game.payoff_fn("bid_8", "bid_3")
        assert p == float(DOLLAR_PRIZE - _EIGHT)
        assert o == float(-_THREE)

    def test_zero_bids_split(self) -> None:
        p, _ = self._game.payoff_fn("bid_0", "bid_0")
        assert p == float(DOLLAR_PRIZE) / _TWO


class TestUnscrupulousDiner:
    _game = get_game("unscrupulous_diner")

    def test_both_cheap_positive(self) -> None:
        p, o = self._game.payoff_fn("order_cheap", "order_cheap")
        assert p == o and p > _ZERO

    def test_pd_structure(self) -> None:
        p_cc, _ = self._game.payoff_fn("order_cheap", "order_cheap")
        p_dc, _ = self._game.payoff_fn("order_expensive", "order_cheap")
        assert p_dc > p_cc


class TestMinorityGame:
    _game = get_game("minority_game")

    def test_same_choice_tie(self) -> None:
        p, _ = self._game.payoff_fn("choose_a", "choose_a")
        assert p == float(MINO_TIE_PAYOFF)

    def test_different_choice_win(self) -> None:
        p, _ = self._game.payoff_fn("choose_a", "choose_b")
        assert p == float(MINO_WIN_PAYOFF)

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE


class TestRPSLS:
    _game = get_game("rpsls")

    def test_five_actions(self) -> None:
        assert len(self._game.actions) == _FIVE

    def test_rock_beats_scissors(self) -> None:
        p, _ = self._game.payoff_fn("rock", "scissors")
        assert p == float(RPSLS_WIN_PAYOFF)

    def test_rock_beats_lizard(self) -> None:
        p, _ = self._game.payoff_fn("rock", "lizard")
        assert p == float(RPSLS_WIN_PAYOFF)

    def test_draw(self) -> None:
        p, _ = self._game.payoff_fn("spock", "spock")
        assert p == float(RPSLS_DRAW_PAYOFF)

    def test_zero_sum(self) -> None:
        for a in self._game.actions:
            for b in self._game.actions:
                p, o = self._game.payoff_fn(a, b)
                assert p + o == _ZERO


class TestRandomZeroSum:
    _game = get_game("random_zero_sum_3x3")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_zero_sum_property(self) -> None:
        for a in self._game.actions:
            for b in self._game.actions:
                p, o = self._game.payoff_fn(a, b)
                assert p + o == _ZERO


class TestRandomCoordination:
    _game = get_game("random_coordination_3x3")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_diagonal_bonus(self) -> None:
        diag = [self._game.payoff_fn(a, a)[_ZERO] for a in self._game.actions]
        off = [
            self._game.payoff_fn(a, b)[_ZERO]
            for a in self._game.actions for b in self._game.actions if a != b
        ]
        assert min(diag) > max(off)


class TestParameterizedChicken:
    _game = get_game("parameterized_chicken")

    def test_dove_dove_splits(self) -> None:
        p, o = self._game.payoff_fn("dove", "dove")
        assert p == float(PCHK_RESOURCE) / _TWO

    def test_hawk_hawk_negative(self) -> None:
        p, _ = self._game.payoff_fn("hawk", "hawk")
        assert p < _ZERO
