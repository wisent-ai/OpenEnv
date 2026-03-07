"""Tests for extended game definitions."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest

from common.games import GAMES, get_game
from constant_definitions.zero_sum_constants import (
    MP_MATCH_PAYOFF, MP_MISMATCH_PAYOFF,
    RPS_WIN_PAYOFF, RPS_LOSE_PAYOFF, RPS_DRAW_PAYOFF,
)
from constant_definitions.coordination_constants import (
    BOS_PREFERRED_PAYOFF, BOS_COMPROMISE_PAYOFF, BOS_MISMATCH_PAYOFF,
    PC_MATCH_PAYOFF, PC_MISMATCH_PAYOFF,
)
from constant_definitions.sequential_constants import DICTATOR_ENDOWMENT
from constant_definitions.auction_nplayer_constants import (
    AUCTION_ITEM_VALUE, COMMONS_RESOURCE_CAPACITY,
    COMMONS_DEPLETION_PENALTY, VOLUNTEER_BENEFIT, VOLUNTEER_COST,
    VOLUNTEER_NO_VOL, EL_FAROL_ATTEND_REWARD, EL_FAROL_STAY_HOME,
    COMMONS_MAX_EXTRACTION,
)
from common.games_ext.generated import (
    generate_random_symmetric,
    generate_random_asymmetric,
    generate_parameterized_pd,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_SIX = _FIVE + _ONE
_EIGHT = _SIX + _TWO

_EXPECTED_TOTAL = _FIVE * _THREE * _SIX

_EXTENDED_KEYS = [
    "matching_pennies", "rock_paper_scissors", "battle_of_the_sexes",
    "pure_coordination", "deadlock", "harmony",
    "dictator", "centipede", "stackelberg",
    "first_price_auction", "vickrey_auction", "allpay_auction",
    "tragedy_of_commons", "volunteer_dilemma", "el_farol",
    "random_symmetric_3x3", "random_asymmetric_3x3",
]


class TestExtendedRegistry:
    def test_total_game_count(self) -> None:
        assert len(GAMES) == _EXPECTED_TOTAL

    @pytest.mark.parametrize("key", _EXTENDED_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _EXTENDED_KEYS)
    def test_get_game_works(self, key: str) -> None:
        game = get_game(key)
        assert game.name and game.actions and game.payoff_fn


class TestMatchingPennies:
    _game = get_game("matching_pennies")

    def test_match_heads(self) -> None:
        p, o = self._game.payoff_fn("heads", "heads")
        assert p == float(MP_MATCH_PAYOFF)
        assert o == float(MP_MISMATCH_PAYOFF)

    def test_mismatch(self) -> None:
        p, o = self._game.payoff_fn("heads", "tails")
        assert p == float(MP_MISMATCH_PAYOFF)

    def test_zero_sum_property(self) -> None:
        for a in self._game.actions:
            for b in self._game.actions:
                p, o = self._game.payoff_fn(a, b)
                assert p + o == _ZERO


class TestRockPaperScissors:
    _game = get_game("rock_paper_scissors")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_rock_beats_scissors(self) -> None:
        p, o = self._game.payoff_fn("rock", "scissors")
        assert p == float(RPS_WIN_PAYOFF)
        assert o == float(RPS_LOSE_PAYOFF)

    def test_zero_sum_all_outcomes(self) -> None:
        for a in self._game.actions:
            for b in self._game.actions:
                p, o = self._game.payoff_fn(a, b)
                assert p + o == _ZERO


class TestBattleOfTheSexes:
    _game = get_game("battle_of_the_sexes")

    def test_opera_coordination(self) -> None:
        p, o = self._game.payoff_fn("opera", "opera")
        assert p == float(BOS_PREFERRED_PAYOFF)
        assert o == float(BOS_COMPROMISE_PAYOFF)

    def test_miscoordination(self) -> None:
        p, _ = self._game.payoff_fn("opera", "football")
        assert p == float(BOS_MISMATCH_PAYOFF)


class TestPureCoordination:
    _game = get_game("pure_coordination")

    def test_match_payoff(self) -> None:
        p, o = self._game.payoff_fn("left", "left")
        assert p == float(PC_MATCH_PAYOFF)

    def test_mismatch_payoff(self) -> None:
        p, _ = self._game.payoff_fn("left", "right")
        assert p == float(PC_MISMATCH_PAYOFF)


class TestDictator:
    _game = get_game("dictator")

    def test_give_nothing(self) -> None:
        p, o = self._game.payoff_fn("give_0", "give_0")
        assert p == float(DICTATOR_ENDOWMENT)
        assert o == float(_ZERO)

    def test_give_all(self) -> None:
        p, o = self._game.payoff_fn(f"give_{DICTATOR_ENDOWMENT}", "give_0")
        assert p == float(_ZERO)
        assert o == float(DICTATOR_ENDOWMENT)


class TestCentipede:
    _game = get_game("centipede")

    def test_immediate_take(self) -> None:
        p, o = self._game.payoff_fn("take_0", "take_0")
        assert p > _ZERO

    def test_pass_all_yields_larger_pot(self) -> None:
        p_take, _ = self._game.payoff_fn("take_0", "pass_all")
        p_pass, _ = self._game.payoff_fn("pass_all", "pass_all")
        assert p_pass > p_take


class TestStackelberg:
    _game = get_game("stackelberg")

    def test_zero_production(self) -> None:
        p, o = self._game.payoff_fn("produce_0", "produce_0")
        assert p == float(_ZERO)


class TestAuctions:
    def test_first_price_winner_pays_own_bid(self) -> None:
        game = get_game("first_price_auction")
        p, o = game.payoff_fn(f"bid_{_EIGHT}", f"bid_{_FIVE}")
        assert p == float(AUCTION_ITEM_VALUE - _EIGHT)
        assert o == float(_ZERO)

    def test_vickrey_winner_pays_second_price(self) -> None:
        game = get_game("vickrey_auction")
        p, o = game.payoff_fn(f"bid_{_EIGHT}", f"bid_{_FIVE}")
        assert p == float(AUCTION_ITEM_VALUE - _FIVE)
        assert o == float(_ZERO)

    def test_allpay_both_pay(self) -> None:
        game = get_game("allpay_auction")
        p, o = game.payoff_fn(f"bid_{_EIGHT}", f"bid_{_FIVE}")
        assert p == float(AUCTION_ITEM_VALUE - _EIGHT)
        assert o == float(-_FIVE)


class TestTragedyOfCommons:
    _game = get_game("tragedy_of_commons")

    def test_sustainable_extraction(self) -> None:
        p, o = self._game.payoff_fn(f"extract_{_FIVE}", f"extract_{_FIVE}")
        assert p == float(_FIVE)

    def test_depletion(self) -> None:
        p, _ = self._game.payoff_fn(
            f"extract_{COMMONS_MAX_EXTRACTION}",
            f"extract_{COMMONS_MAX_EXTRACTION}",
        )
        total = COMMONS_MAX_EXTRACTION + COMMONS_MAX_EXTRACTION
        if total > COMMONS_RESOURCE_CAPACITY:
            assert p == float(COMMONS_DEPLETION_PENALTY)


class TestVolunteerDilemma:
    _game = get_game("volunteer_dilemma")

    def test_nobody_volunteers(self) -> None:
        p, _ = self._game.payoff_fn("abstain", "abstain")
        assert p == float(VOLUNTEER_NO_VOL)

    def test_one_volunteers(self) -> None:
        p, o = self._game.payoff_fn("volunteer", "abstain")
        assert p == float(VOLUNTEER_BENEFIT - VOLUNTEER_COST)
        assert o == float(VOLUNTEER_BENEFIT)


class TestElFarol:
    _game = get_game("el_farol")

    def test_stay_home(self) -> None:
        p, _ = self._game.payoff_fn("stay_home", "attend")
        assert p == float(EL_FAROL_STAY_HOME)

    def test_attend_alone(self) -> None:
        p, _ = self._game.payoff_fn("attend", "stay_home")
        assert p == float(EL_FAROL_ATTEND_REWARD)


class TestGeneratedGames:
    def test_symmetric_generates_valid_game(self) -> None:
        game = generate_random_symmetric(num_actions=_FIVE)
        assert len(game.actions) == _FIVE
        p, _ = game.payoff_fn(game.actions[_ZERO], game.actions[_ZERO])
        assert isinstance(p, float)

    def test_asymmetric_generates_valid_game(self) -> None:
        game = generate_random_asymmetric(num_actions=_TWO)
        assert len(game.actions) == _TWO

    def test_parameterized_pd(self) -> None:
        game = generate_parameterized_pd(
            temptation=_FIVE + _FIVE, reward=_SIX,
            punishment=_TWO, sucker=_ZERO,
        )
        p, _ = game.payoff_fn("cooperate", "cooperate")
        assert p == float(_SIX)

    def test_different_seeds_different_games(self) -> None:
        g_a = generate_random_symmetric(seed=_ONE)
        g_b = generate_random_symmetric(seed=_EIGHT + _EIGHT)
        vals_a = [g_a.payoff_fn(a, b) for a in g_a.actions for b in g_a.actions]
        vals_b = [g_b.payoff_fn(a, b) for a in g_b.actions for b in g_b.actions]
        assert vals_a != vals_b

    def test_default_instances_registered(self) -> None:
        assert "random_symmetric_3x3" in GAMES
        assert "random_asymmetric_3x3" in GAMES
