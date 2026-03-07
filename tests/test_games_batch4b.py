"""Tests for network and advanced market game definitions."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.batch4.network_constants import (
    SG_DEFEND_SUCCESS, SG_ATTACK_FAIL,
    LF_MUTUAL_CONNECT, LF_MUTUAL_ISOLATE,
    TWP_CC, TWP_DD, TWP_PP,
    DG_EARLY_EARLY, DG_LATE_LATE, DG_EARLY_LATE,
)
from constant_definitions.batch4.advanced_constants import (
    PRE_EARLY_EARLY, PRE_EARLY_LATE, PRE_OUT_PAYOFF,
    WOG_LARGE_LARGE, WOG_LARGE_SMALL, WOG_NO_GIFT,
    PS_SAVE_PAYOFF, PS_SCORE_PAYOFF, PS_CENTER_BONUS,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE

_BATCH4B_KEYS = [
    "security_game", "link_formation", "trust_with_punishment",
    "dueling_game", "preemption_game", "war_of_gifts", "penalty_shootout",
]


class TestBatch4BRegistry:
    @pytest.mark.parametrize("key", _BATCH4B_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _BATCH4B_KEYS)
    def test_game_callable(self, key: str) -> None:
        game = get_game(key)
        p, _ = game.payoff_fn(game.actions[_ZERO], game.actions[_ZERO])
        assert isinstance(p, float)


class TestSecurityGame:
    _game = get_game("security_game")

    def test_defend_success(self) -> None:
        p, o = self._game.payoff_fn("target_a", "target_a")
        assert p == float(SG_DEFEND_SUCCESS)
        assert o == float(SG_ATTACK_FAIL)

    def test_symmetric_structure(self) -> None:
        p_aa, o_aa = self._game.payoff_fn("target_a", "target_a")
        p_bb, o_bb = self._game.payoff_fn("target_b", "target_b")
        assert p_aa == p_bb and o_aa == o_bb


class TestLinkFormation:
    _game = get_game("link_formation")

    def test_mutual_connect(self) -> None:
        p, o = self._game.payoff_fn("connect", "connect")
        assert p == float(LF_MUTUAL_CONNECT) and o == float(LF_MUTUAL_CONNECT)

    def test_mutual_isolate(self) -> None:
        p, o = self._game.payoff_fn("isolate", "isolate")
        assert p == float(LF_MUTUAL_ISOLATE)

    def test_unilateral_connect_costly(self) -> None:
        p, _ = self._game.payoff_fn("connect", "isolate")
        assert p < _ZERO


class TestTrustWithPunishment:
    _game = get_game("trust_with_punishment")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_cooperate_cooperate(self) -> None:
        p, _ = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(TWP_CC)

    def test_defect_defect(self) -> None:
        p, _ = self._game.payoff_fn("defect", "defect")
        assert p == float(TWP_DD)

    def test_punish_punish(self) -> None:
        p, _ = self._game.payoff_fn("punish", "punish")
        assert p == float(TWP_PP)

    def test_punishment_costly(self) -> None:
        p_pp, _ = self._game.payoff_fn("punish", "punish")
        p_dd, _ = self._game.payoff_fn("defect", "defect")
        assert p_pp < p_dd


class TestDuelingGame:
    _game = get_game("dueling_game")

    def test_mutual_early(self) -> None:
        p, _ = self._game.payoff_fn("fire_early", "fire_early")
        assert p == float(DG_EARLY_EARLY)

    def test_mutual_late(self) -> None:
        p, _ = self._game.payoff_fn("fire_late", "fire_late")
        assert p == float(DG_LATE_LATE)

    def test_early_beats_late(self) -> None:
        p, _ = self._game.payoff_fn("fire_early", "fire_late")
        assert p == float(DG_EARLY_LATE)


class TestPreemptionGame:
    _game = get_game("preemption_game")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_both_early(self) -> None:
        p, _ = self._game.payoff_fn("enter_early", "enter_early")
        assert p == float(PRE_EARLY_EARLY)

    def test_first_mover_advantage(self) -> None:
        p, _ = self._game.payoff_fn("enter_early", "enter_late")
        assert p == float(PRE_EARLY_LATE)

    def test_stay_out_safe(self) -> None:
        p, _ = self._game.payoff_fn("stay_out", "stay_out")
        assert p == float(PRE_OUT_PAYOFF)


class TestWarOfGifts:
    _game = get_game("war_of_gifts")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_mutual_large(self) -> None:
        p, _ = self._game.payoff_fn("gift_large", "gift_large")
        assert p == float(WOG_LARGE_LARGE)

    def test_large_beats_small(self) -> None:
        p, o = self._game.payoff_fn("gift_large", "gift_small")
        assert p == float(WOG_LARGE_SMALL)
        assert o == _ZERO

    def test_no_gift_safe(self) -> None:
        p, _ = self._game.payoff_fn("no_gift", "no_gift")
        assert p == float(WOG_NO_GIFT)


class TestPenaltyShootout:
    _game = get_game("penalty_shootout")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_save(self) -> None:
        p, o = self._game.payoff_fn("left", "left")
        assert p == float(PS_SAVE_PAYOFF)
        assert o == float(-PS_SAVE_PAYOFF)

    def test_goal(self) -> None:
        p, _ = self._game.payoff_fn("left", "right")
        assert p == float(PS_SCORE_PAYOFF)

    def test_center_bonus(self) -> None:
        p, _ = self._game.payoff_fn("center", "left")
        assert p == float(PS_SCORE_PAYOFF + PS_CENTER_BONUS)

    def test_zero_sum(self) -> None:
        for a in self._game.actions:
            for b in self._game.actions:
                p, o = self._game.payoff_fn(a, b)
                assert p + o == _ZERO
