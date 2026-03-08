"""Tests for PD variants, communication, and infinite games."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.var.pd_variant_constants import (
    OPD_EXIT_PAYOFF,
    APD_A_REWARD, APD_B_PUNISHMENT,
    DONATION_BENEFIT, DONATION_COST,
    FOF_SHARE_PAYOFF,
    PW_DISARM_DISARM, PW_ARM_ARM,
)
from constant_definitions.var.communication_constants import (
    CTPD_REWARD, CTPD_TEMPTATION,
    CE_FOLLOW_FOLLOW, CE_DEVIATE_DEVIATE,
    FP_MATCH_PAYOFF, FP_MISMATCH_PAYOFF,
    MG_ACCEPT_ACCEPT, MG_REJECT_REJECT,
)
from constant_definitions.var.infinite_constants import (
    DPD_REWARD, DPD_DEFAULT_ROUNDS,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE

_VAR_A_KEYS = [
    "optional_pd", "asymmetric_pd", "donation_game", "friend_or_foe",
    "peace_war", "cheap_talk_pd", "binding_commitment",
    "correlated_equilibrium", "focal_point", "mediated_game",
    "continuous_pd", "discounted_pd",
]


class TestVarARegistry:
    @pytest.mark.parametrize("key", _VAR_A_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _VAR_A_KEYS)
    def test_game_callable(self, key: str) -> None:
        game = get_game(key)
        p, o = game.payoff_fn(game.actions[_ZERO], game.actions[_ZERO])
        assert isinstance(p, float)


class TestOptionalPD:
    _game = get_game("optional_pd")

    def test_exit_gives_safe_payoff(self) -> None:
        p, o = self._game.payoff_fn("exit", "defect")
        assert p == float(OPD_EXIT_PAYOFF) and o == float(OPD_EXIT_PAYOFF)

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE


class TestAsymmetricPD:
    _game = get_game("asymmetric_pd")

    def test_mutual_cooperation(self) -> None:
        p, _ = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(APD_A_REWARD)

    def test_mutual_defection_asymmetric(self) -> None:
        _, o = self._game.payoff_fn("defect", "defect")
        assert o == float(APD_B_PUNISHMENT)


class TestDonationGame:
    _game = get_game("donation_game")

    def test_mutual_donation(self) -> None:
        p, _ = self._game.payoff_fn("donate", "donate")
        assert p == float(DONATION_BENEFIT - DONATION_COST)

    def test_keep_keep(self) -> None:
        p, _ = self._game.payoff_fn("keep", "keep")
        assert p == float(_ZERO)


class TestFriendOrFoe:
    _game = get_game("friend_or_foe")

    def test_both_friend(self) -> None:
        p, _ = self._game.payoff_fn("friend", "friend")
        assert p == float(FOF_SHARE_PAYOFF)

    def test_both_foe_zero(self) -> None:
        p, o = self._game.payoff_fn("foe", "foe")
        assert p == float(_ZERO) and o == float(_ZERO)


class TestPeaceWar:
    _game = get_game("peace_war")

    def test_mutual_disarmament(self) -> None:
        p, _ = self._game.payoff_fn("disarm", "disarm")
        assert p == float(PW_DISARM_DISARM)

    def test_mutual_arming(self) -> None:
        p, _ = self._game.payoff_fn("arm", "arm")
        assert p == float(PW_ARM_ARM)


class TestCheapTalkPD:
    _game = get_game("cheap_talk_pd")

    def test_honest_cooperation(self) -> None:
        p, _ = self._game.payoff_fn("msg_cooperate_cooperate", "msg_cooperate_cooperate")
        assert p == float(CTPD_REWARD)

    def test_lying_defection(self) -> None:
        p, _ = self._game.payoff_fn("msg_cooperate_defect", "msg_cooperate_cooperate")
        assert p == float(CTPD_TEMPTATION)

    def test_four_actions(self) -> None:
        assert len(self._game.actions) == _FOUR


class TestBindingCommitment:
    _game = get_game("binding_commitment")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_free_defect_dominates(self) -> None:
        p_c, _ = self._game.payoff_fn("free_cooperate", "free_cooperate")
        p_d, _ = self._game.payoff_fn("free_defect", "free_cooperate")
        assert p_d > p_c


class TestCorrelatedEquilibrium:
    _game = get_game("correlated_equilibrium")

    def test_both_follow(self) -> None:
        p, _ = self._game.payoff_fn("follow", "follow")
        assert p == float(CE_FOLLOW_FOLLOW)

    def test_both_deviate(self) -> None:
        p, _ = self._game.payoff_fn("deviate", "deviate")
        assert p == float(CE_DEVIATE_DEVIATE)


class TestFocalPoint:
    _game = get_game("focal_point")

    def test_match(self) -> None:
        p, _ = self._game.payoff_fn("choose_red", "choose_red")
        assert p == float(FP_MATCH_PAYOFF)

    def test_mismatch(self) -> None:
        p, _ = self._game.payoff_fn("choose_red", "choose_blue")
        assert p == float(FP_MISMATCH_PAYOFF)

    def test_four_actions(self) -> None:
        assert len(self._game.actions) == _FOUR


class TestMediatedGame:
    _game = get_game("mediated_game")

    def test_both_accept(self) -> None:
        p, _ = self._game.payoff_fn("accept", "accept")
        assert p == float(MG_ACCEPT_ACCEPT)

    def test_both_reject(self) -> None:
        p, _ = self._game.payoff_fn("reject", "reject")
        assert p == float(MG_REJECT_REJECT)


class TestContinuousPD:
    _game = get_game("continuous_pd")

    def test_zero_zero(self) -> None:
        p, o = self._game.payoff_fn("level_0", "level_0")
        assert p == float(_ZERO) and o == float(_ZERO)

    def test_symmetric(self) -> None:
        p, o = self._game.payoff_fn("level_5", "level_5")
        assert p == o

    def test_many_actions(self) -> None:
        assert len(self._game.actions) > _FIVE


class TestDiscountedPD:
    _game = get_game("discounted_pd")

    def test_cooperate_cooperate(self) -> None:
        p, _ = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(DPD_REWARD)

    def test_long_horizon(self) -> None:
        assert self._game.default_rounds == DPD_DEFAULT_ROUNDS
