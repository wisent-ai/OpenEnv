"""Tests for stochastic and Bayesian game definitions."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.batch4.stochastic_constants import (
    SPD_CC, SPD_DC,
    RD_PAYOFF_DOMINANT, RD_RISK_DOMINANT, RD_MISCOORDINATION,
    TPG_ENDOWMENT, TPG_THRESHOLD, TPG_SUCCESS_BONUS,
    EPD_COOP_COOP, EPD_DEFECT_COOP, EPD_DEFECT_DEFECT,
    EPD_TFT_DEFECT, EPD_DEFECT_TFT,
)
from constant_definitions.batch4.bayesian_constants import (
    GG_ATTACK_ATTACK, GG_WAIT_WAIT,
    JV_CONVICT_CONVICT, JV_ACQUIT_ACQUIT, JV_SPLIT_VOTE,
    IC_SIGNAL_SIGNAL, IC_CROWD_CROWD,
    ASI_REVEAL_REVEAL, ASI_HIDE_HIDE,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE
_SIX = _FIVE + _ONE

_BATCH4A_KEYS = [
    "stochastic_pd", "risk_dominance", "threshold_public_goods",
    "evolutionary_pd", "global_game", "jury_voting",
    "information_cascade", "adverse_selection_insurance",
]


class TestBatch4ARegistry:
    @pytest.mark.parametrize("key", _BATCH4A_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _BATCH4A_KEYS)
    def test_game_callable(self, key: str) -> None:
        game = get_game(key)
        p, _ = game.payoff_fn(game.actions[_ZERO], game.actions[_ZERO])
        assert isinstance(p, float)


class TestStochasticPD:
    _game = get_game("stochastic_pd")

    def test_mutual_cooperate(self) -> None:
        p, o = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(SPD_CC) and o == float(SPD_CC)

    def test_defect_cooperate(self) -> None:
        p, _ = self._game.payoff_fn("defect", "cooperate")
        assert p == float(SPD_DC)


class TestRiskDominance:
    _game = get_game("risk_dominance")

    def test_payoff_dominant(self) -> None:
        p, o = self._game.payoff_fn("risky", "risky")
        assert p == float(RD_PAYOFF_DOMINANT)

    def test_risk_dominant(self) -> None:
        p, o = self._game.payoff_fn("safe", "safe")
        assert p == float(RD_RISK_DOMINANT)

    def test_miscoordination(self) -> None:
        p, _ = self._game.payoff_fn("risky", "safe")
        assert p == float(RD_MISCOORDINATION)

    def test_payoff_dominant_better(self) -> None:
        p_risky, _ = self._game.payoff_fn("risky", "risky")
        p_safe, _ = self._game.payoff_fn("safe", "safe")
        assert p_risky > p_safe


class TestThresholdPublicGoods:
    _game = get_game("threshold_public_goods")

    def test_actions_count(self) -> None:
        assert len(self._game.actions) == TPG_ENDOWMENT + _ONE

    def test_below_threshold(self) -> None:
        p, _ = self._game.payoff_fn("contribute_0", "contribute_0")
        assert p == float(TPG_ENDOWMENT)

    def test_at_threshold(self) -> None:
        p, _ = self._game.payoff_fn("contribute_3", "contribute_3")
        assert p == float(TPG_ENDOWMENT - _THREE + TPG_SUCCESS_BONUS)

    def test_above_threshold(self) -> None:
        p, _ = self._game.payoff_fn("contribute_5", "contribute_5")
        assert p == float(TPG_ENDOWMENT - _FIVE + TPG_SUCCESS_BONUS)


class TestEvolutionaryPD:
    _game = get_game("evolutionary_pd")

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_coop_coop(self) -> None:
        p, _ = self._game.payoff_fn("always_coop", "always_coop")
        assert p == float(EPD_COOP_COOP)

    def test_defect_coop(self) -> None:
        p, _ = self._game.payoff_fn("always_defect", "always_coop")
        assert p == float(EPD_DEFECT_COOP)

    def test_tft_vs_tft(self) -> None:
        p, _ = self._game.payoff_fn("tit_for_tat", "tit_for_tat")
        assert p == float(EPD_COOP_COOP)

    def test_defect_vs_tft(self) -> None:
        p, o = self._game.payoff_fn("always_defect", "tit_for_tat")
        assert p == float(EPD_DEFECT_TFT)
        assert o == float(EPD_TFT_DEFECT)


class TestGlobalGame:
    _game = get_game("global_game")

    def test_mutual_attack(self) -> None:
        p, _ = self._game.payoff_fn("attack", "attack")
        assert p == float(GG_ATTACK_ATTACK)

    def test_mutual_wait(self) -> None:
        p, _ = self._game.payoff_fn("wait", "wait")
        assert p == float(GG_WAIT_WAIT)


class TestJuryVoting:
    _game = get_game("jury_voting")

    def test_unanimous_convict(self) -> None:
        p, _ = self._game.payoff_fn("guilty", "guilty")
        assert p == float(JV_CONVICT_CONVICT)

    def test_unanimous_acquit(self) -> None:
        p, _ = self._game.payoff_fn("acquit", "acquit")
        assert p == float(JV_ACQUIT_ACQUIT)

    def test_split_vote(self) -> None:
        p, _ = self._game.payoff_fn("guilty", "acquit")
        assert p == float(JV_SPLIT_VOTE)


class TestInformationCascade:
    _game = get_game("information_cascade")

    def test_both_signal(self) -> None:
        p, _ = self._game.payoff_fn("follow_signal", "follow_signal")
        assert p == float(IC_SIGNAL_SIGNAL)

    def test_both_crowd(self) -> None:
        p, _ = self._game.payoff_fn("follow_crowd", "follow_crowd")
        assert p == float(IC_CROWD_CROWD)

    def test_signal_better_than_crowd(self) -> None:
        p_ss, _ = self._game.payoff_fn("follow_signal", "follow_signal")
        p_cc, _ = self._game.payoff_fn("follow_crowd", "follow_crowd")
        assert p_ss > p_cc


class TestAdverseSelection:
    _game = get_game("adverse_selection_insurance")

    def test_both_reveal(self) -> None:
        p, _ = self._game.payoff_fn("reveal_type", "reveal_type")
        assert p == float(ASI_REVEAL_REVEAL)

    def test_both_hide(self) -> None:
        p, _ = self._game.payoff_fn("hide_type", "hide_type")
        assert p == float(ASI_HIDE_HIDE)
