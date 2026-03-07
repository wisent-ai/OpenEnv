"""Tests for cooperative and dynamic games."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES, get_game
from constant_definitions.ext.cooperative_constants import (
    SHAPLEY_GRAND_COALITION_VALUE, SHAPLEY_SINGLE_VALUE,
    CORE_POT,
    WV_PASS_BENEFIT, WV_OPPOSITION_BONUS,
    SM_TOP_MATCH_PAYOFF,
    AV_PREFERRED_WIN, AV_DISLIKED_WIN,
)
from constant_definitions.ext.dynamic_constants import (
    BR_PATIENCE_REWARD, BR_BANK_FAIL_PAYOFF,
    GSH_STAG_PAYOFF, GSH_HARE_PAYOFF,
    BC_WIN_PAYOFF, BC_TIE_PAYOFF,
)
from constant_definitions.game_constants import PD_CC_PAYOFF, PD_DD_PAYOFF

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_SIX = _FIVE + _ONE

_COOP_KEYS = [
    "shapley_allocation", "core_divide_dollar", "weighted_voting",
    "stable_matching", "median_voter", "approval_voting",
    "bank_run", "global_stag_hunt", "beauty_contest",
    "hawk_dove_bourgeois", "finitely_repeated_pd", "markov_game",
]


class TestCoopRegistry:
    @pytest.mark.parametrize("key", _COOP_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _COOP_KEYS)
    def test_game_callable(self, key: str) -> None:
        game = get_game(key)
        a = game.actions
        p, o = game.payoff_fn(a[_ZERO], a[_ZERO])
        assert isinstance(p, float)


class TestShapleyAllocation:
    _game = get_game("shapley_allocation")

    def test_compatible_claims(self) -> None:
        p, o = self._game.payoff_fn("claim_5", "claim_5")
        assert p == float(_FIVE)
        assert o == float(_FIVE)

    def test_excessive_claims(self) -> None:
        p, o = self._game.payoff_fn("claim_8", "claim_8")
        assert p == float(SHAPLEY_SINGLE_VALUE)


class TestCoreDivideDollar:
    _game = get_game("core_divide_dollar")

    def test_feasible_split(self) -> None:
        p, o = self._game.payoff_fn("claim_4", "claim_6")
        assert p == float(_FIVE - _ONE)
        assert o == float(_SIX)

    def test_infeasible_split(self) -> None:
        p, o = self._game.payoff_fn("claim_6", "claim_6")
        assert p == float(_ZERO)


class TestWeightedVoting:
    _game = get_game("weighted_voting")

    def test_both_yes_passes(self) -> None:
        p, o = self._game.payoff_fn("vote_yes", "vote_yes")
        assert p == float(WV_PASS_BENEFIT)

    def test_both_no_fails(self) -> None:
        p, o = self._game.payoff_fn("vote_no", "vote_no")
        assert p == float(WV_OPPOSITION_BONUS)


class TestStableMatching:
    _game = get_game("stable_matching")

    def test_aligned_preferences(self) -> None:
        p, o = self._game.payoff_fn("rank_abc", "rank_abc")
        assert p == float(SM_TOP_MATCH_PAYOFF)

    def test_three_actions(self) -> None:
        assert len(self._game.actions) == _THREE


class TestApprovalVoting:
    _game = get_game("approval_voting")

    def test_same_approval_wins(self) -> None:
        p, o = self._game.payoff_fn("approve_a", "approve_a")
        assert p == float(AV_PREFERRED_WIN)

    def test_different_approvals(self) -> None:
        p, o = self._game.payoff_fn("approve_a", "approve_b")
        assert p == float(AV_DISLIKED_WIN)


class TestBankRun:
    _game = get_game("bank_run")

    def test_both_wait(self) -> None:
        p, o = self._game.payoff_fn("wait", "wait")
        assert p == float(BR_PATIENCE_REWARD)

    def test_both_withdraw(self) -> None:
        p, _ = self._game.payoff_fn("withdraw", "withdraw")
        assert p == float(BR_BANK_FAIL_PAYOFF)


class TestGlobalStagHunt:
    _game = get_game("global_stag_hunt")

    def test_mutual_stag(self) -> None:
        p, o = self._game.payoff_fn("stag", "stag")
        assert p == float(GSH_STAG_PAYOFF)

    def test_hare_safe(self) -> None:
        p, _ = self._game.payoff_fn("hare", "hare")
        assert p == float(GSH_HARE_PAYOFF)


class TestBeautyContest:
    _game = get_game("beauty_contest")

    def test_both_zero(self) -> None:
        p, o = self._game.payoff_fn("guess_0", "guess_0")
        assert p == float(BC_TIE_PAYOFF)

    def test_closer_wins(self) -> None:
        p, o = self._game.payoff_fn("guess_0", "guess_5")
        assert p == float(BC_WIN_PAYOFF)


class TestHawkDoveBourgeois:
    _game = get_game("hawk_dove_bourgeois")

    def test_three_strategies(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_dove_dove_positive(self) -> None:
        p, o = self._game.payoff_fn("dove", "dove")
        assert p > _ZERO
        assert o > _ZERO


class TestFinitelyRepeatedPD:
    _game = get_game("finitely_repeated_pd")

    def test_cooperate_cooperate(self) -> None:
        p, o = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(PD_CC_PAYOFF)

    def test_short_horizon(self) -> None:
        assert self._game.default_rounds == _FIVE


class TestMarkovGame:
    _game = get_game("markov_game")

    def test_defect_defect(self) -> None:
        p, o = self._game.payoff_fn("defect", "defect")
        assert p == float(PD_DD_PAYOFF)

    def test_long_horizon(self) -> None:
        assert self._game.default_rounds > _FIVE
