"""Tests for the composable variant system."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES, get_game
from common.variants import (
    apply_cheap_talk, apply_exit,
    apply_binding_commitment, compose_game,
)
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
    SH_SS_PAYOFF, SH_SH_PAYOFF, SH_HS_PAYOFF, SH_HH_PAYOFF,
    HD_HH_PAYOFF, HD_HD_PAYOFF, HD_DH_PAYOFF, HD_DD_PAYOFF,
)
from constant_definitions.var.pd_variant_constants import (
    OPD_EXIT_PAYOFF,
    VARIANT_CHEAP_TALK, VARIANT_EXIT, VARIANT_BINDING_COMMITMENT,
)
from constant_definitions.var.communication_constants import COMMIT_COST

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE

_PD = GAMES["prisoners_dilemma"]
_SH = GAMES["stag_hunt"]
_HD = GAMES["hawk_dove"]


class TestApplyCheapTalkPD:
    _game = apply_cheap_talk(_PD, base_key="prisoners_dilemma")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _FOUR

    def test_action_names(self) -> None:
        expected = [
            "msg_cooperate_cooperate", "msg_cooperate_defect",
            "msg_defect_cooperate", "msg_defect_defect",
        ]
        assert self._game.actions == expected

    def test_honest_cooperation_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "msg_cooperate_cooperate", "msg_cooperate_cooperate",
        )
        assert p == float(PD_CC_PAYOFF)
        assert o == float(PD_CC_PAYOFF)

    def test_lying_defection_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "msg_cooperate_defect", "msg_cooperate_cooperate",
        )
        assert p == float(PD_DC_PAYOFF)
        assert o == float(PD_CD_PAYOFF)

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_CHEAP_TALK,)
        assert self._game.base_game_key == "prisoners_dilemma"


class TestApplyCheapTalkStagHunt:
    _game = apply_cheap_talk(_SH, base_key="stag_hunt")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _FOUR

    def test_action_names(self) -> None:
        expected = [
            "msg_stag_stag", "msg_stag_hare",
            "msg_hare_stag", "msg_hare_hare",
        ]
        assert self._game.actions == expected

    def test_stag_stag_payoff(self) -> None:
        p, o = self._game.payoff_fn("msg_stag_stag", "msg_hare_stag")
        assert p == float(SH_SS_PAYOFF)
        assert o == float(SH_SS_PAYOFF)

    def test_stag_hare_payoff(self) -> None:
        p, o = self._game.payoff_fn("msg_stag_stag", "msg_stag_hare")
        assert p == float(SH_SH_PAYOFF)
        assert o == float(SH_HS_PAYOFF)


class TestApplyCheapTalkHawkDove:
    _game = apply_cheap_talk(_HD, base_key="hawk_dove")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _FOUR

    def test_hawk_hawk_payoff(self) -> None:
        p, o = self._game.payoff_fn("msg_dove_hawk", "msg_hawk_hawk")
        assert p == float(HD_HH_PAYOFF)
        assert o == float(HD_HH_PAYOFF)


class TestApplyExitPD:
    _game = apply_exit(_PD, base_key="prisoners_dilemma")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_exit_in_actions(self) -> None:
        assert "exit" in self._game.actions

    def test_exit_payoff(self) -> None:
        p, o = self._game.payoff_fn("exit", "defect")
        assert p == float(OPD_EXIT_PAYOFF)
        assert o == float(OPD_EXIT_PAYOFF)

    def test_base_payoff_preserved(self) -> None:
        p, o = self._game.payoff_fn("cooperate", "cooperate")
        assert p == float(PD_CC_PAYOFF)
        assert o == float(PD_CC_PAYOFF)

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_EXIT,)
        assert self._game.base_game_key == "prisoners_dilemma"


class TestApplyExitStagHunt:
    _game = apply_exit(_SH, base_key="stag_hunt")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_exit_payoff(self) -> None:
        p, o = self._game.payoff_fn("exit", "stag")
        assert p == float(OPD_EXIT_PAYOFF)
        assert o == float(OPD_EXIT_PAYOFF)

    def test_stag_stag_preserved(self) -> None:
        p, o = self._game.payoff_fn("stag", "stag")
        assert p == float(SH_SS_PAYOFF)


class TestApplyBindingCommitmentPD:
    _game = apply_binding_commitment(_PD, base_key="prisoners_dilemma")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _THREE

    def test_action_names(self) -> None:
        expected = ["commit_cooperate", "free_cooperate", "free_defect"]
        assert self._game.actions == expected

    def test_commit_commit_cost(self) -> None:
        p, o = self._game.payoff_fn("commit_cooperate", "commit_cooperate")
        cost = float(COMMIT_COST)
        assert p == float(PD_CC_PAYOFF) - cost
        assert o == float(PD_CC_PAYOFF) - cost

    def test_free_defect_vs_free_cooperate(self) -> None:
        p, _ = self._game.payoff_fn("free_defect", "free_cooperate")
        assert p == float(PD_DC_PAYOFF)

    def test_commit_vs_free_defect(self) -> None:
        p, o = self._game.payoff_fn("commit_cooperate", "free_defect")
        cost = float(COMMIT_COST)
        assert p == float(PD_CD_PAYOFF) - cost
        assert o == float(PD_DC_PAYOFF)

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_BINDING_COMMITMENT,)
        assert self._game.base_game_key == "prisoners_dilemma"


class TestApplyBindingCommitmentStagHunt:
    _game = apply_binding_commitment(_SH, base_key="stag_hunt")

    def test_action_names(self) -> None:
        expected = ["commit_stag", "free_stag", "free_hare"]
        assert self._game.actions == expected

    def test_commit_stag_payoff(self) -> None:
        p, o = self._game.payoff_fn("commit_stag", "free_stag")
        cost = float(COMMIT_COST)
        assert p == float(SH_SS_PAYOFF) - cost
        assert o == float(SH_SS_PAYOFF)


class TestComposeGame:
    def test_single_variant(self) -> None:
        game = compose_game("prisoners_dilemma", "cheap_talk")
        assert len(game.actions) == _FOUR
        assert game.applied_variants == (VARIANT_CHEAP_TALK,)

    def test_multiple_variants(self) -> None:
        game = compose_game("stag_hunt", "cheap_talk", "exit")
        assert "exit" in game.actions
        ct_count = _TWO * _TWO
        assert len(game.actions) == ct_count + _ONE
        assert game.applied_variants == (VARIANT_CHEAP_TALK, VARIANT_EXIT)
        assert game.base_game_key == "stag_hunt"


class TestVariantComposition:
    _game = apply_exit(
        apply_cheap_talk(_PD, base_key="prisoners_dilemma"),
        base_key="prisoners_dilemma",
    )

    def test_actions_include_exit(self) -> None:
        assert "exit" in self._game.actions

    def test_cheap_talk_actions_preserved(self) -> None:
        assert "msg_cooperate_cooperate" in self._game.actions

    def test_total_action_count(self) -> None:
        assert len(self._game.actions) == _FOUR + _ONE

    def test_exit_payoff_works(self) -> None:
        p, o = self._game.payoff_fn("exit", "msg_cooperate_cooperate")
        assert p == float(OPD_EXIT_PAYOFF)
        assert o == float(OPD_EXIT_PAYOFF)

    def test_base_payoff_works(self) -> None:
        p, o = self._game.payoff_fn(
            "msg_cooperate_cooperate", "msg_defect_cooperate",
        )
        assert p == float(PD_CC_PAYOFF)
        assert o == float(PD_CC_PAYOFF)

    def test_stacked_variants(self) -> None:
        assert self._game.applied_variants == (
            VARIANT_CHEAP_TALK, VARIANT_EXIT,
        )


class TestRegisteredGamesMatchComposed:
    def test_optional_pd_actions(self) -> None:
        reg = get_game("optional_pd")
        assert reg.actions == ["cooperate", "defect", "exit"]

    def test_optional_pd_payoff(self) -> None:
        reg = get_game("optional_pd")
        p, o = reg.payoff_fn("exit", "cooperate")
        assert p == float(OPD_EXIT_PAYOFF)
        assert o == float(OPD_EXIT_PAYOFF)

    def test_cheap_talk_pd_actions(self) -> None:
        reg = get_game("cheap_talk_pd")
        expected = [
            "msg_cooperate_cooperate", "msg_cooperate_defect",
            "msg_defect_cooperate", "msg_defect_defect",
        ]
        assert reg.actions == expected

    def test_binding_commitment_actions(self) -> None:
        reg = get_game("binding_commitment")
        expected = ["commit_cooperate", "free_cooperate", "free_defect"]
        assert reg.actions == expected
