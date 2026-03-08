"""Tests for the Bayesian (noisy) variant transforms."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES
from common.variants import (
    apply_cheap_talk, apply_noisy_actions, apply_noisy_payoffs,
)
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
)
from constant_definitions.var.pd_variant_constants import (
    VARIANT_CHEAP_TALK, VARIANT_NOISY_ACTIONS, VARIANT_NOISY_PAYOFFS,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_FOUR = _TWO + _TWO
_NOISY_TRIALS = _FOUR * _FOUR * _FOUR

_PD = GAMES["prisoners_dilemma"]


class TestApplyNoisyActionsPD:
    _game = apply_noisy_actions(_PD, base_key="prisoners_dilemma")

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_NOISY_ACTIONS,)
        assert self._game.base_game_key == "prisoners_dilemma"

    def test_actions_unchanged(self) -> None:
        assert self._game.actions == _PD.actions

    def test_payoffs_are_valid_action_pairs(self) -> None:
        valid = {
            (float(PD_CC_PAYOFF), float(PD_CC_PAYOFF)),
            (float(PD_CD_PAYOFF), float(PD_DC_PAYOFF)),
            (float(PD_DC_PAYOFF), float(PD_CD_PAYOFF)),
            (float(PD_DD_PAYOFF), float(PD_DD_PAYOFF)),
        }
        for _ in range(_NOISY_TRIALS):
            result = self._game.payoff_fn("cooperate", "cooperate")
            assert result in valid


class TestApplyNoisyPayoffsPD:
    _game = apply_noisy_payoffs(_PD, base_key="prisoners_dilemma")

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_NOISY_PAYOFFS,)
        assert self._game.base_game_key == "prisoners_dilemma"

    def test_actions_unchanged(self) -> None:
        assert self._game.actions == _PD.actions

    def test_payoffs_close_to_base(self) -> None:
        base_p = float(PD_CC_PAYOFF)
        tolerance = float(_FOUR)
        for _ in range(_NOISY_TRIALS):
            p, o = self._game.payoff_fn("cooperate", "cooperate")
            assert abs(p - base_p) < tolerance
            assert abs(o - base_p) < tolerance


class TestNoisyComposition:
    _game = apply_noisy_payoffs(
        apply_noisy_actions(_PD, base_key="prisoners_dilemma"),
        base_key="prisoners_dilemma",
    )

    def test_stacked_variants(self) -> None:
        assert self._game.applied_variants == (
            VARIANT_NOISY_ACTIONS, VARIANT_NOISY_PAYOFFS,
        )

    def test_payoff_returns_floats(self) -> None:
        p, o = self._game.payoff_fn("cooperate", "defect")
        assert isinstance(p, float)
        assert isinstance(o, float)

    def test_compose_with_existing_variant(self) -> None:
        game = apply_noisy_actions(
            apply_cheap_talk(_PD, base_key="prisoners_dilemma"),
            base_key="prisoners_dilemma",
        )
        assert game.applied_variants == (
            VARIANT_CHEAP_TALK, VARIANT_NOISY_ACTIONS,
        )
