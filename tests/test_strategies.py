"""Tests for opponent strategy implementations."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest

from constant_definitions.game_constants import (
    ULTIMATUM_FAIR_OFFER,
    ULTIMATUM_LOW_OFFER,
    PG_ENDOWMENT,
    PG_FAIR_CONTRIBUTION_NUMERATOR,
    PG_FAIR_CONTRIBUTION_DENOMINATOR,
    PG_FREE_RIDER_CONTRIBUTION,
)
from common.strategies import STRATEGIES, get_strategy

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_SIX = _FIVE + _ONE

_MATRIX_ACTIONS = ["cooperate", "defect"]
_STAG_ACTIONS = ["stag", "hare"]
_GAME_TYPE_MATRIX = "matrix"

_EXPECTED_STRATEGY_COUNT = _FIVE + _FIVE + _FIVE + _TWO

_ALL_STRATEGY_KEYS = [
    "random",
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "tit_for_two_tats",
    "grudger",
    "pavlov",
    "suspicious_tit_for_tat",
    "generous_tit_for_tat",
    "adaptive",
    "mixed",
    "ultimatum_fair",
    "ultimatum_low",
    "trust_fair",
    "trust_generous",
    "public_goods_fair",
    "public_goods_free_rider",
]


def _make_round(player: str, opponent: str) -> dict:
    """Build a single-round history entry."""
    return {"player_action": player, "opponent_action": opponent}


# ── registry tests ──────────────────────────────────────────────────────


class TestStrategyRegistry:
    """Ensure every expected strategy is registered and accessible."""

    def test_registry_count(self) -> None:
        assert len(STRATEGIES) == _EXPECTED_STRATEGY_COUNT

    @pytest.mark.parametrize("key", _ALL_STRATEGY_KEYS)
    def test_strategy_present(self, key: str) -> None:
        assert key in STRATEGIES

    @pytest.mark.parametrize("key", _ALL_STRATEGY_KEYS)
    def test_get_strategy_returns_same_object(self, key: str) -> None:
        assert get_strategy(key) is STRATEGIES[key]

    def test_unknown_strategy_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_strategy("does_not_exist")


# ── AlwaysCooperate ─────────────────────────────────────────────────────


class TestAlwaysCooperate:
    """AlwaysCooperate must always return the first action."""

    _strat = get_strategy("always_cooperate")

    def test_returns_first_action_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_returns_first_action_with_history(self) -> None:
        history = [_make_round("defect", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_works_with_stag_actions(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _STAG_ACTIONS, [])
        assert result == _STAG_ACTIONS[_ZERO]


# ── AlwaysDefect ────────────────────────────────────────────────────────


class TestAlwaysDefect:
    """AlwaysDefect must always return the second action."""

    _strat = get_strategy("always_defect")

    def test_returns_second_action_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ONE]

    def test_returns_second_action_with_history(self) -> None:
        history = [_make_round("cooperate", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ONE]

    def test_works_with_stag_actions(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _STAG_ACTIONS, [])
        assert result == _STAG_ACTIONS[_ONE]


# ── TitForTat ───────────────────────────────────────────────────────────


class TestTitForTat:
    """TitForTat cooperates initially, then mirrors the last move."""

    _strat = get_strategy("tit_for_tat")

    def test_cooperates_on_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_mirrors_opponent_defection(self) -> None:
        history = [_make_round("defect", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ONE]

    def test_mirrors_opponent_cooperation(self) -> None:
        history = [_make_round("cooperate", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_mirrors_latest_move_only(self) -> None:
        history = [
            _make_round("defect", "cooperate"),
            _make_round("cooperate", "defect"),
        ]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]


# ── Grudger ─────────────────────────────────────────────────────────────


class TestGrudger:
    """Grudger cooperates until the opponent defects, then always defects."""

    _strat = get_strategy("grudger")

    def test_cooperates_on_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_cooperates_while_opponent_cooperates(self) -> None:
        history = [_make_round("cooperate", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_defects_after_opponent_defection(self) -> None:
        history = [_make_round("defect", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ONE]

    def test_never_forgives(self) -> None:
        history = [
            _make_round("defect", "cooperate"),
            _make_round("cooperate", "defect"),
            _make_round("cooperate", "defect"),
        ]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ONE]


# ── Pavlov ──────────────────────────────────────────────────────────────


class TestPavlov:
    """Pavlov cooperates first; repeats if both chose the same, else switches."""

    _strat = get_strategy("pavlov")

    def test_cooperates_on_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_repeats_when_both_cooperated(self) -> None:
        history = [_make_round("cooperate", "cooperate")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_switches_when_both_defected(self) -> None:
        history = [_make_round("defect", "defect")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]

    def test_switches_on_mismatch(self) -> None:
        history = [_make_round("cooperate", "defect")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ONE]


# ── SuspiciousTitForTat ─────────────────────────────────────────────────


class TestSuspiciousTitForTat:
    """Suspicious TFT defects first, then mirrors."""

    _strat = get_strategy("suspicious_tit_for_tat")

    def test_defects_on_empty_history(self) -> None:
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert result == _MATRIX_ACTIONS[_ONE]

    def test_mirrors_cooperation(self) -> None:
        history = [_make_round("cooperate", "defect")]
        result = self._strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, history)
        assert result == _MATRIX_ACTIONS[_ZERO]


# ── Ultimatum strategies ────────────────────────────────────────────────


class TestUltimatumStrategies:
    """Ultimatum-specific strategies must produce valid action strings."""

    _offer_actions = [f"offer_{i}" for i in range(ULTIMATUM_FAIR_OFFER + _ONE)]

    def test_fair_strategy_offers_fair_amount(self) -> None:
        strat = get_strategy("ultimatum_fair")
        result = strat.choose_action("ultimatum", self._offer_actions, [])
        assert result == f"offer_{ULTIMATUM_FAIR_OFFER}"

    def test_low_strategy_offers_low_amount(self) -> None:
        strat = get_strategy("ultimatum_low")
        result = strat.choose_action("ultimatum", self._offer_actions, [])
        assert result == f"offer_{ULTIMATUM_LOW_OFFER}"

    def test_fair_strategy_returns_valid_action(self) -> None:
        strat = get_strategy("ultimatum_fair")
        result = strat.choose_action("ultimatum", self._offer_actions, [])
        assert result in self._offer_actions


# ── Public Goods strategies ─────────────────────────────────────────────


class TestPublicGoodsStrategies:
    """Public Goods strategies must return valid contribution strings."""

    _pg_actions = [f"contribute_{i}" for i in range(PG_ENDOWMENT + _ONE)]

    def test_fair_strategy_contributes_half(self) -> None:
        strat = get_strategy("public_goods_fair")
        result = strat.choose_action("public_goods", self._pg_actions, [])
        expected_amount = (
            PG_ENDOWMENT * PG_FAIR_CONTRIBUTION_NUMERATOR
            // PG_FAIR_CONTRIBUTION_DENOMINATOR
        )
        assert result == f"contribute_{expected_amount}"

    def test_free_rider_contributes_minimum(self) -> None:
        strat = get_strategy("public_goods_free_rider")
        result = strat.choose_action("public_goods", self._pg_actions, [])
        assert result == f"contribute_{PG_FREE_RIDER_CONTRIBUTION}"


# ── choose_action interface ─────────────────────────────────────────────


class TestChooseActionInterface:
    """Every strategy must implement choose_action correctly."""

    @pytest.mark.parametrize("key", _ALL_STRATEGY_KEYS[:_FIVE + _SIX])
    def test_choose_action_returns_string(self, key: str) -> None:
        strat = get_strategy(key)
        result = strat.choose_action(_GAME_TYPE_MATRIX, _MATRIX_ACTIONS, [])
        assert isinstance(result, str)
