"""Tests for the game configuration registry and payoff functions."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest

from constant_definitions.game_constants import (
    PD_CC_PAYOFF,
    PD_CD_PAYOFF,
    PD_DC_PAYOFF,
    PD_DD_PAYOFF,
    SH_SS_PAYOFF,
    SH_SH_PAYOFF,
    SH_HS_PAYOFF,
    SH_HH_PAYOFF,
    HD_HH_PAYOFF,
    HD_HD_PAYOFF,
    HD_DH_PAYOFF,
    HD_DD_PAYOFF,
    ULTIMATUM_POT,
    TRUST_MULTIPLIER,
    TRUST_ENDOWMENT,
    PG_MULTIPLIER_NUMERATOR,
    PG_MULTIPLIER_DENOMINATOR,
    PG_ENDOWMENT,
    PG_DEFAULT_NUM_PLAYERS,
    DEFAULT_NUM_ROUNDS,
    SINGLE_SHOT_ROUNDS,
)
from common.games import GAMES, GameConfig, get_game

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE
_SIX = _FIVE + _ONE

_EXPECTED_GAME_COUNT = _SIX * _FOUR * _FOUR + _THREE

_ALL_GAME_KEYS = [
    "prisoners_dilemma",
    "stag_hunt",
    "hawk_dove",
    "ultimatum",
    "trust",
    "public_goods",
]


# ── registry tests ──────────────────────────────────────────────────────


class TestGameRegistry:
    """Ensure the GAMES registry contains every expected entry."""

    def test_registry_has_correct_number_of_games(self) -> None:
        assert len(GAMES) == _EXPECTED_GAME_COUNT

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_game_present_in_registry(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_get_game_returns_game_config(self, key: str) -> None:
        cfg = get_game(key)
        assert isinstance(cfg, GameConfig)

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_get_game_matches_registry(self, key: str) -> None:
        assert get_game(key) is GAMES[key]

    def test_invalid_game_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_game("nonexistent_game")


# ── Prisoner's Dilemma payoffs ──────────────────────────────────────────


class TestPrisonersDilemmaPayoffs:
    """Verify each cell in the PD payoff matrix."""

    _payoff = staticmethod(get_game("prisoners_dilemma").payoff_fn)

    def test_cooperate_cooperate(self) -> None:
        p, o = self._payoff("cooperate", "cooperate")
        assert p == float(PD_CC_PAYOFF)
        assert o == float(PD_CC_PAYOFF)

    def test_cooperate_defect(self) -> None:
        p, o = self._payoff("cooperate", "defect")
        assert p == float(PD_CD_PAYOFF)
        assert o == float(PD_DC_PAYOFF)

    def test_defect_cooperate(self) -> None:
        p, o = self._payoff("defect", "cooperate")
        assert p == float(PD_DC_PAYOFF)
        assert o == float(PD_CD_PAYOFF)

    def test_defect_defect(self) -> None:
        p, o = self._payoff("defect", "defect")
        assert p == float(PD_DD_PAYOFF)
        assert o == float(PD_DD_PAYOFF)

    def test_default_rounds(self) -> None:
        assert get_game("prisoners_dilemma").default_rounds == DEFAULT_NUM_ROUNDS


# ── Stag Hunt payoffs ───────────────────────────────────────────────────


class TestStagHuntPayoffs:
    """Verify each cell in the Stag Hunt payoff matrix."""

    _payoff = staticmethod(get_game("stag_hunt").payoff_fn)

    def test_stag_stag(self) -> None:
        p, o = self._payoff("stag", "stag")
        assert p == float(SH_SS_PAYOFF)
        assert o == float(SH_SS_PAYOFF)

    def test_stag_hare(self) -> None:
        p, o = self._payoff("stag", "hare")
        assert p == float(SH_SH_PAYOFF)
        assert o == float(SH_HS_PAYOFF)

    def test_hare_stag(self) -> None:
        p, o = self._payoff("hare", "stag")
        assert p == float(SH_HS_PAYOFF)
        assert o == float(SH_SH_PAYOFF)

    def test_hare_hare(self) -> None:
        p, o = self._payoff("hare", "hare")
        assert p == float(SH_HH_PAYOFF)
        assert o == float(SH_HH_PAYOFF)


# ── Hawk-Dove payoffs ───────────────────────────────────────────────────


class TestHawkDovePayoffs:
    """Verify each cell in the Hawk-Dove payoff matrix."""

    _payoff = staticmethod(get_game("hawk_dove").payoff_fn)

    def test_hawk_hawk(self) -> None:
        p, o = self._payoff("hawk", "hawk")
        assert p == float(HD_HH_PAYOFF)
        assert o == float(HD_HH_PAYOFF)

    def test_hawk_dove(self) -> None:
        p, o = self._payoff("hawk", "dove")
        assert p == float(HD_HD_PAYOFF)
        assert o == float(HD_DH_PAYOFF)

    def test_dove_hawk(self) -> None:
        p, o = self._payoff("dove", "hawk")
        assert p == float(HD_DH_PAYOFF)
        assert o == float(HD_HD_PAYOFF)

    def test_dove_dove(self) -> None:
        p, o = self._payoff("dove", "dove")
        assert p == float(HD_DD_PAYOFF)
        assert o == float(HD_DD_PAYOFF)


# ── Ultimatum payoffs ───────────────────────────────────────────────────


class TestUltimatumPayoffs:
    """Verify accept/reject logic for the Ultimatum Game."""

    _payoff = staticmethod(get_game("ultimatum").payoff_fn)

    def test_accept_gives_correct_split(self) -> None:
        offer = _FIVE
        p, o = self._payoff(f"offer_{offer}", "accept")
        assert p == float(ULTIMATUM_POT - offer)
        assert o == float(offer)

    def test_reject_gives_zero_for_both(self) -> None:
        offer = _THREE
        p, o = self._payoff(f"offer_{offer}", "reject")
        assert p == float(_ZERO)
        assert o == float(_ZERO)

    def test_single_shot(self) -> None:
        assert get_game("ultimatum").default_rounds == SINGLE_SHOT_ROUNDS


# ── Trust payoffs ───────────────────────────────────────────────────────


class TestTrustPayoffs:
    """Verify Trust Game payoff computation."""

    _payoff = staticmethod(get_game("trust").payoff_fn)

    def test_player_gets_endowment_minus_invest_plus_returned(self) -> None:
        invest = _FIVE
        returned = _THREE
        p, _ = self._payoff(f"invest_{invest}", f"return_{returned}")
        assert p == float(TRUST_ENDOWMENT - invest + returned)

    def test_opponent_gets_multiplied_minus_returned(self) -> None:
        invest = _FOUR
        returned = _TWO
        _, o = self._payoff(f"invest_{invest}", f"return_{returned}")
        assert o == float(invest * TRUST_MULTIPLIER - returned)

    def test_single_shot(self) -> None:
        assert get_game("trust").default_rounds == SINGLE_SHOT_ROUNDS


# ── Public Goods payoffs ────────────────────────────────────────────────


class TestPublicGoodsPayoffs:
    """Verify Public Goods Game pot computation and distribution."""

    _payoff = staticmethod(get_game("public_goods").payoff_fn)

    def test_pot_computation_and_equal_split(self) -> None:
        pc = _FIVE
        oc = _THREE
        total = pc + oc
        multiplied = total * PG_MULTIPLIER_NUMERATOR / PG_MULTIPLIER_DENOMINATOR
        share = multiplied / PG_DEFAULT_NUM_PLAYERS
        expected_player = float(PG_ENDOWMENT - pc) + share
        expected_opponent = float(PG_ENDOWMENT - oc) + share
        p, o = self._payoff(f"contribute_{pc}", f"contribute_{oc}")
        assert p == pytest.approx(expected_player)
        assert o == pytest.approx(expected_opponent)

    def test_zero_contributions(self) -> None:
        p, o = self._payoff(f"contribute_{_ZERO}", f"contribute_{_ZERO}")
        assert p == float(PG_ENDOWMENT)
        assert o == float(PG_ENDOWMENT)

    def test_single_shot(self) -> None:
        assert get_game("public_goods").default_rounds == SINGLE_SHOT_ROUNDS


# ── Game config attribute tests ─────────────────────────────────────────


class TestGameConfigAttributes:
    """Ensure every GameConfig has consistent attributes."""

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_actions_list_is_non_empty(self, key: str) -> None:
        cfg = get_game(key)
        assert len(cfg.actions) > _ZERO

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_payoff_fn_is_callable(self, key: str) -> None:
        cfg = get_game(key)
        assert callable(cfg.payoff_fn)

    @pytest.mark.parametrize("key", _ALL_GAME_KEYS)
    def test_name_is_non_empty_string(self, key: str) -> None:
        cfg = get_game(key)
        assert isinstance(cfg.name, str)
        assert len(cfg.name) > _ZERO
