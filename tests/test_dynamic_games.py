"""Tests for the dynamic game creation API."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

import pytest

from constant_definitions.nplayer.dynamic_constants import (
    MIN_ACTIONS,
    MAX_ACTIONS,
    DYNAMIC_DEFAULT_ROUNDS,
    REGISTRY_PREFIX,
)
from common.games import GAMES, GameConfig, get_game
from common.games_meta.dynamic import (
    create_matrix_game,
    create_symmetric_game,
    create_custom_game,
    unregister_game,
)

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE

_ZERO_F = float()
_ONE_F = float(_ONE)
_TWO_F = float(_TWO)
_THREE_F = float(_THREE)
_FOUR_F = float(_FOUR)
_FIVE_F = float(_FIVE)
_NEG_ONE_F = float(-_ONE)

# ── Fixtures ────────────────────────────────────────────────────────────

_ACTIONS_AB = ["action_a", "action_b"]

_SIMPLE_MATRIX = {
    ("action_a", "action_a"): (_THREE_F, _THREE_F),
    ("action_a", "action_b"): (_ZERO_F, _FIVE_F),
    ("action_b", "action_a"): (_FIVE_F, _ZERO_F),
    ("action_b", "action_b"): (_ONE_F, _ONE_F),
}

_SYMMETRIC_PAYOFFS = {
    ("action_a", "action_a"): _THREE_F,
    ("action_a", "action_b"): _ZERO_F,
    ("action_b", "action_a"): _FIVE_F,
    ("action_b", "action_b"): _ONE_F,
}


# ── create_matrix_game ──────────────────────────────────────────────────


class TestCreateMatrixGame:
    def test_returns_game_config(self) -> None:
        cfg = create_matrix_game("test_mat", _ACTIONS_AB, _SIMPLE_MATRIX)
        assert isinstance(cfg, GameConfig)
        assert cfg.name == "test_mat"
        assert cfg.actions == _ACTIONS_AB
        assert cfg.game_type == "matrix"
        assert cfg.default_rounds == DYNAMIC_DEFAULT_ROUNDS

    def test_payoff_fn_correct(self) -> None:
        cfg = create_matrix_game("test_mat", _ACTIONS_AB, _SIMPLE_MATRIX)
        assert cfg.payoff_fn("action_a", "action_a") == (_THREE_F, _THREE_F)
        assert cfg.payoff_fn("action_a", "action_b") == (_ZERO_F, _FIVE_F)
        assert cfg.payoff_fn("action_b", "action_a") == (_FIVE_F, _ZERO_F)
        assert cfg.payoff_fn("action_b", "action_b") == (_ONE_F, _ONE_F)

    def test_register_adds_to_games(self) -> None:
        key = REGISTRY_PREFIX + "reg_test"
        try:
            create_matrix_game(
                "reg_test", _ACTIONS_AB, _SIMPLE_MATRIX, register=True,
            )
            assert key in GAMES
            assert GAMES[key].name == "reg_test"
        finally:
            GAMES.pop(key, None)

    def test_no_register_by_default(self) -> None:
        before = len(GAMES)
        create_matrix_game("no_reg", _ACTIONS_AB, _SIMPLE_MATRIX)
        assert len(GAMES) == before

    def test_custom_rounds(self) -> None:
        cfg = create_matrix_game(
            "test_rounds", _ACTIONS_AB, _SIMPLE_MATRIX,
            default_rounds=_FIVE,
        )
        assert cfg.default_rounds == _FIVE

    def test_custom_description(self) -> None:
        cfg = create_matrix_game(
            "desc", _ACTIONS_AB, _SIMPLE_MATRIX,
            description="My game",
        )
        assert cfg.description == "My game"


# ── create_symmetric_game ───────────────────────────────────────────────


class TestCreateSymmetricGame:
    def test_returns_game_config(self) -> None:
        cfg = create_symmetric_game("test_sym", _ACTIONS_AB, _SYMMETRIC_PAYOFFS)
        assert isinstance(cfg, GameConfig)
        assert cfg.name == "test_sym"

    def test_payoff_symmetry(self) -> None:
        cfg = create_symmetric_game("test_sym", _ACTIONS_AB, _SYMMETRIC_PAYOFFS)
        p_ab, o_ab = cfg.payoff_fn("action_a", "action_b")
        p_ba, o_ba = cfg.payoff_fn("action_b", "action_a")
        assert p_ab == o_ba
        assert o_ab == p_ba

    def test_register(self) -> None:
        key = REGISTRY_PREFIX + "sym_reg"
        try:
            create_symmetric_game(
                "sym_reg", _ACTIONS_AB, _SYMMETRIC_PAYOFFS, register=True,
            )
            assert key in GAMES
        finally:
            GAMES.pop(key, None)


# ── create_custom_game ──────────────────────────────────────────────────


class TestCreateCustomGame:
    def test_returns_game_config(self) -> None:
        def my_fn(a: str, b: str) -> tuple[float, float]:
            return (_ONE_F, _TWO_F)

        cfg = create_custom_game("test_cust", _ACTIONS_AB, my_fn)
        assert isinstance(cfg, GameConfig)
        assert cfg.payoff_fn("action_a", "action_b") == (_ONE_F, _TWO_F)

    def test_custom_game_type(self) -> None:
        def my_fn(a: str, b: str) -> tuple[float, float]:
            return (_ONE_F, _ONE_F)

        cfg = create_custom_game(
            "typed", _ACTIONS_AB, my_fn, game_type="custom",
        )
        assert cfg.game_type == "custom"

    def test_register(self) -> None:
        def my_fn(a: str, b: str) -> tuple[float, float]:
            return (_ONE_F, _ONE_F)

        key = REGISTRY_PREFIX + "cust_reg"
        try:
            create_custom_game(
                "cust_reg", _ACTIONS_AB, my_fn, register=True,
            )
            assert key in GAMES
        finally:
            GAMES.pop(key, None)


# ── unregister_game ─────────────────────────────────────────────────────


class TestUnregisterGame:
    def test_removes_game(self) -> None:
        key = REGISTRY_PREFIX + "unreg_test"
        create_matrix_game(
            "unreg_test", _ACTIONS_AB, _SIMPLE_MATRIX, register=True,
        )
        assert key in GAMES
        unregister_game(key)
        assert key not in GAMES

    def test_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            unregister_game("nonexistent_game_xyz")


# ── Validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_too_few_actions(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            create_matrix_game("bad", ["only_one"], {})

    def test_too_many_actions(self) -> None:
        actions = [f"act_{i}" for i in range(MAX_ACTIONS + _ONE)]
        with pytest.raises(ValueError, match="At most"):
            create_matrix_game("bad", actions, {})

    def test_duplicate_actions(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            create_matrix_game("bad", ["a", "a"], {})

    def test_missing_matrix_entry(self) -> None:
        incomplete = dict(_SIMPLE_MATRIX)
        del incomplete[("action_a", "action_b")]
        with pytest.raises(ValueError, match="missing"):
            create_matrix_game("bad", _ACTIONS_AB, incomplete)

    def test_extra_matrix_entry(self) -> None:
        extra = dict(_SIMPLE_MATRIX)
        extra[("action_c", "action_a")] = (_ZERO_F, _ZERO_F)
        with pytest.raises(ValueError, match="unknown"):
            create_matrix_game("bad", _ACTIONS_AB, extra)

    def test_missing_symmetric_entry(self) -> None:
        incomplete = dict(_SYMMETRIC_PAYOFFS)
        del incomplete[("action_a", "action_b")]
        with pytest.raises(ValueError, match="missing"):
            create_symmetric_game("bad", _ACTIONS_AB, incomplete)


# ── Backward compatibility ──────────────────────────────────────────────


class TestBackwardCompat:
    def test_existing_games_untouched(self) -> None:
        pd = get_game("prisoners_dilemma")
        assert pd.name == "Prisoner's Dilemma"
        assert pd.actions == ["cooperate", "defect"]

    def test_games_dict_has_existing_keys(self) -> None:
        assert "prisoners_dilemma" in GAMES
        assert "stag_hunt" in GAMES
        assert "hawk_dove" in GAMES
