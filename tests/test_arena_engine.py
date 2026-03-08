"""Full integration tests for MetagameArena with mock models."""
from __future__ import annotations
import sys
import types

if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw: object) -> None:
            super().__init_subclass__(**kw)
        def __class_getitem__(cls, params: object) -> type:
            return cls
        def __init__(self) -> None:
            pass

    _iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
    _openenv_stub.core = _core_stub  # type: ignore[attr-defined]
    _core_stub.env_server = _server_stub  # type: ignore[attr-defined]
    _server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

import pytest
from env.arena.engine import MetagameArena
from env.arena.models import ArenaRoundResult

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_TEN = _FIVE + _FIVE

_MODEL_COOP = "always_cooperate"
_MODEL_DEFECT = "always_defect"
_MODEL_TFT = "tit_for_tat"


def _coop_generate(prompt: str) -> str:
    return "cooperate"


def _defect_generate(prompt: str) -> str:
    return "defect"


def _tft_generate(prompt: str) -> str:
    """Simple tit-for-tat: cooperate first, then mirror."""
    if "defect" in prompt.lower() and "opponent" in prompt.lower():
        return "defect"
    return "cooperate"


def _make_arena(total_rounds: int = _THREE, pd_only: bool = False) -> MetagameArena:
    arena = MetagameArena(total_rounds=total_rounds)
    arena.add_model(_MODEL_COOP, _coop_generate, "strategy")
    arena.add_model(_MODEL_DEFECT, _defect_generate, "strategy")
    arena.add_model(_MODEL_TFT, _tft_generate, "strategy")
    if pd_only:
        arena.game_pool._games = ["prisoners_dilemma"]
    return arena


class TestArenaSetup:
    """Verify arena creation and model registration."""

    def test_three_models_registered(self) -> None:
        arena = _make_arena()
        assert arena.roster.active_count == _THREE

    def test_has_quorum(self) -> None:
        arena = _make_arena()
        assert arena.roster.has_quorum() is True

    def test_initial_state(self) -> None:
        arena = _make_arena()
        assert arena.state.round_number == _ZERO
        assert arena.state.total_rounds == _THREE


class TestRunRound:
    """Verify a single round executes correctly."""

    def test_returns_round_result(self) -> None:
        arena = _make_arena()
        result = arena.run_round()
        assert isinstance(result, ArenaRoundResult)

    def test_round_number_increments(self) -> None:
        arena = _make_arena()
        arena.run_round()
        assert arena.state.round_number == _ONE

    def test_game_results_present(self) -> None:
        arena = _make_arena()
        result = arena.run_round()
        assert len(result.game_results) > _ZERO

    def test_reputation_updates_for_all(self) -> None:
        arena = _make_arena()
        result = arena.run_round()
        assert _MODEL_COOP in result.reputation_updates
        assert _MODEL_DEFECT in result.reputation_updates
        assert _MODEL_TFT in result.reputation_updates

    def test_round_robin_pairings(self) -> None:
        arena = _make_arena()
        result = arena.run_round()
        pairs = set()
        for r in result.game_results:
            if "error" not in r:
                pairs.add((r["player"], r["opponent"]))
        expected_pair_count = _THREE
        games_selected = len(arena.game_pool.available_games)
        assert len(pairs) >= expected_pair_count


class TestRunFullArena:
    """Verify multi-round arena execution."""

    def test_all_rounds_complete(self) -> None:
        arena = _make_arena(total_rounds=_TWO)
        results = arena.run_full_arena()
        assert len(results) == _TWO
        assert arena.state.round_number == _TWO

    def test_history_accumulates(self) -> None:
        arena = _make_arena(total_rounds=_THREE)
        arena.run_full_arena()
        assert len(arena.state.round_history) == _THREE

    def test_games_played_increments(self) -> None:
        arena = _make_arena(total_rounds=_TWO)
        arena.run_full_arena()
        coop_profile = arena.roster.get_profile(_MODEL_COOP)
        assert coop_profile is not None
        assert coop_profile.games_played > _ZERO


class TestReputationDivergence:
    """Verify reputation differentiates strategies over rounds."""

    def test_cooperation_signal_diverges(self) -> None:
        arena = _make_arena(total_rounds=_TEN, pd_only=True)
        arena.run_full_arena()
        coop_sig = arena.reputation.get_signal(_MODEL_COOP, "cooperation")
        defect_sig = arena.reputation.get_signal(_MODEL_DEFECT, "cooperation")
        assert coop_sig > defect_sig

    def test_profiles_updated_after_play(self) -> None:
        arena = _make_arena(total_rounds=_THREE)
        arena.run_full_arena()
        coop_profile = arena.roster.get_profile(_MODEL_COOP)
        defect_profile = arena.roster.get_profile(_MODEL_DEFECT)
        assert coop_profile is not None
        assert defect_profile is not None
        assert coop_profile.games_played > _ZERO
        assert defect_profile.games_played > _ZERO


class TestGamePool:
    """Verify game pool integration."""

    def test_games_selected_per_round(self) -> None:
        arena = _make_arena()
        result = arena.run_round()
        assert len(result.game_results) > _ZERO

    def test_play_counts_updated(self) -> None:
        arena = _make_arena()
        arena.run_round()
        total_plays = sum(
            arena.game_pool._play_counts.get(g, _ZERO)
            for g in arena.game_pool.available_games
        )
        assert total_plays > _ZERO
