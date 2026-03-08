"""Tests for opponent mode variants (self-play, cross-model)."""
from __future__ import annotations

import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")


class _EnvironmentStub:
    """Minimal stand-in for openenv Environment."""
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
    def __class_getitem__(cls, params: object) -> type:
        return cls
    def __init__(self) -> None:
        pass


_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv_stub.core = _core_stub  # type: ignore[attr-defined]
_core_stub.env_server = _server_stub  # type: ignore[attr-defined]
_server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
for _name, _mod in [
    ("openenv", _openenv_stub),
    ("openenv.core", _core_stub),
    ("openenv.core.env_server", _server_stub),
    ("openenv.core.env_server.interfaces", _iface_stub),
]:
    sys.modules.setdefault(_name, _mod)

from env.models import GameAction, GameObservation
from common.games import GAMES
from common.variants import apply_self_play, apply_cross_model
from bench.evaluation.tournament import TournamentRunner
from constant_definitions.game_constants import (
    PD_CC_PAYOFF,
    PD_DC_PAYOFF,
    PD_CD_PAYOFF,
    DEFAULT_NUM_ROUNDS,
    OPPONENT_MODE_SELF,
    OPPONENT_MODE_CROSS,
)
from constant_definitions.var.pd_variant_constants import (
    VARIANT_SELF_PLAY,
    VARIANT_CROSS_MODEL,
)

_ZERO = int()
_ONE = int(bool(True))


def _always_cooperate(obs: GameObservation) -> GameAction:
    return GameAction(action=obs.available_actions[_ZERO])


def _always_defect(obs: GameObservation) -> GameAction:
    return GameAction(action=obs.available_actions[_ONE])


class TestSelfPlayMode:
    """Tests for self-play opponent mode via variant system."""

    def test_variant_metadata(self) -> None:
        base = GAMES["prisoners_dilemma"]
        sp = apply_self_play(base, base_key="prisoners_dilemma")
        assert sp.opponent_mode == OPPONENT_MODE_SELF
        assert VARIANT_SELF_PLAY in sp.applied_variants
        assert sp.base_game_key == "prisoners_dilemma"

    def test_self_play_both_cooperate(self) -> None:
        base = GAMES["prisoners_dilemma"]
        sp_cfg = apply_self_play(base, base_key="prisoners_dilemma")
        GAMES["_test_sp_pd"] = sp_cfg
        try:
            runner = TournamentRunner(agent_fn=_always_cooperate)
            results = runner.run_tournament(
                games=["_test_sp_pd"],
                strategies=["always_cooperate"],
                num_episodes=_ONE,
            )
            ep = results.games["_test_sp_pd"].strategy_results[
                "always_cooperate"
            ].episodes[_ZERO]
            expected = float(PD_CC_PAYOFF) * DEFAULT_NUM_ROUNDS
            assert ep.player_score == expected
            assert ep.opponent_score == expected
            assert ep.opponent_mode == OPPONENT_MODE_SELF
        finally:
            del GAMES["_test_sp_pd"]


class TestCrossModelMode:
    """Tests for cross-model opponent mode via variant system."""

    def test_variant_metadata(self) -> None:
        base = GAMES["prisoners_dilemma"]
        cm = apply_cross_model(base, base_key="prisoners_dilemma")
        assert cm.opponent_mode == OPPONENT_MODE_CROSS
        assert VARIANT_CROSS_MODEL in cm.applied_variants

    def test_cross_model_faces_defection(self) -> None:
        base = GAMES["prisoners_dilemma"]
        cm_cfg = apply_cross_model(base, base_key="prisoners_dilemma")
        GAMES["_test_cm_pd"] = cm_cfg
        try:
            runner = TournamentRunner(
                agent_fn=_always_cooperate,
                opponent_agent_fn=_always_defect,
            )
            results = runner.run_tournament(
                games=["_test_cm_pd"],
                strategies=["always_cooperate"],
                num_episodes=_ONE,
            )
            ep = results.games["_test_cm_pd"].strategy_results[
                "always_cooperate"
            ].episodes[_ZERO]
            expected_player = float(PD_CD_PAYOFF) * DEFAULT_NUM_ROUNDS
            expected_opp = float(PD_DC_PAYOFF) * DEFAULT_NUM_ROUNDS
            assert ep.player_score == expected_player
            assert ep.opponent_score == expected_opp
            assert ep.opponent_mode == OPPONENT_MODE_CROSS
        finally:
            del GAMES["_test_cm_pd"]

    def test_cross_model_falls_back_to_agent_fn(self) -> None:
        base = GAMES["prisoners_dilemma"]
        cm_cfg = apply_cross_model(base, base_key="prisoners_dilemma")
        GAMES["_test_cm_fb"] = cm_cfg
        try:
            runner = TournamentRunner(agent_fn=_always_cooperate)
            results = runner.run_tournament(
                games=["_test_cm_fb"],
                strategies=["always_cooperate"],
                num_episodes=_ONE,
            )
            ep = results.games["_test_cm_fb"].strategy_results[
                "always_cooperate"
            ].episodes[_ZERO]
            expected = float(PD_CC_PAYOFF) * DEFAULT_NUM_ROUNDS
            assert ep.player_score == expected
        finally:
            del GAMES["_test_cm_fb"]
