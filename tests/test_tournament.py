"""Tests for model-vs-model tournament support."""
from __future__ import annotations

import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

# Stub the openenv package so the environment module can be imported
# even when the openenv dependency is not installed.
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
    sys.modules[_name] = _mod

import pytest

from env.models import GameAction, GameObservation
from env.environment import MachiavelliEnvironment
from bench.evaluation.tournament import TournamentRunner
from bench.evaluation.model_matchups import (
    MatchupResult,
    ModelMatchupRunner,
    ModelTournamentResults,
)
from constant_definitions.game_constants import (
    PD_CC_PAYOFF,
    PD_DC_PAYOFF,
    PD_CD_PAYOFF,
    DEFAULT_NUM_ROUNDS,
)

_ZERO = int()
_ONE = int(bool(True))


def _always_cooperate(obs: GameObservation) -> GameAction:
    return GameAction(action=obs.available_actions[_ZERO])


def _always_defect(obs: GameObservation) -> GameAction:
    return GameAction(action=obs.available_actions[_ONE])


def _tit_for_tat(obs: GameObservation) -> GameAction:
    if not obs.history:
        return GameAction(action=obs.available_actions[_ZERO])
    last = obs.history[-_ONE].opponent_action
    if last in obs.available_actions:
        return GameAction(action=last)
    return GameAction(action=obs.available_actions[_ZERO])


class TestOpponentFn:
    """Tests that the environment correctly uses opponent_fn."""

    def test_opponent_fn_overrides_strategy(self) -> None:
        env = MachiavelliEnvironment()
        obs = env.reset(game="prisoners_dilemma", opponent_fn=_always_defect)
        assert obs.opponent_strategy == "agent"
        obs = env.step(GameAction(action="cooperate"))
        assert obs.last_round is not None
        assert obs.last_round.opponent_action == "defect"

    def test_opponent_fn_receives_flipped_history(self) -> None:
        received: list[GameObservation] = []

        def _spy(obs: GameObservation) -> GameAction:
            received.append(obs)
            return GameAction(action=obs.available_actions[_ZERO])

        env = MachiavelliEnvironment()
        env.reset(game="prisoners_dilemma", opponent_fn=_spy)
        env.step(GameAction(action="cooperate"))
        env.step(GameAction(action="defect"))
        assert len(received) >= _ONE + _ONE
        second_obs = received[_ONE]
        r = second_obs.history[_ZERO]
        assert r.player_action == "cooperate"
        assert r.opponent_action == "cooperate"

    def test_opponent_fn_scores_swapped(self) -> None:
        received: list[GameObservation] = []

        def _spy(obs: GameObservation) -> GameAction:
            received.append(obs)
            return GameAction(action="defect")

        env = MachiavelliEnvironment()
        env.reset(game="prisoners_dilemma", opponent_fn=_spy)
        env.step(GameAction(action="cooperate"))
        env.step(GameAction(action="cooperate"))
        second_obs = received[_ONE]
        assert second_obs.player_score == float(PD_DC_PAYOFF)
        assert second_obs.opponent_score == float(PD_CD_PAYOFF)

    def test_strategy_still_works_without_opponent_fn(self) -> None:
        env = MachiavelliEnvironment()
        obs = env.reset(game="prisoners_dilemma", strategy="always_cooperate")
        obs = env.step(GameAction(action="defect"))
        assert obs.last_round is not None
        assert obs.last_round.opponent_action == "cooperate"


class TestModelMatchups:
    """Tests for the ModelMatchupRunner."""

    def test_two_agents_play_full_game(self) -> None:
        runner = ModelMatchupRunner()
        agents = {"coop": _always_cooperate, "defect": _always_defect}
        results = runner.run_model_matchups(
            agents=agents,
            games=["prisoners_dilemma"],
            num_episodes=_ONE,
        )
        assert isinstance(results, ModelTournamentResults)
        coop_vs_defect = [
            m for m in results.matchups
            if m.agent_a == "coop" and m.agent_b == "defect"
        ]
        assert len(coop_vs_defect) == _ONE
        m = coop_vs_defect[_ZERO]
        expected_a = float(PD_CD_PAYOFF) * DEFAULT_NUM_ROUNDS
        expected_b = float(PD_DC_PAYOFF) * DEFAULT_NUM_ROUNDS
        assert m.score_a == expected_a
        assert m.score_b == expected_b

    def test_self_play(self) -> None:
        runner = ModelMatchupRunner()
        agents = {"coop": _always_cooperate}
        results = runner.run_model_matchups(
            agents=agents,
            games=["prisoners_dilemma"],
            num_episodes=_ONE,
        )
        assert len(results.matchups) == _ONE
        m = results.matchups[_ZERO]
        assert m.agent_a == "coop"
        assert m.agent_b == "coop"
        expected = float(PD_CC_PAYOFF) * DEFAULT_NUM_ROUNDS
        assert m.score_a == expected
        assert m.score_b == expected

    def test_all_pairs_generated(self) -> None:
        runner = ModelMatchupRunner()
        agents = {
            "a": _always_cooperate,
            "b": _always_defect,
            "c": _tit_for_tat,
        }
        results = runner.run_model_matchups(
            agents=agents,
            games=["prisoners_dilemma"],
            num_episodes=_ONE,
        )
        pairs = {(m.agent_a, m.agent_b) for m in results.matchups}
        expected_count = len(agents) * len(agents)
        assert len(pairs) == expected_count

    def test_matchup_result_fields(self) -> None:
        runner = ModelMatchupRunner()
        agents = {"coop": _always_cooperate, "defect": _always_defect}
        results = runner.run_model_matchups(
            agents=agents,
            games=["prisoners_dilemma"],
            num_episodes=_ONE,
        )
        for m in results.matchups:
            assert isinstance(m, MatchupResult)
            assert m.rounds_played == DEFAULT_NUM_ROUNDS
            assert len(m.history) == DEFAULT_NUM_ROUNDS


class TestStrategyTournamentUnchanged:
    """Verify the existing strategy tournament still works."""

    def test_strategy_tournament_runs(self) -> None:
        runner = TournamentRunner()
        results = runner.run_tournament(
            games=["prisoners_dilemma"],
            strategies=["always_cooperate", "always_defect"],
            num_episodes=_ONE,
        )
        assert results.total_episodes == _ONE + _ONE
        pd = results.games["prisoners_dilemma"]
        assert "always_cooperate" in pd.strategy_results
        assert "always_defect" in pd.strategy_results
