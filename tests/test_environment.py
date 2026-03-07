"""Tests for the MachiaveliBench environment."""
import sys
import types
from unittest.mock import MagicMock

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
    """Minimal stand-in for openenv.core.env_server.interfaces.Environment."""
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

from constant_definitions.game_constants import (
    DEFAULT_NUM_ROUNDS,
    SINGLE_SHOT_ROUNDS,
    PD_CC_PAYOFF,
)
from env.models import GameAction, GameObservation, GameState
from env.environment import MachiavelliEnvironment

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE


@pytest.fixture()
def env() -> MachiavelliEnvironment:
    """Return a fresh, un-reset environment."""
    return MachiavelliEnvironment()


@pytest.fixture()
def pd_env(env: MachiavelliEnvironment) -> MachiavelliEnvironment:
    """Return an environment reset for Prisoner's Dilemma."""
    env.reset(game="prisoners_dilemma", strategy="always_cooperate")
    return env


# ── reset tests ─────────────────────────────────────────────────────────


class TestReset:
    """Verify that reset returns a valid initial observation."""

    def test_returns_game_observation(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert isinstance(obs, GameObservation)

    def test_observation_not_done(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert obs.done is False

    def test_observation_game_name(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert obs.game_name == "prisoners_dilemma"

    def test_observation_available_actions(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert "cooperate" in obs.available_actions
        assert "defect" in obs.available_actions

    def test_observation_total_rounds(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert obs.total_rounds == DEFAULT_NUM_ROUNDS

    def test_observation_current_round_is_zero(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert obs.current_round == _ZERO

    def test_scores_start_at_zero(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert obs.player_score == float(_ZERO)
        assert obs.opponent_score == float(_ZERO)

    def test_history_empty(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
        assert len(obs.history) == _ZERO

    def test_custom_num_rounds(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="tit_for_tat", num_rounds=_THREE)
        assert obs.total_rounds == _THREE

    def test_opponent_strategy_field(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="prisoners_dilemma", strategy="always_defect")
        assert obs.opponent_strategy == "always_defect"


# ── step tests ──────────────────────────────────────────────────────────


class TestStep:
    """Verify that step processes actions correctly."""

    def test_returns_observation(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert isinstance(obs, GameObservation)

    def test_advances_round(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert obs.current_round == _ONE

    def test_records_history(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert len(obs.history) == _ONE

    def test_reward_is_payoff(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert obs.reward == float(PD_CC_PAYOFF)

    def test_last_round_present(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert obs.last_round is not None
        assert obs.last_round.player_action == "cooperate"

    def test_opponent_action_recorded(self, pd_env: MachiavelliEnvironment) -> None:
        obs = pd_env.step(GameAction(action="cooperate"))
        assert obs.last_round is not None
        assert obs.last_round.opponent_action == "cooperate"


# ── episode completion ──────────────────────────────────────────────────


class TestEpisodeCompletion:
    """Verify the episode terminates after total_rounds."""

    def test_episode_ends_after_total_rounds(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_THREE)
        obs = None
        for _ in range(_THREE):
            obs = env.step(GameAction(action="cooperate"))
        assert obs is not None
        assert obs.done is True

    def test_not_done_before_final_round(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_THREE)
        obs = None
        for _ in range(_THREE - _ONE):
            obs = env.step(GameAction(action="cooperate"))
        assert obs is not None
        assert obs.done is False


# ── score accumulation ──────────────────────────────────────────────────


class TestScoreAccumulation:
    """Verify scores accumulate over multiple rounds."""

    def test_player_score_accumulates(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_THREE)
        obs = None
        for _ in range(_THREE):
            obs = env.step(GameAction(action="cooperate"))
        assert obs is not None
        assert obs.player_score == float(PD_CC_PAYOFF) * _THREE

    def test_opponent_score_accumulates(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_TWO)
        obs = None
        for _ in range(_TWO):
            obs = env.step(GameAction(action="cooperate"))
        assert obs is not None
        assert obs.opponent_score == float(PD_CC_PAYOFF) * _TWO


# ── state tracking ──────────────────────────────────────────────────────


class TestStateTracking:
    """Verify the state property reflects game progress."""

    def test_state_returns_game_state(self, pd_env: MachiavelliEnvironment) -> None:
        assert isinstance(pd_env.state, GameState)

    def test_state_game_name(self, pd_env: MachiavelliEnvironment) -> None:
        assert pd_env.state.game_name == "prisoners_dilemma"

    def test_state_history_grows(self, pd_env: MachiavelliEnvironment) -> None:
        pd_env.step(GameAction(action="cooperate"))
        assert len(pd_env.state.history) == _ONE
        pd_env.step(GameAction(action="defect"))
        assert len(pd_env.state.history) == _TWO

    def test_state_is_done_flag(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_ONE)
        assert env.state.is_done is False
        env.step(GameAction(action="cooperate"))
        assert env.state.is_done is True


# ── error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify proper exceptions for invalid usage."""

    def test_step_before_reset_raises_runtime_error(self, env: MachiavelliEnvironment) -> None:
        with pytest.raises(RuntimeError):
            env.step(GameAction(action="cooperate"))

    def test_invalid_action_raises_value_error(self, pd_env: MachiavelliEnvironment) -> None:
        with pytest.raises(ValueError):
            pd_env.step(GameAction(action="invalid_action"))

    def test_step_after_done_raises_runtime_error(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_ONE)
        env.step(GameAction(action="cooperate"))
        with pytest.raises(RuntimeError):
            env.step(GameAction(action="cooperate"))


# ── different games ─────────────────────────────────────────────────────


class TestDifferentGames:
    """Verify the environment works with various game selections."""

    def test_stag_hunt_reset(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="stag_hunt", strategy="always_cooperate")
        assert obs.game_name == "stag_hunt"
        assert "stag" in obs.available_actions

    def test_hawk_dove_reset(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="hawk_dove", strategy="always_defect")
        assert obs.game_name == "hawk_dove"
        assert "hawk" in obs.available_actions

    def test_ultimatum_single_shot(self, env: MachiavelliEnvironment) -> None:
        obs = env.reset(game="ultimatum", strategy="ultimatum_fair")
        assert obs.total_rounds == SINGLE_SHOT_ROUNDS

    def test_stag_hunt_step(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="stag_hunt", strategy="always_cooperate")
        obs = env.step(GameAction(action="stag"))
        assert obs.current_round == _ONE

    def test_reset_clears_previous_state(self, env: MachiavelliEnvironment) -> None:
        env.reset(game="prisoners_dilemma", strategy="always_cooperate", num_rounds=_TWO)
        env.step(GameAction(action="cooperate"))
        obs = env.reset(game="stag_hunt", strategy="always_defect")
        assert obs.game_name == "stag_hunt"
        assert obs.current_round == _ZERO
        assert len(obs.history) == _ZERO
