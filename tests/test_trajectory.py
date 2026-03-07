"""Tests for train/trajectory.py -- trajectory collection."""

from __future__ import annotations

import sys
import types

# Stub the openenv package so env.environment can be imported
# even when the openenv dependency is not installed.
if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        """Minimal stand-in for Environment base class."""
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

from env.environment import MachiavelliEnvironment
from env.models import GameAction, GameObservation
from train.trajectory import (
    EpisodeTrajectory,
    StepRecord,
    TrajectoryCollector,
)
from train.rewards import episode_reward
from constant_definitions.game_constants import (
    EVAL_ONE,
    EVAL_ZERO,
    EVAL_ZERO_FLOAT,
)

_ONE = int(bool(True))


def _simple_agent(obs: GameObservation) -> GameAction:
    """Always pick the first available action."""
    return GameAction(action=obs.available_actions[EVAL_ZERO])


class _AgentWithTracking:
    """Simple agent that exposes last_prompt and last_completion."""

    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_completion = ""

    def __call__(self, obs: GameObservation) -> GameAction:
        self.last_prompt = f"game={obs.game_name}"
        action = obs.available_actions[EVAL_ZERO]
        self.last_completion = action
        return GameAction(action=action)


# ── StepRecord tests ──


def test_step_record_creation():
    """StepRecord should hold all required fields."""
    step = StepRecord(
        prompt="test prompt",
        completion="cooperate",
        action="cooperate",
        reward=EVAL_ZERO_FLOAT,
        player_payoff=float(EVAL_ONE + EVAL_ONE + EVAL_ONE),
        opponent_payoff=float(EVAL_ONE + EVAL_ONE + EVAL_ONE),
        round_number=_ONE,
    )
    assert step.action == "cooperate"
    assert step.round_number == _ONE


# ── EpisodeTrajectory tests ──


def test_episode_trajectory_defaults():
    """EpisodeTrajectory defaults should be sensible."""
    traj = EpisodeTrajectory(game="test", strategy="test")
    assert traj.episode_reward == EVAL_ZERO_FLOAT
    assert traj.steps == []
    assert traj.rounds_played == int()


# ── TrajectoryCollector tests ──


def test_collector_single_episode():
    """TrajectoryCollector should produce a valid trajectory."""
    env = MachiavelliEnvironment()
    agent = _AgentWithTracking()
    collector = TrajectoryCollector(
        env=env,
        agent=agent,
        reward_fn=episode_reward,
    )
    traj = collector.collect_episode(
        game="prisoners_dilemma",
        strategy="tit_for_tat",
    )
    assert isinstance(traj, EpisodeTrajectory)
    assert traj.game == "prisoners_dilemma"
    assert traj.strategy == "tit_for_tat"
    assert len(traj.steps) > EVAL_ZERO
    assert traj.rounds_played > EVAL_ZERO


def test_collector_steps_have_prompts():
    """Each step should have a non-empty prompt."""
    env = MachiavelliEnvironment()
    agent = _AgentWithTracking()
    collector = TrajectoryCollector(env=env, agent=agent)
    traj = collector.collect_episode(
        game="prisoners_dilemma",
        strategy="always_cooperate",
    )
    for step in traj.steps:
        assert len(step.prompt) > EVAL_ZERO


def test_collector_batch():
    """collect_batch should return trajectories for each combination."""
    env = MachiavelliEnvironment()
    agent = _AgentWithTracking()
    collector = TrajectoryCollector(env=env, agent=agent)
    trajectories = collector.collect_batch(
        games=["prisoners_dilemma"],
        strategies=["tit_for_tat", "always_cooperate"],
    )
    assert len(trajectories) == EVAL_ONE + EVAL_ONE


def test_collector_with_reward_fn():
    """Providing a reward_fn should produce non-default episode_reward."""
    env = MachiavelliEnvironment()
    agent = _AgentWithTracking()
    collector = TrajectoryCollector(
        env=env,
        agent=agent,
        reward_fn=episode_reward,
    )
    traj = collector.collect_episode(
        game="prisoners_dilemma",
        strategy="always_cooperate",
    )
    # With a real reward function, episode_reward should be computed
    assert isinstance(traj.episode_reward, float)
