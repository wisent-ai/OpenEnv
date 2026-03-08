"""Tests for self-play training infrastructure."""
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

import pytest
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.self_play.opponents import FrozenOpponent, OpponentPool
from train.trajectory import TrajectoryCollector, EpisodeTrajectory
from train.rewards import episode_reward
from constant_definitions.game_constants import EVAL_ZERO, EVAL_ZERO_FLOAT
from constant_definitions.var.meta.self_play_constants import (
    SELF_PLAY_COOP_WEIGHT_NUMERATOR, SELF_PLAY_COOP_WEIGHT_DENOMINATOR,
    SELF_PLAY_PARETO_WEIGHT_NUMERATOR, SELF_PLAY_PARETO_WEIGHT_DENOMINATOR,
    SELF_PLAY_FAIRNESS_WEIGHT_NUMERATOR, SELF_PLAY_FAIRNESS_WEIGHT_DENOMINATOR,
    SELF_PLAY_EXPLOIT_WEIGHT_NUMERATOR, SELF_PLAY_EXPLOIT_WEIGHT_DENOMINATOR,
    SELF_PLAY_ADAPT_WEIGHT_NUMERATOR, SELF_PLAY_ADAPT_WEIGHT_DENOMINATOR,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO
_TEN = _FIVE + _FIVE
_THIRTY = _TEN * _THREE


def _echo_generate(prompt: str) -> str:
    return "cooperate"


def _defect_generate(prompt: str) -> str:
    return "defect"


class TestFrozenOpponent:
    """Tests for FrozenOpponent callable wrapper."""

    def test_create_from_generate_fn(self) -> None:
        opp = FrozenOpponent(generate_fn=_echo_generate)
        assert callable(opp)

    def test_call_returns_game_action(self) -> None:
        opp = FrozenOpponent(generate_fn=_echo_generate)
        obs = _make_obs(["cooperate", "defect"])
        result = opp(obs)
        assert isinstance(result, GameAction)
        assert result.action == "cooperate"

    def test_returns_valid_action(self) -> None:
        opp = FrozenOpponent(generate_fn=_defect_generate)
        obs = _make_obs(["cooperate", "defect"])
        result = opp(obs)
        assert result.action in ["cooperate", "defect"]

    def test_bad_response_selects_valid_action(self) -> None:
        opp = FrozenOpponent(generate_fn=lambda p: "garbage_xyz")
        obs = _make_obs(["cooperate", "defect"])
        result = opp(obs)
        assert result.action in ["cooperate", "defect"]

    def test_from_api_wraps_callable(self) -> None:
        def api_fn(system: str, user: str) -> str:
            return "cooperate"
        opp = FrozenOpponent.from_api(api_fn)
        obs = _make_obs(["cooperate", "defect"])
        assert opp(obs).action == "cooperate"


class TestOpponentPool:
    """Tests for OpponentPool add/sample/eviction."""

    def test_add_and_sample(self) -> None:
        pool = OpponentPool(max_size=_THREE)
        opp = FrozenOpponent(generate_fn=_echo_generate)
        pool.add(opp)
        assert pool.size == _ONE
        assert pool.sample() is opp

    def test_max_size_eviction(self) -> None:
        pool = OpponentPool(max_size=_TWO)
        first = FrozenOpponent(generate_fn=_echo_generate)
        pool.add(first)
        pool.add(FrozenOpponent(generate_fn=_echo_generate))
        pool.add(FrozenOpponent(generate_fn=_echo_generate))
        assert pool.size == _TWO
        sampled_ids = {id(pool.sample()) for _ in range(_TEN)}
        assert id(first) not in sampled_ids

    def test_empty_pool_raises(self) -> None:
        pool = OpponentPool()
        with pytest.raises(IndexError):
            pool.sample()

    def test_get_opponent_fn_returns_callable(self) -> None:
        pool = OpponentPool()
        pool.add(FrozenOpponent(generate_fn=_echo_generate))
        assert callable(pool.get_opponent_fn())


class _TrackingAgent:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_completion = ""

    def __call__(self, obs: GameObservation) -> GameAction:
        self.last_prompt = f"game={obs.game_name}"
        action = obs.available_actions[_ZERO]
        self.last_completion = action
        return GameAction(action=action)


class TestSelfPlayTrajectory:
    """Tests for TrajectoryCollector with opponent_fn."""

    def test_collect_episode_with_opponent_fn(self) -> None:
        env = KantEnvironment()
        agent = _TrackingAgent()
        opp = FrozenOpponent(generate_fn=_echo_generate)
        collector = TrajectoryCollector(
            env=env, agent=agent, reward_fn=episode_reward,
        )
        traj = collector.collect_episode(
            game="prisoners_dilemma", opponent_fn=opp,
        )
        assert isinstance(traj, EpisodeTrajectory)
        assert len(traj.steps) > _ZERO

    def test_collect_batch_with_opponent_fn(self) -> None:
        env = KantEnvironment()
        agent = _TrackingAgent()
        opp = FrozenOpponent(generate_fn=_echo_generate)
        collector = TrajectoryCollector(env=env, agent=agent)
        trajs = collector.collect_batch(
            games=["prisoners_dilemma", "stag_hunt"],
            opponent_fn=opp,
        )
        assert len(trajs) == _TWO

    def test_collect_batch_without_opponent_fn(self) -> None:
        env = KantEnvironment()
        agent = _TrackingAgent()
        collector = TrajectoryCollector(env=env, agent=agent)
        trajs = collector.collect_batch(
            games=["prisoners_dilemma"],
            strategies=["tit_for_tat", "always_cooperate"],
        )
        assert len(trajs) == _TWO


class TestSelfPlayReward:
    """Tests for self-play reward weights."""

    def test_episode_reward_with_self_play_weights(self) -> None:
        weights = {
            "cooperation_rate": SELF_PLAY_COOP_WEIGHT_NUMERATOR / SELF_PLAY_COOP_WEIGHT_DENOMINATOR,
            "pareto_efficiency": SELF_PLAY_PARETO_WEIGHT_NUMERATOR / SELF_PLAY_PARETO_WEIGHT_DENOMINATOR,
            "fairness_index": SELF_PLAY_FAIRNESS_WEIGHT_NUMERATOR / SELF_PLAY_FAIRNESS_WEIGHT_DENOMINATOR,
            "exploitation_resistance": SELF_PLAY_EXPLOIT_WEIGHT_NUMERATOR / SELF_PLAY_EXPLOIT_WEIGHT_DENOMINATOR,
            "adaptability": SELF_PLAY_ADAPT_WEIGHT_NUMERATOR / SELF_PLAY_ADAPT_WEIGHT_DENOMINATOR,
        }
        r = episode_reward(
            player_score=float(_THIRTY), opponent_score=float(_THIRTY),
            cooperation_rate=float(_ONE), total_rounds=_TEN,
            weights=weights,
        )
        assert isinstance(r, float)
        assert r > EVAL_ZERO_FLOAT


def _make_obs(actions: list[str]) -> GameObservation:
    return GameObservation(
        done=False, reward=EVAL_ZERO_FLOAT,
        game_name="prisoners_dilemma",
        game_description="test",
        available_actions=actions,
        current_round=_ZERO, total_rounds=_FIVE,
        history=[], player_score=EVAL_ZERO_FLOAT,
        opponent_score=EVAL_ZERO_FLOAT, opponent_strategy="agent",
    )
