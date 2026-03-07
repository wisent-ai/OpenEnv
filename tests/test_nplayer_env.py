"""Tests for the N-player environment core."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

# Stub the openenv package
_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")
class _EnvironmentStub:
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

from constant_definitions.nplayer.nplayer_constants import (
    NVD_BENEFIT,
    NVD_COST,
    NPLAYER_DEFAULT_ROUNDS,
)
from common.games_meta.nplayer_config import NPLAYER_GAMES
import common.games_meta.nplayer_games  # noqa: F401 -- register built-ins
from env.nplayer.models import NPlayerAction, NPlayerObservation
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.strategies import NPLAYER_STRATEGIES

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE


class TestNPlayerGameRegistry:
    def test_public_goods_registered(self) -> None:
        assert "nplayer_public_goods" in NPLAYER_GAMES

    def test_volunteer_dilemma_registered(self) -> None:
        assert "nplayer_volunteer_dilemma" in NPLAYER_GAMES

    def test_el_farol_registered(self) -> None:
        assert "nplayer_el_farol" in NPLAYER_GAMES

    def test_builtin_count(self) -> None:
        assert len(NPLAYER_GAMES) >= _THREE


class TestNPlayerEnvironment:
    def test_reset_returns_observation(self) -> None:
        env = NPlayerEnvironment()
        obs = env.reset("nplayer_volunteer_dilemma")
        assert isinstance(obs, NPlayerObservation)
        assert obs.done is False
        assert obs.num_players == _FIVE
        assert obs.player_index == _ZERO
        assert len(obs.scores) == _FIVE
        assert obs.current_round == _ZERO
        assert obs.total_rounds == NPLAYER_DEFAULT_ROUNDS

    def test_step_returns_observation(self) -> None:
        env = NPlayerEnvironment()
        env.reset("nplayer_volunteer_dilemma")
        obs = env.step(NPlayerAction(action="volunteer"))
        assert isinstance(obs, NPlayerObservation)
        assert obs.current_round == _ONE
        assert len(obs.history) == _ONE
        assert obs.last_round is not None
        assert len(obs.last_round.actions) == _FIVE
        assert len(obs.last_round.payoffs) == _FIVE

    def test_episode_completion(self) -> None:
        env = NPlayerEnvironment()
        env.reset("nplayer_volunteer_dilemma", num_rounds=_THREE)
        for _ in range(_THREE):
            obs = env.step(NPlayerAction(action="volunteer"))
        assert obs.done is True
        assert env.state.is_done is True

    def test_step_after_done_raises(self) -> None:
        env = NPlayerEnvironment()
        env.reset("nplayer_volunteer_dilemma", num_rounds=_ONE)
        env.step(NPlayerAction(action="volunteer"))
        with pytest.raises(RuntimeError, match="finished"):
            env.step(NPlayerAction(action="volunteer"))

    def test_step_before_reset_raises(self) -> None:
        env = NPlayerEnvironment()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(NPlayerAction(action="volunteer"))

    def test_invalid_action_raises(self) -> None:
        env = NPlayerEnvironment()
        env.reset("nplayer_volunteer_dilemma")
        with pytest.raises(ValueError, match="Invalid"):
            env.step(NPlayerAction(action="invalid_action"))

    def test_custom_rounds(self) -> None:
        env = NPlayerEnvironment()
        obs = env.reset("nplayer_volunteer_dilemma", num_rounds=_TWO)
        assert obs.total_rounds == _TWO

    def test_scores_accumulate(self) -> None:
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_TWO,
            opponent_strategies=["always_cooperate"],
        )
        env.step(NPlayerAction(action="volunteer"))
        obs = env.step(NPlayerAction(action="volunteer"))
        expected_per_round = float(NVD_BENEFIT - NVD_COST)
        expected_total = expected_per_round * _TWO
        assert obs.scores[_ZERO] == pytest.approx(expected_total)


class TestNPlayerStrategies:
    def test_all_strategies_registered(self) -> None:
        expected = {"random", "always_cooperate", "always_defect", "tit_for_tat", "adaptive"}
        assert expected.issubset(set(NPLAYER_STRATEGIES.keys()))

    def test_always_cooperate(self) -> None:
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_strategies=["always_cooperate"],
        )
        obs = env.step(NPlayerAction(action="abstain"))
        for i in range(_ONE, _FIVE):
            assert obs.last_round.actions[i] == "volunteer"

    def test_always_defect(self) -> None:
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_strategies=["always_defect"],
        )
        obs = env.step(NPlayerAction(action="volunteer"))
        for i in range(_ONE, _FIVE):
            assert obs.last_round.actions[i] == "abstain"


class TestOpponentFns:
    def test_opponent_fn_used(self) -> None:
        def always_volunteer(obs: NPlayerObservation) -> NPlayerAction:
            return NPlayerAction(action="volunteer")
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_fns=[always_volunteer] * _FOUR,
        )
        obs = env.step(NPlayerAction(action="abstain"))
        for i in range(_ONE, _FIVE):
            assert obs.last_round.actions[i] == "volunteer"

    def test_mixed_fns_and_strategies(self) -> None:
        def always_volunteer(obs: NPlayerObservation) -> NPlayerAction:
            return NPlayerAction(action="volunteer")
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_strategies=["always_defect"],
            opponent_fns=[always_volunteer, None, always_volunteer, None],
        )
        obs = env.step(NPlayerAction(action="abstain"))
        assert obs.last_round.actions[_ONE] == "volunteer"
        assert obs.last_round.actions[_THREE] == "volunteer"
        assert obs.last_round.actions[_TWO] == "abstain"
        assert obs.last_round.actions[_FOUR] == "abstain"

    def test_invalid_opponent_fn_action_raises(self) -> None:
        def bad_fn(obs: NPlayerObservation) -> NPlayerAction:
            return NPlayerAction(action="nonexistent_xyz")
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_fns=[bad_fn, None, None, None],
        )
        with pytest.raises(ValueError, match="invalid"):
            env.step(NPlayerAction(action="volunteer"))

    def test_opponent_fn_receives_correct_player_index(self) -> None:
        indices_seen: list[int] = []
        def capture_index(obs: NPlayerObservation) -> NPlayerAction:
            indices_seen.append(obs.player_index)
            return NPlayerAction(action="volunteer")
        env = NPlayerEnvironment()
        env.reset(
            "nplayer_volunteer_dilemma",
            num_rounds=_ONE,
            opponent_fns=[capture_index] * _FOUR,
        )
        env.step(NPlayerAction(action="volunteer"))
        assert indices_seen == [_ONE, _TWO, _THREE, _FOUR]
