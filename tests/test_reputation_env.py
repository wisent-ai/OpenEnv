"""Environment integration tests for the reputation system."""
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
    sys.modules.setdefault(_name, _mod)

import pytest

from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_DD_PAYOFF,
)
from constant_definitions.var.meta.reputation_constants import (
    META_KEY_REPUTATION,
    META_KEY_INTERACTION_COUNT,
    META_KEY_GOSSIP_HISTORY,
    META_KEY_COOPERATION_RATE,
    RATING_TRUSTWORTHY,
    VARIANT_GOSSIP,
)
from env.models import GameAction
from env.environment import KantEnvironment
from env.reputation.reputation_env import ReputationEnvironment
from common.meta.memory_store import CogneeMemoryStore

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE

_CC = float(PD_CC_PAYOFF)
_DD = float(PD_DD_PAYOFF)


@pytest.fixture()
def store() -> CogneeMemoryStore:
    return CogneeMemoryStore()


@pytest.fixture()
def rep_env(store: CogneeMemoryStore) -> ReputationEnvironment:
    return ReputationEnvironment(memory_store=store, env=KantEnvironment())


class TestReputationEnvReset:
    def test_reset_injects_reputation(self, rep_env: ReputationEnvironment) -> None:
        obs = rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
        )
        assert META_KEY_REPUTATION in obs.metadata
        assert META_KEY_INTERACTION_COUNT in obs.metadata

    def test_reputation_has_default_score(self, rep_env: ReputationEnvironment) -> None:
        obs = rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
        )
        rep = obs.metadata[META_KEY_REPUTATION]
        assert "score" in rep
        assert rep[META_KEY_INTERACTION_COUNT] == _ZERO


class TestReputationEnvStep:
    def test_step_with_gossip_action(self, rep_env: ReputationEnvironment) -> None:
        rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
        )
        action = GameAction(action="gossip_trustworthy_cooperate")
        obs = rep_env.step(action)
        assert obs.current_round == _ONE
        assert META_KEY_REPUTATION in obs.metadata

    def test_gossip_recorded_in_store(
        self,
        rep_env: ReputationEnvironment,
        store: CogneeMemoryStore,
    ) -> None:
        rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
        )
        rep_env.step(GameAction(action="gossip_trustworthy_cooperate"))
        stats = store.get_stats("always_cooperate")
        history = stats.get(META_KEY_GOSSIP_HISTORY, [])
        assert len(history) == _ONE
        assert history[_ZERO]["rating"] == RATING_TRUSTWORTHY


class TestReputationEnvEpisode:
    def test_episode_recording_on_done(
        self,
        rep_env: ReputationEnvironment,
        store: CogneeMemoryStore,
    ) -> None:
        rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
            num_rounds=_ONE,
        )
        obs = rep_env.step(
            GameAction(action="gossip_trustworthy_cooperate"),
        )
        assert obs.done
        stats = store.get_stats("always_cooperate")
        assert stats[META_KEY_INTERACTION_COUNT] == _ONE

    def test_reputation_updates_across_episodes(
        self,
        rep_env: ReputationEnvironment,
        store: CogneeMemoryStore,
    ) -> None:
        rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
            num_rounds=_ONE,
        )
        rep_env.step(GameAction(action="gossip_trustworthy_cooperate"))

        obs = rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
            num_rounds=_ONE,
        )
        rep = obs.metadata[META_KEY_REPUTATION]
        assert rep[META_KEY_INTERACTION_COUNT] == _ONE

        rep_env.step(GameAction(action="gossip_neutral_cooperate"))
        stats = store.get_stats("always_cooperate")
        assert stats[META_KEY_INTERACTION_COUNT] == _TWO


class TestReputationEnvState:
    def test_state_delegates_to_inner(
        self, rep_env: ReputationEnvironment,
    ) -> None:
        rep_env.reset(
            game="gossip_prisoners_dilemma",
            strategy="always_cooperate",
        )
        state = rep_env.state
        assert state.game_name == "gossip_prisoners_dilemma"
