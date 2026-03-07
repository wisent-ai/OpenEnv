"""Tests for the CoalitionEnvironment."""
import sys
import types

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")
class _EnvironmentStub:
    def __init_subclass__(cls, **kwargs: object) -> None: super().__init_subclass__(**kwargs)
    def __class_getitem__(cls, params: object) -> type: return cls
    def __init__(self) -> None: pass
_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv_stub.core = _core_stub  # type: ignore[attr-defined]
_core_stub.env_server = _server_stub  # type: ignore[attr-defined]
_server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
for _name, _mod in [("openenv", _openenv_stub), ("openenv.core", _core_stub),
                     ("openenv.core.env_server", _server_stub),
                     ("openenv.core.env_server.interfaces", _iface_stub)]:
    sys.modules[_name] = _mod

import pytest
from constant_definitions.nplayer.coalition_constants import (
    COALITION_PHASE_NEGOTIATE, COALITION_PHASE_ACTION,
    ENFORCEMENT_PENALTY, CARTEL_NUM_PLAYERS, CARTEL_COLLUDE_HIGH,
    COMMONS_LOW_SUSTAINABLE,
)
from common.games_meta.coalition_config import COALITION_GAMES
import common.games_meta.coalition_config  # noqa: F401
from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionObservation, CoalitionProposal,
)
from env.nplayer.coalition.environment import CoalitionEnvironment
from env.nplayer.coalition.strategies import COALITION_STRATEGIES
from env.nplayer.models import NPlayerAction

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE
_SEVEN = _FIVE + _TWO
_ZERO_F = float()


class TestCoalitionGameRegistry:
    def test_all_seven_registered(self) -> None:
        expected = {"coalition_cartel", "coalition_alliance", "coalition_voting",
                    "coalition_ostracism", "coalition_resource_trading",
                    "coalition_rule_voting", "coalition_commons"}
        assert expected.issubset(set(COALITION_GAMES.keys()))

    def test_strategies_registered(self) -> None:
        expected = {"coalition_random", "coalition_loyal",
                    "coalition_betrayer", "coalition_conditional"}
        assert expected.issubset(set(COALITION_STRATEGIES.keys()))


class TestCoalitionEnvironmentReset:
    def test_reset_returns_observation(self) -> None:
        env = CoalitionEnvironment()
        obs = env.reset("coalition_cartel")
        assert isinstance(obs, CoalitionObservation)
        assert obs.base.done is False
        assert obs.base.num_players == CARTEL_NUM_PLAYERS
        assert obs.phase == COALITION_PHASE_NEGOTIATE
        assert len(obs.active_players) == CARTEL_NUM_PLAYERS


class TestPhaseEnforcement:
    def test_action_before_negotiate_raises(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        with pytest.raises(RuntimeError, match="action phase"):
            env.action_step(NPlayerAction(action="collude"))

    def test_negotiate_after_negotiate_raises(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        env.negotiate_step(CoalitionAction())
        with pytest.raises(RuntimeError, match="negotiate phase"):
            env.negotiate_step(CoalitionAction())


class TestNegotiateActionCycle:
    def test_full_round(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        obs = env.negotiate_step(CoalitionAction())
        assert obs.phase == COALITION_PHASE_ACTION
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.base.done is True
        assert len(obs.coalition_history) == _ONE

    def test_multi_round(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        env.negotiate_step(CoalitionAction())
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.phase == COALITION_PHASE_NEGOTIATE
        env.negotiate_step(CoalitionAction())
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.base.done is True


class TestCoalitionFormation:
    def test_proposal_accepted_by_loyal(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO], agreed_action="collude",
        )
        obs = env.negotiate_step(CoalitionAction(proposals=[prop]))
        assert any(_ZERO in c.members for c in obs.active_coalitions)


class TestEnforcementModes:
    def test_penalty_on_betrayer(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_betrayer"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO, _THREE],
            agreed_action="collude",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="collude"))
        assert len(obs.coalition_history[-_ONE].defectors) > _ZERO

    def test_cheap_talk_no_penalty(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_alliance", num_rounds=_ONE,
                  coalition_strategies=["coalition_betrayer"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO], agreed_action="support",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="support"))
        assert all(p == _ZERO_F for p in obs.coalition_history[-_ONE].penalties)

    def test_binding_overrides_action(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_voting", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO, _THREE, _FOUR],
            agreed_action="vote_A",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="vote_B"))
        assert obs.base.last_round.actions[_ZERO] == "vote_A"


class TestStrategiesEndToEnd:
    def test_loyal_honours(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO, _THREE],
            agreed_action="collude",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="collude"))
        for i in range(_ONE, CARTEL_NUM_PLAYERS):
            assert obs.base.last_round.actions[i] == "collude"

    def test_betrayer_defects(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_betrayer"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO, _THREE],
            agreed_action="collude",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="collude"))
        for i in range(_ONE, CARTEL_NUM_PLAYERS):
            assert obs.base.last_round.actions[i] == "compete"


class TestAddRemovePlayers:
    def test_remove_player(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", coalition_strategies=["coalition_loyal"])
        env.remove_player(_THREE)
        assert _THREE not in env.active_players
        assert len(env.active_players) == _THREE

    def test_removed_player_gets_zero_payoff(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        env.remove_player(_THREE)
        env.negotiate_step(CoalitionAction())
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.adjusted_scores[_THREE] == pytest.approx(_ZERO_F)

    def test_add_player_back(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        env.remove_player(_TWO)
        env.negotiate_step(CoalitionAction())
        env.action_step(NPlayerAction(action="collude"))
        env.add_player(_TWO)
        assert _TWO in env.active_players
        env.negotiate_step(CoalitionAction())
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.base.done is True

    def test_remove_during_action_raises(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        env.negotiate_step(CoalitionAction())
        with pytest.raises(RuntimeError, match="negotiate phase"):
            env.remove_player(_ONE)

    def test_add_already_active_raises(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        with pytest.raises(ValueError, match="already active"):
            env.add_player(_ONE)

    def test_remove_already_inactive_raises(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        env.remove_player(_ONE)
        with pytest.raises(ValueError, match="already inactive"):
            env.remove_player(_ONE)

    def test_add_with_new_strategy(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        env.remove_player(_ONE)
        env.negotiate_step(CoalitionAction())
        env.action_step(NPlayerAction(action="collude"))
        env.add_player(_ONE, strategy="coalition_betrayer")
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE], agreed_action="collude",
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.base.last_round.actions[_ONE] == "compete"

    def test_active_players_in_observation(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel")
        env.remove_player(_TWO)
        obs = env.negotiate_step(CoalitionAction())
        assert _TWO not in obs.active_players


class TestAgentDrivenExclusion:
    def test_exclude_via_proposal(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO],
            agreed_action="collude", exclude_target=_THREE,
        )
        obs = env.negotiate_step(CoalitionAction(proposals=[prop]))
        assert _THREE not in obs.active_players

    def test_include_via_proposal(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        env.remove_player(_THREE)
        env.negotiate_step(CoalitionAction())
        env.action_step(NPlayerAction(action="collude"))
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO],
            agreed_action="collude", include_target=_THREE,
        )
        obs = env.negotiate_step(CoalitionAction(proposals=[prop]))
        assert _THREE in obs.active_players

    def test_excluded_player_gets_zero(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO],
            agreed_action="collude", exclude_target=_THREE,
        )
        env.negotiate_step(CoalitionAction(proposals=[prop]))
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.adjusted_scores[_THREE] == pytest.approx(_ZERO_F)
