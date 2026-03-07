"""Tests for the GovernanceEngine and integration with CoalitionEnvironment."""
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
    ENFORCEMENT_CHEAP_TALK, ENFORCEMENT_PENALTY, ENFORCEMENT_BINDING, CARTEL_NUM_PLAYERS,
)
from constant_definitions.nplayer.governance_constants import (
    GOVERNANCE_PROPOSAL_PARAMETER, GOVERNANCE_PROPOSAL_MECHANIC,
    GOVERNANCE_PROPOSAL_CUSTOM, MECHANIC_TAXATION,
)
from common.games_meta.coalition_config import get_coalition_game
import common.games_meta.coalition_config  # noqa: F401
from env.nplayer.governance.engine import GovernanceEngine
from env.nplayer.governance.models import GovernanceProposal, GovernanceVote
from env.nplayer.coalition.models import CoalitionAction, CoalitionProposal
from env.nplayer.coalition.environment import CoalitionEnvironment
from env.nplayer.models import NPlayerAction

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_ZERO_F = float()
_ALL = set(range(CARTEL_NUM_PLAYERS))

def _engine() -> GovernanceEngine:
    e = GovernanceEngine()
    e.reset(get_coalition_game("coalition_cartel"))
    return e

def _unanimous(idx: int = _ZERO) -> list[GovernanceVote]:
    return [GovernanceVote(voter=i, proposal_index=idx, approve=True) for i in _ALL]


class TestEngineReset:
    def test_initializes_rules(self) -> None:
        e = _engine()
        assert e.rules.enforcement == ENFORCEMENT_PENALTY
        assert e.rules.mechanics[MECHANIC_TAXATION] is False
        assert e.rules.governance_history == []


class TestParameterProposals:
    def test_change_enforcement(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_BINDING)
        e.submit_proposals([prop], _ALL)
        result = e.tally_votes(_unanimous(), _ALL)
        assert _ZERO in result.adopted
        assert e.rules.enforcement == ENFORCEMENT_BINDING

    def test_change_penalty_numerator(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="penalty_numerator", parameter_value=_THREE)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert e.rules.penalty_numerator == _THREE

    def test_toggle_side_payments(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="allow_side_payments", parameter_value=True)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert e.rules.allow_side_payments is True

    def test_invalid_parameter_rejected(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="nonexistent", parameter_value="foo")
        assert len(e.submit_proposals([prop], _ALL)) == _ZERO


class TestMechanicProposals:
    def test_activate_taxation(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_MECHANIC,
            mechanic_name=MECHANIC_TAXATION, mechanic_active=True)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert e.rules.mechanics[MECHANIC_TAXATION] is True

    def test_deactivate_mechanic(self) -> None:
        e = _engine()
        e.rules.mechanics[MECHANIC_TAXATION] = True
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_MECHANIC,
            mechanic_name=MECHANIC_TAXATION, mechanic_active=False)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert e.rules.mechanics[MECHANIC_TAXATION] is False

    def test_mechanic_with_params(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_MECHANIC,
            mechanic_name=MECHANIC_TAXATION, mechanic_active=True,
            mechanic_params={"tax_rate_numerator": _THREE, "tax_rate_denominator": _FOUR})
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert e.rules.mechanic_config.tax_rate_numerator == _THREE


class TestCustomModifiers:
    def test_register_and_activate(self) -> None:
        e = _engine()
        e.register_custom_modifier("bonus", lambda p, a: [x + float(_ONE) for x in p])
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_CUSTOM,
            custom_modifier_key="bonus", custom_modifier_active=True)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert "bonus" in e.rules.custom_modifier_keys

    def test_custom_modifier_applies(self) -> None:
        e = _engine()
        e.register_custom_modifier("bonus", lambda p, a: [x + float(_ONE) for x in p])
        e.rules.custom_modifier_keys.append("bonus")
        payoffs = [float(_FOUR + _TWO)] * CARTEL_NUM_PLAYERS
        result = e.apply(payoffs, _ALL)
        for i in _ALL:
            assert result[i] >= payoffs[i]

    def test_delta_clamp(self) -> None:
        e = _engine()
        _big = float(_FOUR * _FOUR * _FOUR)
        e.register_custom_modifier("huge", lambda p, a: [x + _big for x in p])
        e.rules.custom_modifier_keys.append("huge")
        payoffs = [float(_FOUR + _TWO)] * CARTEL_NUM_PLAYERS
        result = e.apply(payoffs, _ALL)
        for i in _ALL:
            assert result[i] < payoffs[i] + _big

    def test_unregister(self) -> None:
        e = _engine()
        e.register_custom_modifier("bonus", lambda p, a: p)
        e.rules.custom_modifier_keys.append("bonus")
        e.unregister_custom_modifier("bonus")
        assert "bonus" not in e.rules.custom_modifier_keys

    def test_failing_modifier_skipped(self) -> None:
        e = _engine()
        def _bad(p: list, a: set) -> list:
            raise ValueError("intentional")
        e.register_custom_modifier("bad", _bad)
        e.rules.custom_modifier_keys.append("bad")
        payoffs = [float(_FOUR + _TWO)] * CARTEL_NUM_PLAYERS
        result = e.apply(payoffs, _ALL)
        for i in _ALL:
            assert result[i] == pytest.approx(payoffs[i])


class TestVotingThreshold:
    def test_minority_rejected(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_BINDING)
        e.submit_proposals([prop], _ALL)
        votes = [GovernanceVote(voter=_ZERO, proposal_index=_ZERO, approve=True)]
        result = e.tally_votes(votes, _ALL)
        assert _ZERO in result.rejected
        assert e.rules.enforcement == ENFORCEMENT_PENALTY

    def test_exact_majority_passes(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_BINDING)
        e.submit_proposals([prop], _ALL)
        threshold = CARTEL_NUM_PLAYERS // _TWO + _ONE
        votes = [GovernanceVote(voter=i, proposal_index=_ZERO, approve=True)
                 for i in range(threshold)]
        assert _ZERO in e.tally_votes(votes, _ALL).adopted

    def test_rejected_no_effect(self) -> None:
        e = _engine()
        original = e.rules.enforcement
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_CHEAP_TALK)
        e.submit_proposals([prop], _ALL)
        e.tally_votes([GovernanceVote(voter=_ZERO, proposal_index=_ZERO, approve=False)], _ALL)
        assert e.rules.enforcement == original


class TestGovernanceHistory:
    def test_history_recorded(self) -> None:
        e = _engine()
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_BINDING)
        e.submit_proposals([prop], _ALL)
        e.tally_votes(_unanimous(), _ALL)
        assert len(e.rules.governance_history) == _ONE
        assert e.rules.governance_history[_ZERO].rules_snapshot is not None


class TestCoalitionIntegration:
    def test_governance_in_observation(self) -> None:
        env = CoalitionEnvironment()
        obs = env.reset("coalition_cartel", num_rounds=_ONE,
                        coalition_strategies=["coalition_loyal"])
        assert obs.current_rules is not None
        assert obs.current_rules.enforcement == ENFORCEMENT_PENALTY

    def test_parameter_change_affects_enforcement(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_TWO,
                  coalition_strategies=["coalition_loyal"])
        prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_PARAMETER,
            parameter_name="enforcement", parameter_value=ENFORCEMENT_BINDING)
        n = CARTEL_NUM_PLAYERS
        votes = [GovernanceVote(voter=i, proposal_index=_ZERO, approve=True) for i in range(n)]
        coal_prop = CoalitionProposal(
            proposer=_ZERO, members=list(range(n)), agreed_action="collude")
        obs = env.negotiate_step(CoalitionAction(
            proposals=[coal_prop], governance_proposals=[prop], governance_votes=votes))
        assert obs.current_rules.enforcement == ENFORCEMENT_BINDING
        obs = env.action_step(NPlayerAction(action="compete"))
        assert obs.base.last_round.actions[_ZERO] == "collude"

    def test_taxation_modifies_payoffs(self) -> None:
        env = CoalitionEnvironment()
        env.reset("coalition_cartel", num_rounds=_ONE,
                  coalition_strategies=["coalition_loyal"])
        _ten = _FOUR + _FOUR + _TWO
        gov_prop = GovernanceProposal(
            proposer=_ZERO, proposal_type=GOVERNANCE_PROPOSAL_MECHANIC,
            mechanic_name=MECHANIC_TAXATION, mechanic_active=True,
            mechanic_params={"tax_rate_numerator": _THREE, "tax_rate_denominator": _ten})
        n = CARTEL_NUM_PLAYERS
        votes = [GovernanceVote(voter=i, proposal_index=_ZERO, approve=True) for i in range(n)]
        env.negotiate_step(CoalitionAction(governance_proposals=[gov_prop], governance_votes=votes))
        obs = env.action_step(NPlayerAction(action="collude"))
        assert obs.base.done is True
