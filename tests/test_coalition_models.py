"""Tests for coalition data models."""
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

from constant_definitions.nplayer.coalition_constants import (
    COALITION_PHASE_NEGOTIATE,
    COALITION_PHASE_ACTION,
    ENFORCEMENT_CHEAP_TALK,
    ENFORCEMENT_PENALTY,
)
from env.nplayer.coalition.models import (
    CoalitionProposal,
    CoalitionResponse,
    ActiveCoalition,
    CoalitionRoundResult,
    CoalitionObservation,
    CoalitionAction,
)
from env.nplayer.models import NPlayerObservation

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_ZERO_F = float()
_TWO_F = float(_TWO)
_THREE_F = float(_THREE)


class TestCoalitionProposal:
    def test_create(self) -> None:
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE, _TWO], agreed_action="collude",
        )
        assert prop.proposer == _ZERO
        assert prop.members == [_ZERO, _ONE, _TWO]
        assert prop.agreed_action == "collude"
        assert prop.side_payment == _ZERO_F

    def test_with_side_payment(self) -> None:
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE],
            agreed_action="collude", side_payment=_TWO_F,
        )
        assert prop.side_payment == _TWO_F


class TestCoalitionResponse:
    def test_create(self) -> None:
        resp = CoalitionResponse(
            responder=_ONE, proposal_index=_ZERO, accepted=True,
        )
        assert resp.responder == _ONE
        assert resp.proposal_index == _ZERO
        assert resp.accepted is True

    def test_reject(self) -> None:
        resp = CoalitionResponse(
            responder=_TWO, proposal_index=_ZERO, accepted=False,
        )
        assert resp.accepted is False


class TestActiveCoalition:
    def test_create(self) -> None:
        ac = ActiveCoalition(
            members=[_ZERO, _ONE, _TWO], agreed_action="collude",
        )
        assert ac.members == [_ZERO, _ONE, _TWO]
        assert ac.agreed_action == "collude"
        assert ac.side_payment == _ZERO_F


class TestCoalitionRoundResult:
    def test_create(self) -> None:
        result = CoalitionRoundResult(round_number=_ONE)
        assert result.round_number == _ONE
        assert result.proposals == []
        assert result.responses == []
        assert result.active_coalitions == []
        assert result.defectors == []
        assert result.penalties == []
        assert result.side_payments == []

    def test_with_defectors(self) -> None:
        result = CoalitionRoundResult(
            round_number=_ONE,
            defectors=[_TWO],
            penalties=[_ZERO_F, _ZERO_F, _THREE_F],
        )
        assert result.defectors == [_TWO]
        assert len(result.penalties) == _THREE


class TestCoalitionObservation:
    def test_defaults(self) -> None:
        obs = CoalitionObservation()
        assert obs.phase == COALITION_PHASE_NEGOTIATE
        assert obs.active_coalitions == []
        assert obs.pending_proposals == []
        assert obs.coalition_history == []
        assert obs.enforcement == ENFORCEMENT_CHEAP_TALK
        assert obs.adjusted_scores == []

    def test_with_base(self) -> None:
        base = NPlayerObservation(
            game_name="coalition_cartel",
            num_players=_THREE,
            available_actions=["collude", "compete"],
        )
        obs = CoalitionObservation(
            base=base, phase=COALITION_PHASE_ACTION,
            enforcement=ENFORCEMENT_PENALTY,
        )
        assert obs.base.game_name == "coalition_cartel"
        assert obs.base.num_players == _THREE
        assert obs.phase == COALITION_PHASE_ACTION

    def test_with_coalitions(self) -> None:
        ac = ActiveCoalition(
            members=[_ZERO, _ONE], agreed_action="collude",
        )
        obs = CoalitionObservation(active_coalitions=[ac])
        assert len(obs.active_coalitions) == _ONE
        assert obs.active_coalitions[_ZERO].agreed_action == "collude"


class TestCoalitionAction:
    def test_empty(self) -> None:
        action = CoalitionAction()
        assert action.proposals == []
        assert action.responses == []

    def test_with_proposals_and_responses(self) -> None:
        prop = CoalitionProposal(
            proposer=_ZERO, members=[_ZERO, _ONE], agreed_action="collude",
        )
        resp = CoalitionResponse(
            responder=_ZERO, proposal_index=_ZERO, accepted=True,
        )
        action = CoalitionAction(proposals=[prop], responses=[resp])
        assert len(action.proposals) == _ONE
        assert len(action.responses) == _ONE
        assert action.proposals[_ZERO].proposer == _ZERO
        assert action.responses[_ZERO].accepted is True
