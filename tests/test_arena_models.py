"""Tests for arena Pydantic data models."""
import sys
import types

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

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
    ("openenv", _openenv_stub),
    ("openenv.core", _core_stub),
    ("openenv.core.env_server", _server_stub),
    ("openenv.core.env_server.interfaces", _iface_stub),
]:
    sys.modules[_n] = _m

import pytest

from env.arena.models import (
    ArenaMessage,
    ArenaModelProfile,
    ArenaProposal,
    ArenaVote,
    ArenaRoundResult,
    ArenaState,
)
from constant_definitions.arena.arena_constants import (
    PROPOSAL_BAN,
    PROPOSAL_ADD,
    PROPOSAL_RULE,
    PROPOSAL_NEW_GAME,
)
from constant_definitions.arena.messaging_constants import (
    MSG_TYPE_DIRECT,
    MSG_TYPE_BROADCAST,
    MSG_TYPE_GOSSIP,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_ZERO_F = float()
_HALF_F = float(_ONE) / float(_TWO)

_MODEL_A = "alpha"
_MODEL_B = "beta"
_MODEL_C = "gamma"
_MODEL_API = "api"
_MODEL_LOCAL = "local"
_MODEL_ID_A = "modelA"
_MODEL_ID_B = "modelB"
_GAME_NAME = "prisoners_dilemma"
_RULE_TEXT = "no collusion"
_GAME_DEF_NAME = "ultimatum"
_RULE_LABEL = "ruleA"


class TestArenaMessage:
    def test_minimal(self) -> None:
        msg = ArenaMessage(sender=_MODEL_A)
        assert msg.sender == _MODEL_A
        assert msg.recipients == []
        assert msg.msg_type == MSG_TYPE_DIRECT
        assert msg.content == ""
        assert msg.gossip_target is None
        assert msg.gossip_rating is None

    def test_broadcast(self) -> None:
        msg = ArenaMessage(
            sender=_MODEL_A,
            recipients=[_MODEL_B, _MODEL_C],
            msg_type=MSG_TYPE_BROADCAST,
            content="hello all",
        )
        assert msg.msg_type == MSG_TYPE_BROADCAST
        assert len(msg.recipients) == _TWO

    def test_gossip_fields(self) -> None:
        msg = ArenaMessage(
            sender=_MODEL_A,
            msg_type=MSG_TYPE_GOSSIP,
            gossip_target=_MODEL_B,
            gossip_rating="positive",
        )
        assert msg.gossip_target == _MODEL_B
        assert msg.gossip_rating == "positive"

    def test_serialization_roundtrip(self) -> None:
        msg = ArenaMessage(sender=_MODEL_A, recipients=[_MODEL_B], content="hi")
        data = msg.model_dump()
        restored = ArenaMessage(**data)
        assert restored.sender == msg.sender
        assert restored.recipients == msg.recipients


class TestArenaModelProfile:
    def test_defaults(self) -> None:
        profile = ArenaModelProfile(model_id=_MODEL_ID_A, model_type=_MODEL_API)
        assert profile.reputation == _HALF_F
        assert profile.honesty == _HALF_F
        assert profile.fairness == _HALF_F
        assert profile.games_played == _ZERO
        assert profile.is_active is True
        assert profile.banned_round is None
        assert profile.cooperation_history == []
        assert profile.peer_ratings == []

    def test_custom_values(self) -> None:
        profile = ArenaModelProfile(
            model_id=_MODEL_ID_B,
            model_type=_MODEL_LOCAL,
            reputation=_HALF_F,
            games_played=_THREE,
            is_active=False,
            banned_round=_TWO,
        )
        assert profile.games_played == _THREE
        assert profile.is_active is False
        assert profile.banned_round == _TWO

    def test_cooperation_history_stored(self) -> None:
        profile = ArenaModelProfile(
            model_id=_MODEL_ID_A,
            cooperation_history=[_HALF_F, _HALF_F],
        )
        assert len(profile.cooperation_history) == _TWO


class TestArenaProposal:
    def test_defaults(self) -> None:
        prop = ArenaProposal(proposer=_MODEL_A)
        assert prop.proposal_type == PROPOSAL_BAN
        assert prop.target_model is None
        assert prop.rule_description is None
        assert prop.game_definition is None

    def test_ban_proposal(self) -> None:
        prop = ArenaProposal(
            proposer=_MODEL_A,
            proposal_type=PROPOSAL_BAN,
            target_model=_MODEL_B,
        )
        assert prop.target_model == _MODEL_B
        assert prop.proposal_type == PROPOSAL_BAN

    def test_rule_proposal(self) -> None:
        prop = ArenaProposal(
            proposer=_MODEL_A,
            proposal_type=PROPOSAL_RULE,
            rule_description=_RULE_TEXT,
        )
        assert prop.proposal_type == PROPOSAL_RULE
        assert prop.rule_description == _RULE_TEXT

    def test_new_game_proposal(self) -> None:
        game_def = {"name": _GAME_DEF_NAME, "rounds": _THREE}
        prop = ArenaProposal(
            proposer=_MODEL_A,
            proposal_type=PROPOSAL_NEW_GAME,
            game_definition=game_def,
        )
        assert prop.proposal_type == PROPOSAL_NEW_GAME
        assert prop.game_definition == game_def

    def test_add_proposal(self) -> None:
        prop = ArenaProposal(proposer=_MODEL_A, proposal_type=PROPOSAL_ADD)
        assert prop.proposal_type == PROPOSAL_ADD


class TestArenaVote:
    def test_defaults(self) -> None:
        vote = ArenaVote(voter=_MODEL_A)
        assert vote.proposal_index == _ZERO
        assert vote.approve is True
        assert vote.weight == _HALF_F

    def test_reject_vote(self) -> None:
        vote = ArenaVote(voter=_MODEL_B, proposal_index=_ONE, approve=False)
        assert vote.approve is False
        assert vote.proposal_index == _ONE


class TestArenaRoundResult:
    def test_defaults(self) -> None:
        result = ArenaRoundResult(round_number=_ONE)
        assert result.round_number == _ONE
        assert result.messages == []
        assert result.proposals == []
        assert result.votes == []
        assert result.adopted == []
        assert result.game_results == []
        assert result.reputation_updates == {}

    def test_with_messages_and_votes(self) -> None:
        msg = ArenaMessage(sender=_MODEL_A, content="test")
        vote = ArenaVote(voter=_MODEL_B, approve=False)
        result = ArenaRoundResult(
            round_number=_TWO,
            messages=[msg],
            votes=[vote],
            adopted=[_ZERO],
        )
        assert len(result.messages) == _ONE
        assert len(result.votes) == _ONE
        assert result.adopted == [_ZERO]

    def test_reputation_updates_stored(self) -> None:
        result = ArenaRoundResult(
            round_number=_ONE,
            reputation_updates={_MODEL_A: _HALF_F},
        )
        assert result.reputation_updates[_MODEL_A] == _HALF_F


class TestArenaState:
    def test_defaults(self) -> None:
        state = ArenaState()
        assert state.round_number == _ZERO
        assert state.total_rounds == _ZERO
        assert state.roster == {}
        assert state.game_pool == []
        assert state.custom_games == []
        assert state.round_history == []
        assert state.active_rules == []

    def test_roster_with_profile(self) -> None:
        profile = ArenaModelProfile(model_id=_MODEL_ID_A, model_type=_MODEL_API)
        state = ArenaState(
            round_number=_ONE,
            total_rounds=_THREE,
            roster={_MODEL_ID_A: profile},
            game_pool=[_GAME_NAME],
        )
        assert state.roster[_MODEL_ID_A].model_id == _MODEL_ID_A
        assert len(state.game_pool) == _ONE

    def test_serialization_roundtrip(self) -> None:
        profile = ArenaModelProfile(model_id=_MODEL_ID_A)
        state = ArenaState(
            roster={_MODEL_ID_A: profile},
            active_rules=[_RULE_LABEL],
        )
        data = state.model_dump()
        restored = ArenaState(**data)
        assert _MODEL_ID_A in restored.roster
        assert restored.active_rules == [_RULE_LABEL]
