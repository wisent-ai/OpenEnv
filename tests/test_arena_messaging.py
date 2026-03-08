"""Tests for ArenaMessaging inter-model communication."""
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

    _iface_stub.Environment = _EnvironmentStub
    _openenv_stub.core = _core_stub
    _core_stub.env_server = _server_stub
    _server_stub.interfaces = _iface_stub
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

import sys as _sys
_sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

import pytest
from env.arena.messaging import ArenaMessaging
from env.arena.models import ArenaMessage
from constant_definitions.arena.messaging_constants import (
    MSG_TYPE_DIRECT,
    MSG_TYPE_BROADCAST,
    MSG_TYPE_GOSSIP,
    MAX_MESSAGES_PER_PHASE,
    MAX_MESSAGE_LENGTH,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE

_MODELS = ["alpha", "beta", "gamma"]
_ALPHA = _MODELS[_ZERO]
_BETA = _MODELS[_ONE]
_GAMMA = _MODELS[_TWO]

_ROUND_HEADER_PREFIX = "--- Round "
_ROUND_ONE_HEADER = _ROUND_HEADER_PREFIX + str(_ONE)
_ROUND_TWO_HEADER = _ROUND_HEADER_PREFIX + str(_TWO)


def _messaging() -> ArenaMessaging:
    m = ArenaMessaging()
    m.start_round(_ONE)
    return m


def _direct(sender: str, recipient: str, content: str = "hello") -> ArenaMessage:
    return ArenaMessage(
        sender=sender,
        recipients=[recipient],
        msg_type=MSG_TYPE_DIRECT,
        content=content,
    )


def _broadcast(sender: str, content: str = "announce") -> ArenaMessage:
    return ArenaMessage(sender=sender, msg_type=MSG_TYPE_BROADCAST, content=content)


def _gossip(
    sender: str,
    target: str,
    rating: str = "trustworthy",
) -> ArenaMessage:
    return ArenaMessage(
        sender=sender,
        msg_type=MSG_TYPE_GOSSIP,
        gossip_target=target,
        gossip_rating=rating,
    )


class TestSubmitMessage:
    def test_inactive_sender_rejected(self) -> None:
        m = _messaging()
        msg = _direct("outsider", _ALPHA)
        assert m.submit_message(msg, _MODELS) is False

    def test_active_sender_accepted(self) -> None:
        m = _messaging()
        assert m.submit_message(_direct(_ALPHA, _BETA), _MODELS) is True

    def test_message_count_limit(self) -> None:
        m = _messaging()
        for _ in range(MAX_MESSAGES_PER_PHASE):
            assert m.submit_message(_direct(_ALPHA, _BETA), _MODELS) is True
        assert m.submit_message(_direct(_ALPHA, _BETA), _MODELS) is False

    def test_limit_resets_on_new_round(self) -> None:
        m = _messaging()
        for _ in range(MAX_MESSAGES_PER_PHASE):
            m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        m.start_round(_TWO)
        assert m.submit_message(_direct(_ALPHA, _BETA), _MODELS) is True

    def test_long_content_truncated(self) -> None:
        m = _messaging()
        long_content = "x" * (MAX_MESSAGE_LENGTH + _TWO)
        msg = _direct(_ALPHA, _BETA, long_content)
        m.submit_message(msg, _MODELS)
        stored = m.get_messages_for(_BETA)[_ZERO]
        assert len(stored.content) == MAX_MESSAGE_LENGTH

    def test_exact_length_content_not_truncated(self) -> None:
        m = _messaging()
        content = "y" * MAX_MESSAGE_LENGTH
        msg = _direct(_ALPHA, _BETA, content)
        m.submit_message(msg, _MODELS)
        stored = m.get_messages_for(_BETA)[_ZERO]
        assert len(stored.content) == MAX_MESSAGE_LENGTH


class TestDirectVisibility:
    def test_recipient_sees_direct(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        assert len(m.get_messages_for(_BETA)) == _ONE

    def test_sender_sees_own_direct(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        assert len(m.get_messages_for(_ALPHA)) == _ONE

    def test_uninvolved_cannot_see_direct(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        assert len(m.get_messages_for(_GAMMA)) == _ZERO

    def test_multiple_recipients_both_see(self) -> None:
        m = _messaging()
        msg = ArenaMessage(
            sender=_ALPHA,
            recipients=[_BETA, _GAMMA],
            msg_type=MSG_TYPE_DIRECT,
            content="group note",
        )
        m.submit_message(msg, _MODELS)
        assert len(m.get_messages_for(_BETA)) == _ONE
        assert len(m.get_messages_for(_GAMMA)) == _ONE


class TestBroadcastVisibility:
    def test_all_models_see_broadcast(self) -> None:
        m = _messaging()
        m.submit_message(_broadcast(_ALPHA), _MODELS)
        for model in _MODELS:
            msgs = m.get_messages_for(model)
            assert len(msgs) == _ONE

    def test_broadcast_recipients_set_to_others(self) -> None:
        m = _messaging()
        msg = _broadcast(_ALPHA)
        m.submit_message(msg, _MODELS)
        all_msgs = m.end_round()
        recipients = all_msgs[_ZERO].recipients
        assert _ALPHA not in recipients
        assert _BETA in recipients
        assert _GAMMA in recipients

    def test_broadcast_separate_from_direct(self) -> None:
        m = _messaging()
        m.submit_message(_broadcast(_ALPHA), _MODELS)
        m.submit_message(_direct(_BETA, _ALPHA), _MODELS)
        assert len(m.get_messages_for(_ALPHA)) == _TWO
        assert len(m.get_messages_for(_GAMMA)) == _ONE


class TestGossipVisibility:
    def test_gossip_visible_to_all(self) -> None:
        m = _messaging()
        m.submit_message(_gossip(_ALPHA, _BETA), _MODELS)
        for model in _MODELS:
            assert len(m.get_messages_for(model)) == _ONE

    def test_get_gossip_about_filters_by_target(self) -> None:
        m = _messaging()
        m.submit_message(_gossip(_ALPHA, _BETA), _MODELS)
        m.submit_message(_gossip(_BETA, _GAMMA), _MODELS)
        assert len(m.get_gossip_about(_BETA)) == _ONE
        assert len(m.get_gossip_about(_GAMMA)) == _ONE
        assert len(m.get_gossip_about(_ALPHA)) == _ZERO

    def test_get_gossip_about_respects_round(self) -> None:
        m = _messaging()
        m.submit_message(_gossip(_ALPHA, _BETA), _MODELS)
        m.start_round(_TWO)
        m.submit_message(_gossip(_GAMMA, _BETA), _MODELS)
        assert len(m.get_gossip_about(_BETA, round_number=_ONE)) == _ONE
        assert len(m.get_gossip_about(_BETA, round_number=_TWO)) == _ONE


class TestEndRound:
    def test_end_round_returns_all_messages(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        m.submit_message(_broadcast(_GAMMA), _MODELS)
        result = m.end_round()
        assert len(result) == _TWO

    def test_end_round_is_copy(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        result = m.end_round()
        result.clear()
        assert len(m.end_round()) == _ONE

    def test_end_round_empty_before_any_message(self) -> None:
        m = ArenaMessaging()
        m.start_round(_ONE)
        assert m.end_round() == []


class TestGetMessagesForRound:
    def test_round_isolation(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA, "round one"), _MODELS)
        m.start_round(_TWO)
        m.submit_message(_direct(_ALPHA, _BETA, "round two"), _MODELS)
        r1 = m.get_messages_for(_BETA, round_number=_ONE)
        r2 = m.get_messages_for(_BETA, round_number=_TWO)
        assert len(r1) == _ONE
        assert r1[_ZERO].content == "round one"
        assert len(r2) == _ONE
        assert r2[_ZERO].content == "round two"

    def test_default_round_is_current(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA), _MODELS)
        assert m.get_messages_for(_BETA) == m.get_messages_for(_BETA, round_number=_ONE)


class TestBuildMessageContext:
    def test_context_contains_round_header(self) -> None:
        m = _messaging()
        m.submit_message(_broadcast(_ALPHA, "hello all"), _MODELS)
        ctx = m.build_message_context(_BETA, _ONE)
        assert _ROUND_ONE_HEADER in ctx

    def test_context_shows_broadcast_content(self) -> None:
        m = _messaging()
        m.submit_message(_broadcast(_ALPHA, "announcement"), _MODELS)
        ctx = m.build_message_context(_BETA, _ONE)
        assert "announcement" in ctx

    def test_context_shows_gossip_rating(self) -> None:
        m = _messaging()
        m.submit_message(_gossip(_ALPHA, _GAMMA, "reliable"), _MODELS)
        ctx = m.build_message_context(_BETA, _ONE)
        assert "reliable" in ctx
        assert _GAMMA in ctx

    def test_context_empty_when_no_messages(self) -> None:
        m = _messaging()
        assert m.build_message_context(_ALPHA, _ONE) == ""

    def test_context_spans_history_window(self) -> None:
        m = _messaging()
        m.submit_message(_broadcast(_ALPHA, "r1"), _MODELS)
        m.start_round(_TWO)
        m.submit_message(_broadcast(_BETA, "r2"), _MODELS)
        m.start_round(_THREE)
        ctx = m.build_message_context(_GAMMA, _THREE)
        assert _ROUND_ONE_HEADER in ctx
        assert _ROUND_TWO_HEADER in ctx

    def test_context_excludes_invisible_direct(self) -> None:
        m = _messaging()
        m.submit_message(_direct(_ALPHA, _BETA, "secret"), _MODELS)
        ctx = m.build_message_context(_GAMMA, _ONE)
        assert "secret" not in ctx
