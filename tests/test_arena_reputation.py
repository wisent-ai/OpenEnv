"""Tests for the ArenaReputation weighted reputation system."""
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

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from unittest.mock import MagicMock, patch
from env.arena.subsystems.reputation import ArenaReputation
from constant_definitions.arena.reputation_weights import (
    DEFAULT_ARENA_SCORE_NUMERATOR,
    DEFAULT_ARENA_SCORE_DENOMINATOR,
    VOTING_WEIGHT_FLOOR_NUMERATOR,
    VOTING_WEIGHT_FLOOR_DENOMINATOR,
    ARENA_DECAY_NUMERATOR,
    ARENA_DECAY_DENOMINATOR,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE

_ZERO_F = float()
_ONE_F = float(_ONE)
_HALF_F = _ONE_F / (_ONE + _ONE)

_DEFAULT = DEFAULT_ARENA_SCORE_NUMERATOR / DEFAULT_ARENA_SCORE_DENOMINATOR
_DECAY = ARENA_DECAY_NUMERATOR / ARENA_DECAY_DENOMINATOR
_FLOOR = VOTING_WEIGHT_FLOOR_NUMERATOR / VOTING_WEIGHT_FLOOR_DENOMINATOR

_W_COOP = _THREE / (_THREE * _THREE + _ONE)
_W_HONESTY = _THREE / (_THREE * _THREE + _ONE)
_W_FAIRNESS = _TWO / (_THREE * _THREE + _ONE)
_W_PEER = _TWO / (_THREE * _THREE + _ONE)

_EPS = float(_ONE) / (float(_ONE + _ONE) ** (_THREE * _THREE + _THREE))


def _make_rep() -> ArenaReputation:
    with patch(
        "env.arena.subsystems.reputation.CogneeMemoryStore",
        return_value=MagicMock(),
    ):
        return ArenaReputation()


class TestComputeReputationDefaults:
    def test_unknown_model_returns_half(self) -> None:
        rep = _make_rep()
        score = rep.compute_reputation("unknown_model")
        assert abs(score - _DEFAULT) < _EPS

    def test_weighted_sum_at_default_equals_default(self) -> None:
        rep = _make_rep()
        expected = (
            _DEFAULT * _W_COOP
            + _DEFAULT * _W_HONESTY
            + _DEFAULT * _W_FAIRNESS
            + _DEFAULT * _W_PEER
        )
        assert abs(rep.compute_reputation("x") - expected) < _EPS

    def test_weights_sum_to_one(self) -> None:
        total = _W_COOP + _W_HONESTY + _W_FAIRNESS + _W_PEER
        assert abs(total - _ONE_F) < _EPS


class TestUpdateCooperation:
    def test_single_update_ema(self) -> None:
        rep = _make_rep()
        rep.update_cooperation("m1", _ONE_F)
        expected = _DEFAULT * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("m1", "cooperation") - expected) < _EPS

    def test_update_toward_zero_decays(self) -> None:
        rep = _make_rep()
        rep.update_cooperation("m2", _ZERO_F)
        expected = _DEFAULT * _DECAY + _ZERO_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("m2", "cooperation") - expected) < _EPS

    def test_two_updates_compound(self) -> None:
        rep = _make_rep()
        rep.update_cooperation("m3", _ONE_F)
        after_first = _DEFAULT * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        rep.update_cooperation("m3", _ONE_F)
        expected = after_first * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("m3", "cooperation") - expected) < _EPS

    def test_independent_models_dont_interfere(self) -> None:
        rep = _make_rep()
        rep.update_cooperation("ma", _ONE_F)
        rep.update_cooperation("mb", _ZERO_F)
        assert rep.get_signal("ma", "cooperation") > rep.get_signal("mb", "cooperation")


class TestUpdateHonesty:
    def test_matching_strings_gives_one(self) -> None:
        rep = _make_rep()
        rep.update_honesty("m1", "cooperate", "cooperate")
        expected = _DEFAULT * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("m1", "honesty") - expected) < _EPS

    def test_mismatching_strings_gives_zero(self) -> None:
        rep = _make_rep()
        rep.update_honesty("m2", "cooperate", "defect")
        expected = _DEFAULT * _DECAY + _ZERO_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("m2", "honesty") - expected) < _EPS

    def test_repeated_matches_raise_honesty(self) -> None:
        rep = _make_rep()
        for _ in range(_FOUR):
            rep.update_honesty("m3", "stag", "stag")
        score = rep.get_signal("m3", "honesty")
        assert score > _DEFAULT

    def test_repeated_mismatches_lower_honesty(self) -> None:
        rep = _make_rep()
        for _ in range(_FOUR):
            rep.update_honesty("m4", "stag", "hare")
        score = rep.get_signal("m4", "honesty")
        assert score < _DEFAULT


class TestUpdateFairness:
    def test_high_fairness_update(self) -> None:
        rep = _make_rep()
        rep.update_fairness("f1", _ONE_F)
        expected = _DEFAULT * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("f1", "fairness") - expected) < _EPS

    def test_low_fairness_update(self) -> None:
        rep = _make_rep()
        rep.update_fairness("f2", _ZERO_F)
        expected = _DEFAULT * _DECAY + _ZERO_F * (_ONE_F - _DECAY)
        assert abs(rep.get_signal("f2", "fairness") - expected) < _EPS


class TestRecordPeerRating:
    def test_trustworthy_pushes_peer_above_default(self) -> None:
        rep = _make_rep()
        rep.record_peer_rating("rater1", "target1", "trustworthy")
        score = rep.get_signal("target1", "peer_ratings")
        expected = _DEFAULT * _DECAY + _ONE_F * (_ONE_F - _DECAY)
        assert abs(score - expected) < _EPS

    def test_untrustworthy_pushes_peer_below_default(self) -> None:
        rep = _make_rep()
        rep.record_peer_rating("rater1", "target2", "untrustworthy")
        score = rep.get_signal("target2", "peer_ratings")
        expected = _DEFAULT * _DECAY + _ZERO_F * (_ONE_F - _DECAY)
        assert abs(score - expected) < _EPS

    def test_neutral_rating_stays_near_default(self) -> None:
        rep = _make_rep()
        rep.record_peer_rating("rater1", "target3", "neutral")
        score = rep.get_signal("target3", "peer_ratings")
        expected = _DEFAULT * _DECAY + _DEFAULT * (_ONE_F - _DECAY)
        assert abs(score - expected) < _EPS

    def test_record_gossip_called_on_store(self) -> None:
        rep = _make_rep()
        rep.record_peer_rating("rater_a", "target_b", "trustworthy")
        rep._store.record_gossip.assert_called_once_with(
            "rater_a", "target_b", "trustworthy",
        )


class TestGetVotingWeight:
    def test_default_model_above_floor(self) -> None:
        rep = _make_rep()
        weight = rep.get_voting_weight("new_model")
        assert weight >= _FLOOR

    def test_floor_applies_when_reputation_low(self) -> None:
        rep = _make_rep()
        for _ in range(_THREE * _THREE + _THREE):
            rep.update_cooperation("bad", _ZERO_F)
            rep.update_honesty("bad", "a", "b")
            rep.update_fairness("bad", _ZERO_F)
            rep.record_peer_rating("r", "bad", "untrustworthy")
        weight = rep.get_voting_weight("bad")
        assert weight >= _FLOOR

    def test_high_reputation_exceeds_floor(self) -> None:
        rep = _make_rep()
        for _ in range(_THREE * _THREE + _THREE):
            rep.update_cooperation("good", _ONE_F)
            rep.update_honesty("good", "x", "x")
            rep.update_fairness("good", _ONE_F)
            rep.record_peer_rating("r", "good", "trustworthy")
        weight = rep.get_voting_weight("good")
        assert weight > _FLOOR


class TestGetSignal:
    def test_unknown_signal_returns_default(self) -> None:
        rep = _make_rep()
        val = rep.get_signal("m", "nonexistent_signal")
        assert abs(val - _DEFAULT) < _EPS

    def test_known_signals_are_retrievable(self) -> None:
        rep = _make_rep()
        rep.update_cooperation("m", _ONE_F)
        rep.update_honesty("m", "a", "a")
        rep.update_fairness("m", _ONE_F)
        rep.record_peer_rating("r", "m", "trustworthy")
        assert rep.get_signal("m", "cooperation") > _DEFAULT
        assert rep.get_signal("m", "honesty") > _DEFAULT
        assert rep.get_signal("m", "fairness") > _DEFAULT
        assert rep.get_signal("m", "peer_ratings") > _DEFAULT
