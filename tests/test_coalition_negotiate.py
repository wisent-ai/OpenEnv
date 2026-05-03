"""Unit tests for scripts/diag/_coalition_negotiate.py parser functions."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass

# openenv stub so importing env layer works without the upstream package.
if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)
        def __class_getitem__(cls, params): return cls
        def __init__(self): pass

    _iface_stub.Environment = _EnvironmentStub
    _openenv_stub.core = _core_stub
    _core_stub.env_server = _server_stub
    _server_stub.interfaces = _iface_stub
    for _n, _m in [
        ("openenv", _openenv_stub),
        ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

# scripts/diag uses sibling-relative imports (`from registry import ...`),
# but _coalition_negotiate.py only depends on env.nplayer.coalition.models
# and constant_definitions, both at top-level. So adding scripts/diag to
# sys.path is sufficient.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIAG_DIR = os.path.normpath(os.path.join(_HERE, "..", "scripts", "diag"))
if _DIAG_DIR not in sys.path:
    sys.path.insert(0, _DIAG_DIR)

from _coalition_negotiate import (  # noqa: E402  type: ignore[import-not-found]
    _parse_acceptance_indices,
    _parse_proposal,
    _SIDE_PAYMENT_MAX,
)


@dataclass
class _StubBase:
    player_index: int = 0
    num_players: int = 5
    available_actions: tuple = ("cooperate", "defect")


@dataclass
class _StubObs:
    base: _StubBase
    pending_proposals: tuple = ()


def _obs(player_index: int = 0, num_players: int = 5,
         actions=("cooperate", "defect")):
    return _StubObs(base=_StubBase(
        player_index=player_index, num_players=num_players,
        available_actions=actions,
    ))


# -------------------- _parse_acceptance_indices --------------------

def test_acceptance_none_means_no_indices():
    assert _parse_acceptance_indices("none", num_proposals=3) == []


def test_acceptance_empty_completion():
    assert _parse_acceptance_indices("", num_proposals=3) == []


def test_acceptance_single_index():
    assert _parse_acceptance_indices("0", num_proposals=3) == [0]


def test_acceptance_comma_separated():
    assert _parse_acceptance_indices("0,2", num_proposals=3) == [0, 2]


def test_acceptance_dedup_and_sort():
    assert _parse_acceptance_indices("2,0,2", num_proposals=3) == [0, 2]


def test_acceptance_out_of_range_dropped():
    # num_proposals=2 means valid indices are {0, 1}.
    assert _parse_acceptance_indices("0, 1, 5, 9", num_proposals=2) == [0, 1]


def test_acceptance_natural_language_with_indices():
    assert _parse_acceptance_indices(
        "I accept 0 and 2 but reject 1", num_proposals=3,
    ) == [0, 1, 2]


# -------------------- _parse_proposal --------------------

def test_proposal_two_player():
    p = _parse_proposal("P2 cooperate pay 1", _obs())
    assert p is not None
    assert p.proposer == 0
    assert p.members == [0, 2]
    assert p.agreed_action == "cooperate"
    assert abs(p.side_payment - 1.0) < 1e-9


def test_proposal_three_player():
    p = _parse_proposal("P2 P3 cooperate pay 1", _obs())
    assert p is not None
    assert p.members == [0, 2, 3]


def test_proposal_dedup_targets():
    p = _parse_proposal("P2 P2 P3 cooperate pay 0.5", _obs())
    assert p is not None
    assert p.members == [0, 2, 3]


def test_proposal_self_target_rejected():
    # If P0 (self) is the only target, the proposal is invalid.
    assert _parse_proposal("P0 cooperate pay 1", _obs()) is None


def test_proposal_self_target_dropped_keeps_others():
    p = _parse_proposal("P0 P2 cooperate pay 1", _obs())
    assert p is not None
    assert p.members == [0, 2]  # self dropped from targets


def test_proposal_out_of_range_target_dropped():
    p = _parse_proposal("P9 P2 cooperate pay 1", _obs(num_players=5))
    assert p is not None
    assert p.members == [0, 2]  # P9 dropped


def test_proposal_no_targets_returns_none():
    assert _parse_proposal("P9 cooperate pay 1", _obs(num_players=5)) is None


def test_proposal_no_action_returns_none():
    assert _parse_proposal("P2 pay 1", _obs()) is None


def test_proposal_none_completion():
    assert _parse_proposal("none", _obs()) is None


def test_proposal_default_payment_when_missing():
    p = _parse_proposal("P2 cooperate", _obs())
    assert p is not None
    assert p.side_payment == 0.0


def test_proposal_payment_clamped_to_max():
    huge = _SIDE_PAYMENT_MAX + 100
    p = _parse_proposal(f"P2 cooperate pay {huge}", _obs())
    assert p is not None
    assert p.side_payment == _SIDE_PAYMENT_MAX


def test_proposal_payment_clamped_to_zero():
    p = _parse_proposal("P2 cooperate pay 0", _obs())
    assert p is not None
    assert p.side_payment == 0.0
