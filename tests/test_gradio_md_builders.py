"""Smoke test for bench/gradio_app/md/builders.py and registry.format_nash."""

from __future__ import annotations

import os
import sys
import types

# Same openenv stub as test_nash so importing the env layer doesn't fail.
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

# bench/gradio_app uses sibling-relative imports (`from registry import ...`),
# so we have to add that directory to sys.path explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GRADIO_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "bench", "gradio_app"),
)
if _GRADIO_DIR not in sys.path:
    sys.path.insert(0, _GRADIO_DIR)

from md.builders import _build_matrix_md  # noqa: E402
from registry import format_nash  # noqa: E402


def test_format_nash_empty():
    """No declared equilibria yields an empty string, not noise."""
    assert format_nash(()) == ""


def test_format_nash_single_pure_equilibrium():
    """A pure-strategy equilibrium prints as a single bullet with one P(action)=1 entry."""
    rendered = format_nash(({"defect": 1.0, "cooperate": 0.0},))
    assert "**Nash equilibria:**" in rendered
    assert "P(defect)=1" in rendered
    assert "P(cooperate)=0" not in rendered  # zero-prob actions are dropped


def test_format_nash_multiple_equilibria_each_get_a_bullet():
    """Stag Hunt has three NE; each renders as its own bullet."""
    rendered = format_nash((
        {"stag": 1.0, "hare": 0.0},
        {"stag": 0.0, "hare": 1.0},
        {"stag": 2.0 / 3.0, "hare": 1.0 / 3.0},
    ))
    bullets = [line for line in rendered.split("\n") if line.startswith("- ")]
    assert len(bullets) == 3


def test_build_matrix_md_appends_nash_for_pd():
    """_build_matrix_md output for PD includes the {defect: 1.0} equilibrium."""
    rendered = _build_matrix_md("Prisoner's Dilemma", None)
    assert "**Nash equilibria:**" in rendered
    assert "P(defect)=1" in rendered


def test_build_matrix_md_appends_nash_for_hawk_dove():
    """Hawk-Dove output includes the symmetric mixed (1/3 hawk, 2/3 dove) equilibrium."""
    rendered = _build_matrix_md("Hawk-Dove", None)
    assert "**Nash equilibria:**" in rendered
    # 1/3 prints as 0.333333 with %g formatting; just check substring.
    assert "P(hawk)=0.33" in rendered
    assert "P(dove)=0.66" in rendered


def test_build_matrix_md_appends_nash_for_ultimatum():
    """Ultimatum output includes the {offer_0: 1.0} subgame-perfect equilibrium."""
    rendered = _build_matrix_md("Ultimatum Game", None)
    assert "**Nash equilibria:**" in rendered
    assert "P(offer_0)=1" in rendered
