"""Smoke test for bench/gradio_app/md/builders.py and md/tournament.py."""

from __future__ import annotations

import os
import sys
import types

# Stub openenv so importing the env layer transitively does not fail.
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

from md.builders import _build_matrix_md, _build_all_matrices_md  # noqa: E402
from md.tournament import run_metrics_tournament  # noqa: E402


def test_build_matrix_md_renders_pd_payoffs():
    """Prisoner's Dilemma matrix renders with the cooperate/defect labels and payoffs."""
    rendered = _build_matrix_md("Prisoner's Dilemma", None)
    assert "**cooperate**" in rendered
    assert "**defect**" in rendered
    assert "3, 3" in rendered  # CC payoff
    assert "5, 0" in rendered or "0, 5" in rendered  # asymmetric payoff


def test_build_matrix_md_unknown_game():
    """Looking up a game that does not exist returns a placeholder, not an error."""
    rendered = _build_matrix_md("Definitely Not A Game", None)
    assert "Game not found" in rendered


def test_build_all_matrices_md_includes_pd_section():
    """The aggregate matrix view prints a section per two-player game."""
    rendered = _build_all_matrices_md()
    assert "Prisoner's Dilemma" in rendered
    assert "Stag Hunt" in rendered
    assert "Hawk-Dove" in rendered


def test_run_metrics_tournament_no_games_selected():
    """Empty game selection returns a placeholder, not an error."""
    rendered = run_metrics_tournament("tit_for_tat", 1, [])
    assert "Select at least one game" in rendered


def test_run_metrics_tournament_reports_self_payoff():
    """The headline column in the rendered table is mean self-payoff."""
    rendered = run_metrics_tournament(
        "tit_for_tat", 1, ["Prisoner's Dilemma"],
    )
    assert "Tournament Results" in rendered
    assert "Mean self-payoff" in rendered
    assert "Prisoner's Dilemma" in rendered


def test_run_metrics_tournament_always_defect_outscores_always_cooperate_in_pd():
    """In PD, always_defect's mean payoff strictly exceeds always_cooperate's
    when the opponent pool is the full strategy library minus self."""
    rendered_def = run_metrics_tournament(
        "always_defect", 1, ["Prisoner's Dilemma"],
    )
    rendered_coop = run_metrics_tournament(
        "always_cooperate", 1, ["Prisoner's Dilemma"],
    )
    # Pull the score column out of the table line:
    # "| Prisoner's Dilemma | <self_payoff> | <coop_rate> |"
    def _extract_payoff(rendered: str) -> float:
        for line in rendered.split("\n"):
            if line.startswith("| Prisoner's Dilemma "):
                cells = [c.strip() for c in line.split("|")]
                # cells = ['', "Prisoner's Dilemma", '<payoff>', '<coop>', '']
                return float(cells[2])
        raise AssertionError(f"no PD row found in:\n{rendered}")

    assert _extract_payoff(rendered_def) > _extract_payoff(rendered_coop)
