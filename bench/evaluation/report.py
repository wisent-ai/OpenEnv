"""Report generation for KantBench evaluation results.

Produces both a JSON string and a Markdown string from tournament results
and computed metrics.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from constant_definitions.game_constants import (
    EVAL_FOUR,
    EVAL_HUNDRED,
    EVAL_INDENT_SPACES,
    EVAL_ONE,
    EVAL_TWO,
    EVAL_ZERO,
    EVAL_ZERO_FLOAT,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    tournament_results: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Tuple[str, str]:
    """Create JSON and Markdown reports.

    Parameters
    ----------
    tournament_results : dict
        Nested dict from ``TournamentRunner.run_tournament_as_dict``.
    metrics : dict
        Flat dict from ``compute_metrics``.

    Returns
    -------
    tuple[str, str]
        ``(json_string, markdown_string)``
    """
    json_str = _build_json(tournament_results, metrics)
    md_str = _build_markdown(tournament_results, metrics)
    return json_str, md_str


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------


def _build_json(
    tournament_results: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    """Assemble the structured JSON report."""
    report_data: Dict[str, Any] = {
        "summary": _summary_block(tournament_results, metrics),
        "per_game_results": _per_game_block(tournament_results),
        "strategy_analysis": _strategy_analysis_block(tournament_results),
        "metrics": dict(metrics),
    }
    return json.dumps(report_data, indent=EVAL_INDENT_SPACES, sort_keys=True)


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


def _build_markdown(
    tournament_results: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    """Assemble the Markdown report."""
    sections: List[str] = []
    sections.append(_md_summary(tournament_results, metrics))
    sections.append(_md_per_game(tournament_results))
    sections.append(_md_strategy_analysis(tournament_results))
    sections.append(_md_metrics(metrics))
    separator = "\n\n"
    return separator.join(sections)


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------


def _summary_block(
    tr: Dict[str, Any], met: Dict[str, Any],
) -> Dict[str, Any]:
    total_ep = tr.get("total_episodes", EVAL_ZERO)
    games_list = tr.get("games_played", [])
    strats_list = tr.get("strategies_tested", [])
    return {
        "total_episodes": total_ep,
        "games_count": len(games_list),
        "strategies_count": len(strats_list),
        "games": games_list,
        "strategies": strats_list,
        "strategic_reasoning_score": met.get(
            "strategic_reasoning", EVAL_ZERO_FLOAT,
        ),
    }


def _per_game_block(tr: Dict[str, Any]) -> Dict[str, Any]:
    games = tr.get("games", {})
    block: Dict[str, Any] = {}
    for g_key, strat_map in games.items():
        game_entry: Dict[str, Any] = {}
        for s_key, entry in strat_map.items():
            game_entry[s_key] = {
                "player_score": entry["total_player_score"],
                "opponent_score": entry["total_opponent_score"],
                "cooperation_rate": entry["mean_cooperation_rate"],
                "episode_count": len(entry.get("episodes", [])),
            }
        block[g_key] = game_entry
    return block


def _strategy_analysis_block(tr: Dict[str, Any]) -> Dict[str, Any]:
    """Per-strategy aggregation across all games."""
    games = tr.get("games", {})
    strat_totals: Dict[str, Dict[str, Any]] = {}
    for strat_map in games.values():
        for s_key, entry in strat_map.items():
            if s_key not in strat_totals:
                strat_totals[s_key] = {
                    "total_player_score": EVAL_ZERO_FLOAT,
                    "total_opponent_score": EVAL_ZERO_FLOAT,
                    "cooperation_rates": [],
                    "game_count": EVAL_ZERO,
                }
            bucket = strat_totals[s_key]
            bucket["total_player_score"] += entry["total_player_score"]
            bucket["total_opponent_score"] += entry["total_opponent_score"]
            bucket["cooperation_rates"].append(entry["mean_cooperation_rate"])
            bucket["game_count"] += EVAL_ONE
    analysis: Dict[str, Any] = {}
    for s_key, bucket in strat_totals.items():
        rates = bucket["cooperation_rates"]
        avg_coop = sum(rates) / len(rates) if rates else EVAL_ZERO_FLOAT
        analysis[s_key] = {
            "total_player_score": bucket["total_player_score"],
            "total_opponent_score": bucket["total_opponent_score"],
            "mean_cooperation_rate": avg_coop,
            "games_played": bucket["game_count"],
        }
    return analysis


# ---------------------------------------------------------------------------
# Markdown section renderers
# ---------------------------------------------------------------------------


def _md_summary(tr: Dict[str, Any], met: Dict[str, Any]) -> str:
    games_list = tr.get("games_played", [])
    strats_list = tr.get("strategies_tested", [])
    total_ep = tr.get("total_episodes", EVAL_ZERO)
    score = met.get("strategic_reasoning", EVAL_ZERO_FLOAT)
    lines: List[str] = [
        "# KantBench Evaluation Report",
        "",
        "## Summary",
        "",
        "| Attribute | Value |",
        "|---|---|",
        f"| Games | {len(games_list)} |",
        f"| Strategies | {len(strats_list)} |",
        f"| Total Episodes | {total_ep} |",
        f"| Strategic Reasoning Score | {_pct(score)} |",
    ]
    return "\n".join(lines)


def _md_per_game(tr: Dict[str, Any]) -> str:
    games = tr.get("games", {})
    lines: List[str] = ["## Per-Game Results"]
    for g_key, strat_map in games.items():
        lines.append("")
        lines.append(f"### {g_key}")
        lines.append("")
        lines.append(
            "| Strategy | Player Score | Opponent Score | Coop Rate |"
        )
        lines.append("|---|---|---|---|")
        for s_key, entry in strat_map.items():
            p = entry["total_player_score"]
            o = entry["total_opponent_score"]
            c = entry["mean_cooperation_rate"]
            lines.append(f"| {s_key} | {_fmt(p)} | {_fmt(o)} | {_pct(c)} |")
    return "\n".join(lines)


def _md_strategy_analysis(tr: Dict[str, Any]) -> str:
    analysis = _strategy_analysis_block(tr)
    lines: List[str] = [
        "## Strategy Analysis",
        "",
        "| Strategy | Total Player | Total Opponent | Avg Coop | Games |",
        "|---|---|---|---|---|",
    ]
    for s_key, data in analysis.items():
        p = data["total_player_score"]
        o = data["total_opponent_score"]
        c = data["mean_cooperation_rate"]
        g = data["games_played"]
        lines.append(
            f"| {s_key} | {_fmt(p)} | {_fmt(o)} | {_pct(c)} | {g} |"
        )
    return "\n".join(lines)


def _md_metrics(met: Dict[str, Any]) -> str:
    lines: List[str] = [
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    display_order = [
        "cooperation_rate",
        "exploitation_resistance",
        "pareto_efficiency",
        "fairness_index",
        "adaptability",
        "strategic_reasoning",
    ]
    for key in display_order:
        if key in met:
            lines.append(f"| {_label(key)} | {_pct(met[key])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_ROUND_DIGITS = EVAL_TWO


def _fmt(value: float) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{_ROUND_DIGITS}f}"


def _pct(value: float) -> str:
    """Format a fraction as a percentage string."""
    scaled = value * EVAL_HUNDRED
    return f"{scaled:.{_ROUND_DIGITS}f}%"


def _label(key: str) -> str:
    """Convert a snake_case metric key into a human-readable label."""
    return key.replace("_", " ").title()
