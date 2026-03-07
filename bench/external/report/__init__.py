"""Report generation for external benchmark evaluation results.

Produces both a JSON string and a Markdown string from a mapping of
benchmark names to ``BenchmarkResult`` instances.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from bench.external._base import BenchmarkResult
from bench.external.constants import (
    REPORT_HUNDRED,
    REPORT_INDENT_SPACES,
    REPORT_ROUND_DIGITS,
)


def generate_external_report(
    results: Dict[str, BenchmarkResult],
    model_name: str,
) -> Tuple[str, str]:
    """Create JSON and Markdown reports for external benchmarks.

    Parameters
    ----------
    results : dict
        Mapping of benchmark name to ``BenchmarkResult``.
    model_name : str
        Model identifier for the report header.

    Returns
    -------
    tuple[str, str]
        ``(json_string, markdown_string)``
    """
    json_str = _build_json(results, model_name)
    md_str = _build_markdown(results, model_name)
    return json_str, md_str


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------


def _build_json(
    results: Dict[str, BenchmarkResult],
    model_name: str,
) -> str:
    report: Dict[str, Any] = {
        "model": model_name,
        "summary": _summary_block(results),
        "benchmarks": _benchmarks_block(results),
    }
    return json.dumps(
        report, indent=REPORT_INDENT_SPACES, sort_keys=True,
    )


def _summary_block(
    results: Dict[str, BenchmarkResult],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, result in results.items():
        entry: Dict[str, Any] = {"primary_metric": result.primary_metric}
        if result.error is not None:
            entry["error"] = result.error
        else:
            entry["primary_score"] = result.primary_score
        entry["elapsed_seconds"] = round(
            result.elapsed_seconds, REPORT_ROUND_DIGITS,
        )
        summary[name] = entry
    return summary


def _benchmarks_block(
    results: Dict[str, BenchmarkResult],
) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    for name, result in results.items():
        entry: Dict[str, Any] = {
            "scores": result.scores,
            "metadata": result.metadata,
        }
        if result.error is not None:
            entry["error"] = result.error
        block[name] = entry
    return block


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


def _build_markdown(
    results: Dict[str, BenchmarkResult],
    model_name: str,
) -> str:
    sections: List[str] = []
    sections.append(_md_header(model_name))
    sections.append(_md_summary_table(results))
    sections.append(_md_details(results))
    separator = "\n\n"
    return separator.join(sections)


def _md_header(model_name: str) -> str:
    return f"# External Benchmark Report: {model_name}"


def _md_summary_table(results: Dict[str, BenchmarkResult]) -> str:
    lines: List[str] = [
        "## Summary",
        "",
        "| Benchmark | Primary Metric | Score | Time (s) |",
        "|---|---|---|---|",
    ]
    for name, result in results.items():
        metric = result.primary_metric
        if result.error is not None:
            score_str = "ERROR"
        else:
            score_str = _pct(result.primary_score) if result.primary_score is not None else "N/A"
        elapsed = _fmt(result.elapsed_seconds)
        lines.append(f"| {name} | {metric} | {score_str} | {elapsed} |")
    return "\n".join(lines)


def _md_details(results: Dict[str, BenchmarkResult]) -> str:
    lines: List[str] = ["## Details"]
    for name, result in results.items():
        lines.append("")
        lines.append(f"### {result.display_name if hasattr(result, 'display_name') else name}")
        if result.error is not None:
            lines.append(f"\nError: {result.error}")
            continue
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for metric_name, value in result.scores.items():
            lines.append(f"| {_label(metric_name)} | {_pct(value)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.{REPORT_ROUND_DIGITS}f}"


def _pct(value: float) -> str:
    scaled = value * REPORT_HUNDRED
    return f"{scaled:.{REPORT_ROUND_DIGITS}f}%"


def _label(key: str) -> str:
    return key.replace("_", " ").title()
