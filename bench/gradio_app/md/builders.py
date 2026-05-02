"""Markdown rendering for gradio's Payoff Matrices and Game Theory Reference tabs.

Pure string assembly. Reads from the registry's flat dicts (``_GAME_INFO``,
``_KEY_TO_NAME``, etc.) and from the per-game ``nash_equilibria`` field
populated in ``registry.py``. Extracted from ``callbacks.py`` to keep that
module under the per-file line cap once Nash-equilibrium display landed.
"""

from __future__ import annotations

from registry import (
    _TWO, _FOUR,
    _HAS_REGISTRY, _HAS_VARIANTS, _HAS_NPLAYER_ENV,
    _GAME_INFO, _KEY_TO_NAME, _CATEGORY_DIMS,
    compose_game, get_games_by_tag,
    _NPLAYER_STRAT_NAMES, _HUMAN_VARIANTS,
    _GENERIC_STRATEGIES, _GAME_TYPE_STRATEGIES,
    _LLM_OPPONENT_LABEL,
    format_nash,
)


def _info_for(gname, variants):
    base = _GAME_INFO.get(gname)
    if not base or not variants or not _HAS_VARIANTS:
        return base
    try:
        cfg = compose_game(base["key"], *variants)
        return {
            "actions": cfg.actions, "description": cfg.description,
            "payoff_fn": cfg.payoff_fn, "default_rounds": cfg.default_rounds,
            "key": base["key"], "num_players": cfg.num_players,
            "game_type": cfg.game_type, "opponent_actions": cfg.opponent_actions,
            "nash_equilibria": (),
        }
    except (KeyError, ValueError):
        return base


def _build_matrix_md(game_name, variant_list):
    """Payoff-matrix markdown table for *game_name*, plus Nash equilibria when declared."""
    info = _info_for(game_name, variant_list if variant_list else None)
    if not info:
        return "Game not found."
    actions = info["actions"]
    opp_actions = list(info["opponent_actions"]) if info.get("opponent_actions") else actions
    MAX_ACTIONS = 8
    row_acts, col_acts = actions[:MAX_ACTIONS], opp_actions[:MAX_ACTIONS]
    row_trunc, col_trunc = len(actions) > MAX_ACTIONS, len(opp_actions) > MAX_ACTIONS
    header_inner = " | ".join(f"**{a}**" for a in col_acts) + (" | ..." if col_trunc else "")
    rows = [
        f"| P1 \\ P2 | {header_inner} |",
        "|" + "|".join(["---"] * (len(col_acts) + 1 + (1 if col_trunc else 0))) + "|",
    ]
    for ra in row_acts:
        cells = [f"**{ra}**"]
        for ca in col_acts:
            try:
                p, o = info["payoff_fn"](ra, ca)
                cells.append(f"{p:g}, {o:g}")
            except Exception:
                cells.append("—")
        if col_trunc:
            cells.append("…")
        rows.append("| " + " | ".join(cells) + " |")
    if row_trunc:
        rows.append("| ... |" + " |" * (len(col_acts) + (1 if col_trunc else 0)))
    note = (
        f"\n\n*Showing {len(row_acts)}×{len(col_acts)} of "
        f"{len(actions)}×{len(opp_actions)} actions.*"
    ) if (row_trunc or col_trunc) else ""
    return "\n".join(rows) + note + format_nash(info.get("nash_equilibria", ()))


def _build_all_matrices_md():
    """Payoff matrices for every two-player base game, with Nash equilibria when declared."""
    if not _HAS_REGISTRY:
        return "# Payoff Matrices\n\nFull registry not available."
    sections = ["# Payoff Matrices\n"]
    for gname in sorted(_GAME_INFO.keys()):
        info = _GAME_INFO[gname]
        if info.get("num_players", _TWO) > _TWO:
            continue
        sections.append(f"## {gname}\n")
        sections.append(f"*{info['description']}*\n")
        sections.append(_build_matrix_md(gname, None))
        sections.append("")
    return "\n\n".join(sections)


def _build_reference_md():
    if not _HAS_REGISTRY:
        return "# Game Theory Reference\n\nFull registry not available."
    sections = []
    for dim_name, tags in sorted(_CATEGORY_DIMS.items()):
        sec = [f"## {dim_name.replace('_', ' ').title()}"]
        for tag in tags:
            names = sorted(
                _KEY_TO_NAME[k] for k in get_games_by_tag(tag) if k in _KEY_TO_NAME
            )
            if names:
                sec.append(f"**{tag}** ({len(names)}): {', '.join(names)}")
        sections.append("\n\n".join(sec))
    np_games = [
        (gn, gi) for gn, gi in _GAME_INFO.items()
        if gi.get("num_players", _TWO) > _TWO
    ]
    if np_games:
        np_lines = [
            "## Multiplayer Games", "| Game | Players | Actions | Rounds |",
            "|------|---------|---------|--------|",
        ]
        for gn, gi in sorted(np_games):
            acts = gi["actions"]
            act_str = ", ".join(acts[:_FOUR]) + (
                f" ... ({len(acts)} total)" if len(acts) > _FOUR else ""
            )
            np_lines.append(
                f"| {gn} | {gi['num_players']} | {act_str} | {gi['default_rounds']} |"
            )
        sections.append("\n".join(np_lines))
    if _HUMAN_VARIANTS:
        sections.append(
            "## Composable Variants\n"
            + "\n".join(f"- **{v}**" for v in _HUMAN_VARIANTS)
        )
    slines = [
        "## Opponent Strategies",
        f"**Generic** ({len(_GENERIC_STRATEGIES)}): {', '.join(_GENERIC_STRATEGIES)}",
    ]
    for gt, strats in sorted(_GAME_TYPE_STRATEGIES.items()):
        slines.append(f"**{gt}**: {', '.join(strats)}")
    if _HAS_NPLAYER_ENV:
        slines.append(f"**N-player**: {', '.join(_NPLAYER_STRAT_NAMES)}")
    slines.append(
        f"\n**LLM Opponents**: Select '{_LLM_OPPONENT_LABEL}' as strategy "
        "and play against Claude or GPT using built-in OAuth tokens."
    )
    sections.append("\n\n".join(slines))
    nash_section = _build_nash_reference()
    if nash_section:
        sections.append(nash_section)
    total, np_count = len(_GAME_INFO), len(np_games)
    return (
        f"# Game Theory Reference\n\n**{total} games** "
        f"({total - np_count} two-player, {np_count} multiplayer)\n\n"
        + "\n\n---\n\n".join(sections)
    )


def _build_nash_reference():
    """One-line summary of declared Nash equilibria, sorted by game name."""
    annotated = sorted(
        (gname, gi.get("nash_equilibria", ())) for gname, gi in _GAME_INFO.items()
    )
    rows = []
    for gname, eqs in annotated:
        if not eqs:
            continue
        eq_strs = []
        for eq in eqs:
            parts = ", ".join(f"P({a})={p:g}" for a, p in eq.items() if p > 0)
            eq_strs.append(parts)
        rows.append(f"- **{gname}**: " + "  ;  ".join(eq_strs))
    if not rows:
        return ""
    return "## Nash Equilibria (analytical)\n" + "\n".join(rows)
