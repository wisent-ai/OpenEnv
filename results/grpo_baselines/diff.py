"""Diff a pre/post pair of GRPO baseline JSONs.

Reads results/grpo_baselines/<name>_pre.json and <name>_post.json,
prints a markdown comparison: per-game aggregate, per-(game, opponent),
and the headline diagnostic (vs always_defect rows toward Nash floor).

Run:

    python3 results/grpo_baselines/diff.py llama_3_2_1b
"""

from __future__ import annotations

import argparse
import json
import os


def _load(name: str, suffix: str):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"{name}_{suffix}.json")
    if not os.path.isfile(path):
        return None, path
    with open(path) as fh:
        return json.load(fh), path


def _delta(pre: float, post: float) -> str:
    d = post - pre
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _table_per_game(pre, post):
    rows = ["| Game | pre | post | Δ |", "|------|-----|------|---|"]
    pre_g = pre["mean_self_payoff_per_game"]
    post_g = post["mean_self_payoff_per_game"]
    games = sorted(set(pre_g) | set(post_g))
    for g in games:
        a = pre_g.get(g, float("nan"))
        b = post_g.get(g, float("nan"))
        rows.append(f"| {g} | {a:.3f} | {b:.3f} | {_delta(a, b)} |")
    return "\n".join(rows)


def _table_per_opp(pre, post):
    rows = ["| Game | Opponent | pre | post | Δ |",
            "|------|----------|-----|------|---|"]
    pre_p = pre.get("per_game_per_opponent", {})
    post_p = post.get("per_game_per_opponent", {})
    games = sorted(set(pre_p) | set(post_p))
    for g in games:
        opps = sorted(
            set(pre_p.get(g, {})) | set(post_p.get(g, {}))
        )
        for o in opps:
            a = pre_p.get(g, {}).get(o, float("nan"))
            b = post_p.get(g, {}).get(o, float("nan"))
            rows.append(f"| {g} | {o} | {a:.3f} | {b:.3f} | {_delta(a, b)} |")
    return "\n".join(rows)


def _diagnostic_summary(pre, post) -> str:
    """The headline: how far did the *_vs_always_defect rows move?"""
    pre_p = pre.get("per_game_per_opponent", {})
    post_p = post.get("per_game_per_opponent", {})
    lines = ["**Diagnostic — vs always_defect (the Nash-floor rows):**"]
    for g in sorted(set(pre_p) | set(post_p)):
        a = pre_p.get(g, {}).get("always_defect")
        b = post_p.get(g, {}).get("always_defect")
        if a is None or b is None:
            continue
        lines.append(f"- **{g}**: {a:.3f} → {b:.3f} ({_delta(a, b)})")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name", help="Filename stem before _pre / _post (e.g. llama_3_2_1b).")
    args = p.parse_args()

    pre, pre_path = _load(args.name, "pre")
    post, post_path = _load(args.name, "post")
    if pre is None:
        raise SystemExit(f"missing pre file: {pre_path}")
    if post is None:
        raise SystemExit(
            f"missing post file: {post_path} -- training has not finished. "
            "Wait for the wisent-compute job to write a *_post.json here."
        )

    print(f"# GRPO pre/post diff: {args.name}\n")
    print(f"Pre:  `{pre_path}`")
    print(f"Post: `{post_path}`\n")
    print("## Per-game aggregate (mean self-payoff per round)\n")
    print(_table_per_game(pre, post))
    print("\n## Per-(game, opponent)\n")
    print(_table_per_opp(pre, post))
    print()
    print(_diagnostic_summary(pre, post))


if __name__ == "__main__":
    main()
