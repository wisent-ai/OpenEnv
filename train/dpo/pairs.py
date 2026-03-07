"""Preference pair generation for DPO training."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from train.trajectory import EpisodeTrajectory
from constant_definitions.game_constants import EVAL_ONE, EVAL_ZERO
from constant_definitions.train.dpo_constants import (
    DPO_BOTTOM_QUANTILE_DENOMINATOR,
    DPO_BOTTOM_QUANTILE_NUMERATOR,
    DPO_MIN_REWARD_MARGIN_DENOMINATOR,
    DPO_MIN_REWARD_MARGIN_NUMERATOR,
    DPO_TOP_QUANTILE_DENOMINATOR,
    DPO_TOP_QUANTILE_NUMERATOR,
)

_ONE = int(bool(True))


def generate_preference_pairs(
    trajectories: List[EpisodeTrajectory],
    min_margin_numerator: int = DPO_MIN_REWARD_MARGIN_NUMERATOR,
    min_margin_denominator: int = DPO_MIN_REWARD_MARGIN_DENOMINATOR,
) -> List[Dict[str, Any]]:
    """Generate chosen/rejected preference pairs from trajectories.

    Groups trajectories by (game, strategy), ranks by episode_reward,
    pairs top-quartile (chosen) vs bottom-quartile (rejected), and
    filters by minimum reward margin.

    Returns list of dicts with keys: prompt, chosen, rejected, margin.
    """
    min_margin = min_margin_numerator / min_margin_denominator

    # Group by (game, strategy)
    groups: Dict[Tuple[str, str], List[EpisodeTrajectory]] = {}
    for traj in trajectories:
        key = (traj.game, traj.strategy)
        if key not in groups:
            groups[key] = []
        groups[key].append(traj)

    pairs: List[Dict[str, Any]] = []
    for _key, group in groups.items():
        group_pairs = _pairs_from_group(group, min_margin)
        pairs.extend(group_pairs)

    return pairs


def _pairs_from_group(
    group: List[EpisodeTrajectory],
    min_margin: float,
) -> List[Dict[str, Any]]:
    """Generate pairs from a single (game, strategy) group."""
    if len(group) < EVAL_ONE + EVAL_ONE:
        return []

    # Sort by episode reward descending
    ranked = sorted(group, key=lambda t: t.episode_reward, reverse=True)
    n = len(ranked)

    # Top and bottom quartile boundaries
    top_boundary = max(
        _ONE,
        (n * DPO_TOP_QUANTILE_NUMERATOR) // DPO_TOP_QUANTILE_DENOMINATOR,
    )
    bottom_boundary = max(
        _ONE,
        (n * DPO_BOTTOM_QUANTILE_NUMERATOR) // DPO_BOTTOM_QUANTILE_DENOMINATOR,
    )

    chosen_set = ranked[:top_boundary]
    rejected_set = ranked[n - bottom_boundary:]

    pairs: List[Dict[str, Any]] = []
    for chosen in chosen_set:
        for rejected in rejected_set:
            margin = chosen.episode_reward - rejected.episode_reward
            if margin < min_margin:
                continue
            # Use the full episode as prompt + chosen/rejected completions
            chosen_text = _trajectory_to_text(chosen)
            rejected_text = _trajectory_to_text(rejected)
            prompt = _trajectory_prompt(chosen)
            pairs.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "margin": margin,
                "game": chosen.game,
                "strategy": chosen.strategy,
            })

    return pairs


def _trajectory_to_text(traj: EpisodeTrajectory) -> str:
    """Convert trajectory actions to a single completion string."""
    return "\n".join(step.completion for step in traj.steps)


def _trajectory_prompt(traj: EpisodeTrajectory) -> str:
    """Extract the first step's prompt as the shared prompt."""
    if traj.steps:
        return traj.steps[EVAL_ZERO].prompt
    return ""
