"""Convert episode trajectories to HuggingFace Dataset format for GRPO."""

from __future__ import annotations

from typing import Any, Dict, List

from train.trajectory import EpisodeTrajectory, StepRecord
from constant_definitions.game_constants import EVAL_ONE, EVAL_ZERO_FLOAT
from constant_definitions.train.grpo_constants import (
    GRPO_SHAPING_ALPHA_DENOMINATOR,
    GRPO_SHAPING_ALPHA_NUMERATOR,
)

_ONE = int(bool(True))


def trajectories_to_dataset(
    trajectories: List[EpisodeTrajectory],
) -> List[Dict[str, Any]]:
    """Convert trajectories into per-round records for GRPO training.

    Each round becomes a separate training example with:
    - ``prompt``: the structured game prompt for that round
    - ``completion``: the model's action text
    - ``reward``: episode reward for the final round, shaping reward otherwise

    This keeps completions short (one action per round) rather than
    generating entire multi-round episodes as single completions.
    """
    records: List[Dict[str, Any]] = []
    for traj in trajectories:
        num_steps = len(traj.steps)
        if num_steps == EVAL_ONE - EVAL_ONE:
            continue
        last_idx = num_steps - _ONE
        for idx, step in enumerate(traj.steps):
            if idx == last_idx:
                reward = traj.episode_reward
            else:
                reward = step.reward
            records.append({
                "prompt": step.prompt,
                "completion": step.completion,
                "reward": reward,
                "game": traj.game,
                "strategy": traj.strategy,
                "round_number": step.round_number,
                "is_terminal": idx == last_idx,
            })
    return records


def records_to_hf_dict(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """Convert list-of-dicts to dict-of-lists for HF Dataset.from_dict()."""
    if not records:
        return {
            "prompt": [],
            "completion": [],
            "reward": [],
            "game": [],
            "strategy": [],
            "round_number": [],
            "is_terminal": [],
        }
    keys = list(records[EVAL_ONE - EVAL_ONE].keys())
    return {k: [r[k] for r in records] for k in keys}
