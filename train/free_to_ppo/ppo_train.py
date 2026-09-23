"""KantBench REINFORCE training over immutable, machine-staged artifacts.

The policy, frozen reference, and tokenizer are loaded offline from the model
directory materialized by Stado. Prompt records come from the staged dataset;
this workload never acquires or publishes provider artifacts.
"""
from __future__ import annotations

import logging
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from env.environment import KantEnvironment
from train.agent import parse_action
from train.rewards_to_trajectory.splits import get_train_eval_split
from train.train import (
    SYSTEM_PROMPT,
    REWARD_STRATEGIES,
    _play_batch_interactive_episodes,
    load_staged_dataset,
    require_local_artifact,
    require_stado_workload,
)

logger = logging.getLogger(__name__)


import sys as _size_split_sys
from pathlib import Path as _SizeSplitPath
_size_split_bytecode = _size_split_sys.dont_write_bytecode
_size_split_sys.dont_write_bytecode = True
if str(_SizeSplitPath(__file__).resolve().parents[1]) not in _size_split_sys.path:
    _size_split_sys.path.insert(0, str(_SizeSplitPath(__file__).resolve().parents[1]))
from ppo_train_parts.compute_reward import compute_reward, compute_log_probs, compute_kl, parse_args  # noqa: F401
from ppo_train_parts.main import main  # noqa: F401
_size_split_sys.dont_write_bytecode = _size_split_bytecode
del _size_split_sys, _SizeSplitPath, _size_split_bytecode


if __name__ == "__main__":
    main()
