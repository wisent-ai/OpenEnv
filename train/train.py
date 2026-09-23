"""KantBench GRPO training over immutable, machine-staged artifacts.

The trusted Stado agent materializes the model and training dataset before this
workload starts. This module accepts only local paths and keeps all Hub clients
offline; publishing and provider credential handling are outside the workload.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.games import GAMES
from train.rewards_to_trajectory.splits import get_train_eval_split

logger = logging.getLogger(__name__)


import sys as _size_split_sys
from pathlib import Path as _SizeSplitPath
_size_split_bytecode = _size_split_sys.dont_write_bytecode
_size_split_sys.dont_write_bytecode = True
if str(_SizeSplitPath(__file__).resolve().parent) not in _size_split_sys.path:
    _size_split_sys.path.insert(0, str(_SizeSplitPath(__file__).resolve().parent))
from train_parts.system_prompt import SYSTEM_PROMPT, _FORBIDDEN_WORKLOAD_CREDENTIALS, require_stado_workload, require_local_artifact, load_staged_dataset, REWARD_STRATEGIES, _build_local_prompt, _local_coop_rate, format_reward_fn  # noqa: F401
from train_parts.parse_args import _batch_generate_actions, _play_batch_interactive_episodes, parse_args  # noqa: F401
from train_parts.make_reward_fn import make_reward_fn  # noqa: F401
from train_parts.main import main  # noqa: F401
_size_split_sys.dont_write_bytecode = _size_split_bytecode
del _size_split_sys, _SizeSplitPath, _size_split_bytecode


if __name__ == "__main__":
    main()
