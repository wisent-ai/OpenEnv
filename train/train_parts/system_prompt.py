"""Parts of train.py, split by the tama size splitter; train.py imports every name back."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from datasets import Dataset
from common.games import GAMES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are playing a game-theory game. Analyse the situation and choose "
    "the best action. Respond with ONLY the action name, nothing else."
)



_FORBIDDEN_WORKLOAD_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_CLIENT_SECRET",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "STADO_API_TOKEN",
    "SUPABASE_SERVICE_ROLE_KEY",
    "WANDB_API_KEY",
)


def require_stado_workload() -> None:
    """Fail closed unless Stado owns the workload and no provider secret leaked."""
    if not os.environ.get("WC_JOB_ID", "").strip():
        raise RuntimeError("training must run as a Stado machine workload")
    leaked = [
        name for name in _FORBIDDEN_WORKLOAD_CREDENTIALS
        if os.environ.get(name, "").strip()
    ]
    if leaked:
        raise RuntimeError(
            "provider credentials are forbidden in training workloads: "
            + ", ".join(leaked)
        )
    os.environ["HF_HUB_OFFLINE"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "true"
    os.environ["HF_DATASETS_OFFLINE"] = "true"


# ---------------------------------------------------------------------------
# Immutable staged artifacts
# ---------------------------------------------------------------------------


def require_local_artifact(
    value: str,
    label: str,
    *,
    directory: bool,
) -> Path:
    """Resolve one machine-staged artifact without accepting remote locators."""
    if not value or "://" in value:
        raise ValueError(f"{label} must be a machine-staged local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute machine-staged path")
    path = path.resolve(strict=True)
    valid_kind = path.is_dir() if directory else path.is_file()
    if not valid_kind:
        expected = "directory" if directory else "file"
        raise ValueError(f"{label} must be a local {expected}: {path}")
    return path


def load_staged_dataset(
    value: str,
    n_samples: int,
    *,
    games: list[str] | None = None,
) -> Dataset:
    """Load deterministic prompt records from a staged JSON or JSONL object."""
    path = require_local_artifact(value, "training dataset", directory=False)
    if n_samples <= int("0"):
        raise ValueError("training sample count must be positive")

    with path.open("r", encoding="utf-8") as stream:
        first = stream.read(int("1"))
        stream.seek(int("0"))
        if first == "[":
            raw_records = json.load(stream)
        else:
            raw_records = [
                json.loads(line)
                for line in stream
                if line.strip()
            ]
    if not isinstance(raw_records, list):
        raise ValueError("training dataset must contain a JSON array or JSONL records")

    allowed_games = set(games) if games is not None else None
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"training dataset record {index} must be an object")
        prompt = raw.get("prompt")
        game_key = raw.get("game_key")
        available_moves = raw.get("available_moves")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"training dataset record {index} has no prompt")
        if not isinstance(game_key, str) or game_key not in GAMES:
            raise ValueError(f"training dataset record {index} has an unknown game_key")
        if not isinstance(available_moves, list) or not available_moves or not all(
            isinstance(move, str) and move for move in available_moves
        ):
            raise ValueError(
                f"training dataset record {index} has invalid available_moves"
            )
        if allowed_games is not None and game_key not in allowed_games:
            continue
        records.append({
            "prompt": prompt,
            "game_key": game_key,
            "strategy": str(raw.get("strategy", "")),
            "variant": str(raw.get("variant", "")),
            "available_moves": available_moves,
            "rounds_remaining": raw.get("rounds_remaining", int("0")),
        })

    if len(records) < n_samples:
        raise ValueError(
            f"staged dataset has {len(records)} eligible records; "
            f"{n_samples} required"
        )
    return Dataset.from_list(records[:n_samples])


# ---------------------------------------------------------------------------
# Reward function — full episode rollout
# ---------------------------------------------------------------------------


REWARD_STRATEGIES = ["always_defect", "tit_for_tat", "always_cooperate"]


def _build_local_prompt(obs) -> str:
    """Build prompt from a local GameObservation (env.models.GameObservation)."""
    sections = [f"[Game]\n{obs.game_name}"]
    if obs.history:
        lines = []
        for r in obs.history[-5:]:
            lines.append(
                f"Round {r.round_number}"
                f" | You played: {r.player_action}"
                f" | Opponent played: {r.opponent_action}"
                f" | Your payoff: {r.player_payoff}"
                f" | Opp payoff: {r.opponent_payoff}"
            )
        sections.append("[History]\n" + "\n".join(lines))
    sections.append(
        f"[Scores]\nYour score: {obs.player_score}"
        f"\nRound: {obs.current_round} of {obs.total_rounds}"
    )
    sections.append(
        "[Available Actions]\n"
        + "\n".join(f"- {a}" for a in obs.available_actions)
    )
    # Phase-aware instruction. The 2-phase free_chat rollout passes
    # obs.metadata["phase"] in ("message", "action"); use this to give
    # the model a sharp instruction per phase. Legacy single-phase obs
    # has no phase set -> falls back to the generic SYSTEM_PROMPT.
    _phase = (obs.metadata or {}).get("phase")
    if _phase == "message":
        sections.append(
            "[Instruction]\nSay one short sentence to your opponent before "
            "the next move. Keep it under 20 words."
        )
    elif _phase == "action":
        _act_options = " or ".join(obs.available_actions)
        sections.append(
            f"[Instruction]\nReply with EXACTLY ONE word: {_act_options}. "
            "No prose, no punctuation, no quotes — just the single word."
        )
    else:
        sections.append(f"[Instruction]\n{SYSTEM_PROMPT}")
    return "\n\n".join(sections)


def _local_coop_rate(history) -> float:
    """Cooperation rate from local RoundResult history."""
    if not history:
        return 0.0
    coop = {"cooperate", "stag", "dove", "contribute"}
    return sum(1 for r in history if any(c in r.player_action for c in coop)) / len(history)


def format_reward_fn(
    completions: list[str],
    prompts: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward function that encourages concise, exact-match action output.

    Returns 1.0 for exact match, 0.5 for case-insensitive, 0.1 for substring,
    -0.5 when no action token appears in the completion. The -0.5 is an
    explicit penalty for unparseable output, not a fallback that
    substitutes a random action; parse_action now raises ParseActionError
    for unmatched responses (no random.choice fallback).
    """
    rewards = []
    available_moves_batch = kwargs.get(
        "available_moves", [["cooperate", "defect"]] * len(completions)
    )
    for completion, moves in zip(completions, available_moves_batch):
        stripped = completion.strip()
        if stripped in moves:
            rewards.append(1.0)
        elif stripped.lower() in [m.lower() for m in moves]:
            rewards.append(0.5)
        elif any(m.lower() in stripped.lower() for m in moves):
            rewards.append(0.1)
        else:
            rewards.append(-0.5)
    return rewards
