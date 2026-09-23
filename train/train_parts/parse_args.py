"""Parts of train.py, split by the tama size splitter; train.py imports every name back."""

from __future__ import annotations
import argparse
import logging
import torch
logger = logging.getLogger(__name__)
from train_parts.system_prompt import SYSTEM_PROMPT, _build_local_prompt, _local_coop_rate


def _batch_generate_actions(model, tokenizer, obs_list, device):
    """Generate actions for MULTIPLE observations in a single batched call.

    10-20x faster than sequential _generate_action_local calls.
    """
    if not obs_list:
        return []

    # Phase-aware generation params: the 2-phase free_chat rollout
    # sets obs.metadata["phase"] to "message" or "action"; all obs in a
    # single batch share the same phase. Action phase wants a tight,
    # near-greedy decode (temp 0.1, 4 tokens) so the model emits a
    # single action token rather than wandering for 16 tokens. Message
    # phase keeps higher temp + longer budget for short prose. Legacy
    # single-phase obs (no metadata.phase) keeps the original config.
    _phase = (obs_list[0].metadata or {}).get("phase") if obs_list else None
    if _phase == "action":
        _gen_max, _gen_temp = 4, 0.1
    elif _phase == "message":
        _gen_max, _gen_temp = 24, 0.7
    else:
        _gen_max, _gen_temp = 16, 0.7

    # Build all prompts
    texts = []
    for obs in obs_list:
        prompt = _build_local_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Try batched generation first, fall back to sequential if it fails
    # (4-bit quantized models often can't handle batched padded inputs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=_gen_max, temperature=_gen_temp,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        actions = []
        for idx, obs in enumerate(obs_list):
            input_len = inputs["attention_mask"][idx].sum().item()
            completion_ids = outputs[idx][input_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            # Always return RAW completion. Downstream callers
            # (play_batch_free_chat_episodes and the legacy single-phase loop
            # in _play_batch_interactive_episodes) each have their own
            # parse_action + per-episode try/except, so a parse miss on one
            # completion no longer kills the whole batch.
            actions.append(completion.strip())
        return actions
    except RuntimeError as exc:
        # Narrowly handle the documented quantization-shape error and re-raise
        # everything else (OOM, kernel-launch failures, real bugs) so they
        # surface instead of being silently re-tried in sequential mode.
        msg = str(exc).lower()
        if not ("quantiz" in msg or "4-bit" in msg or "shape" in msg or "padding" in msg):
            raise
        # Recognised batched-vs-quantized incompatibility — log and continue
        # to the sequential path.
        logger.warning("batched generate failed (quantization-related): %s; falling back to sequential", exc)

    # Sequential fallback for quantized models
    actions = []
    for text, obs in zip(texts, obs_list):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=_gen_max, temperature=_gen_temp,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        actions.append(completion.strip())
    return actions


def _play_batch_interactive_episodes(
    envs, episode_configs, model, tokenizer, device,
):
    """Play multiple interactive episodes in BATCHED mode.

    Instead of 288 sequential model.generate() calls per step,
    this does ~9 batched calls (one per round after round 1),
    each processing all active episodes simultaneously.

    Free_chat dispatch: if EVERY game in the batch is a free_chat
    variant, route to the 2-phase rollout in train.free_to_ppo.free_chat_rollouts
    (one model call per phase per round, env.step called twice). Mixed
    batches fall through to the legacy single-phase loop, which now
    raises if any free_chat game appears so the caller has to bucket
    free_chat envs separately rather than silently treating them as
    single-phase.
    """
    from env.models import GameAction as LocalGameAction
    from train.free_to_ppo.free_chat_rollouts import (
        play_batch_free_chat_episodes,
        is_free_chat_game,
    )

    fc_flags = [is_free_chat_game(c[0]) for c in episode_configs]
    if all(fc_flags) and fc_flags:
        return play_batch_free_chat_episodes(
            envs, episode_configs, model, tokenizer, device,
            batch_generate_fn=_batch_generate_actions,
        )
    if any(fc_flags):
        raise ValueError(
            "Mixed free_chat / non-free_chat batch in "
            "_play_batch_interactive_episodes. Bucket free_chat games "
            "separately so the 2-phase rollout dispatch is unambiguous."
        )

    n = len(episode_configs)
    results = [None] * n

    # Initialize all environments
    obs_list = [None] * n
    actions = [None] * n
    active = [True] * n

    for i, (game_key, strategy, first_action) in enumerate(episode_configs):
        try:
            obs_list[i] = envs[i].reset(game=game_key, strategy=strategy)
            actions[i] = first_action
        except Exception as exc:
            logger.debug("Init error %s/%s: %s", game_key, strategy, exc)
            active[i] = False

    # Play rounds until all episodes finish
    max_rounds = 20  # safety limit
    for _ in range(max_rounds):
        # Step all active episodes with their current actions
        for i in range(n):
            if not active[i]:
                continue
            try:
                action_str = actions[i]
                if action_str not in obs_list[i].available_actions:
                    action_str = parse_action(action_str, obs_list[i].available_actions)
                obs_list[i] = envs[i].step(LocalGameAction(action=action_str))
                if obs_list[i].done:
                    active[i] = False
                    obs = obs_list[i]
                    coop_actions = {"cooperate", "stag", "dove", "contribute"}
                    opp_coop = (
                        sum(1 for r in obs.history
                            if any(c in r.opponent_action for c in coop_actions))
                        / len(obs.history)
                        if obs.history else 0.0
                    )
                    results[i] = {
                        "player_score": obs.player_score,
                        "opponent_score": obs.opponent_score,
                        "cooperation_rate": _local_coop_rate(obs.history),
                        "opponent_cooperation_rate": opp_coop,
                        "rounds": obs.current_round,
                        "strategy": episode_configs[i][1],
                    }
            except Exception as exc:
                logger.debug("Step error episode %d: %s", i, exc)
                active[i] = False

        # Collect observations from still-active episodes for batched generation
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break

        active_obs = [obs_list[i] for i in active_indices]
        if model is not None and tokenizer is not None:
            batch_actions = _batch_generate_actions(
                model, tokenizer, active_obs, device,
            )
            for j, i in enumerate(active_indices):
                actions[i] = batch_actions[j]
        else:
            # No silent random fallback — when model+tokenizer aren't
            # available the reward function MUST surface the failure rather than
            # contaminate the training distribution with coin-flipped actions.
            for i in active_indices:
                raise RuntimeError(
                    f"_play_batch_interactive_episodes called without model+tokenizer; "
                    f"refusing to substitute random actions (active env={i}). "
                    f"Pass model and tokenizer to the reward function or fix "
                    f"the calling code path."
                )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="KantBench GRPO Training")
    p.add_argument("--model-path", required=True,
                   help="Absolute path to the machine-staged model directory")
    p.add_argument("--data-path", required=True,
                   help="Absolute path to the machine-staged JSON or JSONL dataset")
    p.add_argument("--output-dir", default="./kantbench-grpo")
    p.add_argument("--episodes", type=int, default=int("1000"),
                   help="Number of staged dataset records to consume")
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=50,
                    help="Checkpoint save interval (steps)")
    p.add_argument("--temperature", type=float, default=0.8,
                    help="Generation temperature (higher = more GRPO diversity)")
    p.add_argument(
        "--report-to",
        choices=("none", "tensorboard"),
        default="none",
        help="Local-only trainer reporting backend",
    )
    p.add_argument("--use-train-split", action="store_true",
                    help="Use stratified train/eval split (eval games held out)")
    p.add_argument("--games", default=None,
                    help="Comma-separated game keys to restrict the training dataset to (overrides --use-train-split).")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Path to checkpoint or 'latest' to resume training")
    # LoRA / QLoRA options
    p.add_argument("--use-lora", action="store_true",
                    help="Use LoRA for parameter-efficient training")
    p.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32,
                    help="LoRA alpha scaling factor")
    p.add_argument("--quantize-4bit", action="store_true",
                    help="Load model in 4-bit quantization (requires bitsandbytes)")
    p.add_argument("--kl-beta", type=float, default=0.1,
                    help="KL penalty coefficient (higher = more diverse outputs, fights mode collapse)")
    return p.parse_args()
