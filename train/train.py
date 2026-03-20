"""KantBench GRPO Training Script.

Trains a language model to play 2-player game theory games optimally
using Group Relative Policy Optimization (GRPO) via TRL.

The KantBench environment runs as a remote OpenEnv server (HF Space):
  - Each GRPO completion is a single move
  - The reward function plays a FULL multi-round episode using that move
    as the agent's consistent strategy via the OpenEnv client
  - The composite reward (payoff + cooperation + Pareto efficiency + fairness)
    becomes the GRPO signal

Supports the full KantBench game library including:
  - 90+ base 2-player games and 3 N-player games
  - 9 pre-registered meta-games (rule_proposal, rule_signal, gossip)
  - Dynamic variant composition (cheap_talk, exit, binding_commitment,
    constitutional, proposer_responder, noisy_actions, noisy_payoffs)

Usage:
    python -m train.train --model Qwen/Qwen2.5-7B-Instruct --max-steps 200
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Any, List

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.games import GAMES
from common.strategies import STRATEGIES as STRATEGY_REGISTRY
from spaces.kant.client import KantBenchEnv
from spaces.kant.models import KantBenchAction, KantBenchObservation
from train.agent import parse_action
from train.rewards import episode_reward
from train.splits import get_train_eval_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KANTBENCH_URL = "https://openenv-community-kantbench.hf.space"

SYSTEM_PROMPT = (
    "You are playing a game-theory game. Analyse the situation and choose "
    "the best action. Respond with ONLY the action name, nothing else."
)

# Variants that can be dynamically composed on top of base games.
# These are applied server-side via the variant= reset parameter.
TRAINABLE_VARIANTS = [
    "cheap_talk",
    "exit",
    "binding_commitment",
    "constitutional",
    "noisy_actions",
    "noisy_payoffs",
    "rule_proposal",
    "rule_signal",
    "gossip",
]

# Base games suitable for variant composition (2-player matrix games).
VARIANT_BASE_GAMES = [
    "prisoners_dilemma",
    "stag_hunt",
    "hawk_dove",
]

# Fraction of dataset samples that use dynamic variant composition.
VARIANT_FRACTION = 0.3


# ---------------------------------------------------------------------------
# Helpers to bridge KantBenchObservation -> training code
# ---------------------------------------------------------------------------


def _obs_cooperation_rate(obs: KantBenchObservation) -> float:
    """Compute cooperation rate from a KantBenchObservation's history."""
    if not obs.history:
        return 0.0
    coop_actions = {"cooperate", "stag", "dove", "contribute"}
    coop_count = sum(
        1 for h in obs.history
        if any(ca in h.get("your_move", "") for ca in coop_actions)
    )
    return coop_count / len(obs.history)


def _build_prompt(obs: KantBenchObservation) -> str:
    """Build a structured prompt from a KantBenchObservation.

    Mirrors PromptBuilder.build() but works with the OpenEnv client's
    observation format.
    """
    sections: list[str] = []

    # Game section
    sections.append(
        f"[Game]\n{obs.game_name}\n{obs.game_description}"
    )

    # History section
    if obs.history:
        history_lines: list[str] = []
        for h in obs.history[-5:]:  # Last 5 rounds
            line = (
                f"Round {h.get('round', '?')}"
                f" | You played: {h.get('your_move', '?')}"
                f" | Opponent played: {h.get('opponent_move', '?')}"
                f" | Your payoff: {h.get('your_payoff', '?')}"
                f" | Opp payoff: {h.get('opponent_payoff', '?')}"
            )
            history_lines.append(line)
        sections.append("[History]\n" + "\n".join(history_lines))

    # Scores section
    sections.append(
        f"[Scores]\nYour score: {obs.cumulative_score}"
        f"\nRound: {obs.round_number} of {obs.max_rounds}"
    )

    # Available actions
    action_lines = [f"- {a}" for a in obs.available_moves]
    sections.append("[Available Actions]\n" + "\n".join(action_lines))

    # Instruction
    sections.append(f"[Instruction]\n{SYSTEM_PROMPT}")

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# Dataset generation using PromptBuilder
# ---------------------------------------------------------------------------


def build_dataset(
    base_url: str,
    n_samples: int = 1000,
    games: list[str] | None = None,
    strategies: list[str] | None = None,
    variant_fraction: float = VARIANT_FRACTION,
) -> Dataset:
    """Generate diverse game theory prompts for GRPO training.

    Connects to the KantBench OpenEnv server to generate real observations,
    then builds structured prompts from diverse game states.

    A fraction of samples use dynamic variant composition (cheap_talk,
    constitutional, gossip, etc.) to train on meta-gaming scenarios.
    """
    game_keys = games or list(GAMES.keys())
    strat_names = strategies or list(STRATEGY_REGISTRY.keys())
    samples = []

    with KantBenchEnv(base_url=base_url) as env:
        attempts = 0
        while len(samples) < n_samples:
            attempts += 1

            # Decide whether to use a variant
            use_variant = random.random() < variant_fraction
            if use_variant:
                game_key = random.choice(VARIANT_BASE_GAMES)
                variant = random.choice(TRAINABLE_VARIANTS)
            else:
                game_key = random.choice(game_keys)
                variant = None

            strategy = random.choice(strat_names)

            try:
                # Reset env — pass variant for dynamic composition
                reset_kwargs = {"game": game_key, "strategy": strategy}
                if variant:
                    reset_kwargs["variant"] = variant

                result = env.reset(**reset_kwargs)
                obs = result.observation

                # Play 0..N-1 random rounds to create diverse game states
                max_rounds = obs.max_rounds
                rounds_to_play = random.randint(0, max(max_rounds - 1, 0))
                for _ in range(rounds_to_play):
                    move = random.choice(obs.available_moves)
                    result = env.step(KantBenchAction(move=move))
                    obs = result.observation
                    if result.done:
                        break

                if result.done:
                    # Replay without filling all rounds
                    result = env.reset(**reset_kwargs)
                    obs = result.observation

                prompt = _build_prompt(obs)

                samples.append({
                    "prompt": prompt,
                    "game_key": game_key,
                    "strategy": strategy,
                    "variant": variant or "",
                    "available_moves": list(obs.available_moves),
                    "rounds_remaining": obs.max_rounds - obs.round_number,
                })
            except (RuntimeError, ConnectionError, Exception) as exc:
                logger.debug(
                    "Skipping %s/%s (variant=%s): %s",
                    game_key, strategy, variant, exc,
                )
                continue

    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Reward function — full episode rollout
# ---------------------------------------------------------------------------


def make_reward_fn(base_url: str):
    """Returns a GRPO reward function that plays full episodes via OpenEnv.

    For each completion:
    1. Parse the move from the LLM output
    2. Reset the KantBench server with the correct game/strategy/variant
    3. Play the FULL episode using the parsed move as a consistent strategy
    4. Compute composite reward: payoff + cooperation + Pareto + fairness
    """
    env = KantBenchEnv(base_url=base_url)
    env.connect()

    def reward_fn(
        completions: list[str],
        prompts: list[str],
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        game_keys = kwargs.get("game_key", ["prisoners_dilemma"] * len(completions))
        strategies = kwargs.get("strategy", ["tit_for_tat"] * len(completions))
        variants = kwargs.get("variant", [""] * len(completions))
        available_moves_batch = kwargs.get(
            "available_moves", [["cooperate", "defect"]] * len(completions)
        )

        for i, (completion, game_key, strategy, variant, moves) in enumerate(zip(
            completions, game_keys, strategies, variants, available_moves_batch
        )):
            # Parse move from LLM output
            action_str = parse_action(completion.strip(), moves)

            # Log first few completions per batch for debugging
            if i < 3:
                logger.info(
                    "Completion [%d] game=%s moves=%s -> parsed=%s | raw=%r",
                    i, game_key, moves, action_str, completion[:200],
                )

            # Retry with exponential backoff for transient connection issues
            reward = -1.0
            for attempt in range(3):
                try:
                    reset_kwargs = {"game": game_key, "strategy": strategy}
                    if variant:
                        reset_kwargs["variant"] = variant

                    result = env.reset(**reset_kwargs)
                    while not result.done:
                        result = env.step(KantBenchAction(move=action_str))

                    obs = result.observation
                    coop_rate = _obs_cooperation_rate(obs)

                    opp_score = sum(
                        h.get("opponent_payoff", 0.0) for h in obs.history
                    )
                    reward = episode_reward(
                        player_score=obs.cumulative_score,
                        opponent_score=opp_score,
                        cooperation_rate=coop_rate,
                        total_rounds=obs.round_number,
                    )
                    break
                except (ConnectionError, RuntimeError, OSError) as exc:
                    if attempt < 2:
                        wait = 2 ** attempt
                        logger.warning(
                            "Reward retry %d/3 for %s/%s: %s (wait %ds)",
                            attempt + 1, game_key, action_str, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.debug(
                            "Reward error for %s/%s after 3 attempts: %s",
                            game_key, action_str, exc,
                        )
                except (ValueError, KeyError) as exc:
                    logger.debug("Reward error for %s/%s: %s", game_key, action_str, exc)
                    break
            rewards.append(reward)

        return rewards

    return reward_fn


def format_reward_fn(
    completions: list[str],
    prompts: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward function that encourages concise, exact-match action output.

    Returns 1.0 for exact match, 0.5 for case-insensitive, 0.1 for substring,
    -0.5 for random fallback (action not found in output).
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="KantBench GRPO Training")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="./kantbench-grpo")
    p.add_argument("--env-url", default=KANTBENCH_URL,
                    help="KantBench OpenEnv server URL")
    p.add_argument("--episodes", type=int, default=1000, help="Training dataset size")
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=50,
                    help="Checkpoint save interval (steps)")
    p.add_argument("--temperature", type=float, default=0.8,
                    help="Generation temperature (higher = more GRPO diversity)")
    p.add_argument("--report-to", default="wandb", help="wandb, tensorboard, or none")
    p.add_argument("--wandb-run-name", type=str, default=None,
                    help="wandb run name for identification")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default="jtowarek/kantbench-qwen2.5-7b")
    p.add_argument("--use-train-split", action="store_true",
                    help="Use stratified train/eval split (eval games held out)")
    p.add_argument("--variant-fraction", type=float, default=VARIANT_FRACTION,
                    help="Fraction of samples using dynamic variant composition")
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    print(f"Loading model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"OpenEnv server: {args.env_url}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model loading (with optional 4-bit quantization) ---
    model_or_path = args.model
    peft_config = None

    if args.use_lora or args.quantize_4bit:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        if args.quantize_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"Loading with 4-bit quantization")

        model_or_path = AutoModelForCausalLM.from_pretrained(
            args.model, **load_kwargs
        )

        if args.use_lora:
            from peft import LoraConfig, TaskType
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            print(f"Using LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # Optionally use stratified train/eval split
    train_games = None
    if args.use_train_split:
        train_set, eval_set = get_train_eval_split()
        train_games = sorted(train_set)
        print(f"Using stratified split: {len(train_games)} train, {len(eval_set)} eval games")

    dataset = build_dataset(
        args.env_url, args.episodes, games=train_games,
        variant_fraction=args.variant_fraction,
    )
    variant_count = sum(1 for v in dataset["variant"] if v)
    print(f"Dataset: {len(dataset)} prompts across {len(GAMES)} games")
    print(f"  Variant samples: {variant_count} ({variant_count*100//max(len(dataset),1)}%)")

    # Format prompts with chat template
    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }

    dataset = dataset.map(format_prompt)

    reward_fn = make_reward_fn(args.env_url)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=32,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        beta=args.kl_beta,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        generation_kwargs={"temperature": args.temperature},
    )

    # Add newline token as an extra EOS so generation stops after one line
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
    if newline_token_id:
        config.generation_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id, newline_token_id[0],
        ]

    trainer_kwargs = {
        "model": model_or_path,
        "reward_funcs": [reward_fn, format_reward_fn],
        "args": config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        # Check if any checkpoint actually exists; if not, start fresh
        import glob as _glob
        ckpt_dirs = _glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if ckpt_dirs:
            resume_ckpt = True  # Trainer auto-finds latest checkpoint in output_dir
        else:
            print("No existing checkpoints found, starting fresh.")
            resume_ckpt = None

    print("Starting GRPO training...")
    print(f"  Reward: composite (payoff + cooperation + Pareto + fairness)")
    print(f"  Episode: full multi-round rollout via OpenEnv @ {args.env_url}")
    print(f"  Variants: {args.variant_fraction*100:.0f}% of samples use dynamic composition")
    if resume_ckpt:
        print(f"  Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
