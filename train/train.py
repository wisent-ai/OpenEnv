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
    """Generate diverse game theory prompts using LOCAL environment.

    Falls back to remote KantBench server only if local env fails.
    """
    from env.environment import KantEnvironment
    from env.models import GameAction as LocalGameAction

    game_keys = games or list(GAMES.keys())
    strat_names = strategies or list(STRATEGY_REGISTRY.keys())
    samples = []
    local_env = KantEnvironment()

    while len(samples) < n_samples:
        game_key = random.choice(game_keys)
        strategy = random.choice(strat_names)

        try:
            obs = local_env.reset(game=game_key, strategy=strategy)

            # Play 0..N-1 random rounds to create diverse game states
            rounds_to_play = random.randint(0, max(obs.total_rounds - 1, 0))
            for _ in range(rounds_to_play):
                move = random.choice(obs.available_actions)
                obs = local_env.step(LocalGameAction(action=move))
                if obs.done:
                    break

            if obs.done:
                obs = local_env.reset(game=game_key, strategy=strategy)

            prompt = _build_local_prompt(obs)

            samples.append({
                "prompt": prompt,
                "game_key": game_key,
                "strategy": strategy,
                "variant": "",
                "available_moves": list(obs.available_actions),
                "rounds_remaining": obs.total_rounds - obs.current_round,
            })
        except Exception as exc:
            logger.debug("Skipping %s/%s: %s", game_key, strategy, exc)
            continue

    return Dataset.from_list(samples)


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
    sections.append(f"[Instruction]\n{SYSTEM_PROMPT}")
    return "\n\n".join(sections)


def _batch_generate_actions(model, tokenizer, obs_list, device):
    """Generate actions for MULTIPLE observations in a single batched call.

    10-20x faster than sequential _generate_action_local calls.
    """
    if not obs_list:
        return []

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
                **inputs, max_new_tokens=16, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        actions = []
        for idx, obs in enumerate(obs_list):
            input_len = inputs["attention_mask"][idx].sum().item()
            completion_ids = outputs[idx][input_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            actions.append(parse_action(completion.strip(), obs.available_actions))
        return actions
    except RuntimeError:
        pass

    # Sequential fallback for quantized models
    actions = []
    for text, obs in zip(texts, obs_list):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=16, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        actions.append(parse_action(completion.strip(), obs.available_actions))
    return actions


def _local_coop_rate(history) -> float:
    """Cooperation rate from local RoundResult history."""
    if not history:
        return 0.0
    coop = {"cooperate", "stag", "dove", "contribute"}
    return sum(1 for r in history if any(c in r.player_action for c in coop)) / len(history)


def _play_batch_interactive_episodes(
    envs, episode_configs, model, tokenizer, device,
):
    """Play multiple interactive episodes in BATCHED mode.

    Instead of 288 sequential model.generate() calls per step,
    this does ~9 batched calls (one per round after round 1),
    each processing all active episodes simultaneously.

    Args:
        envs: list of KantEnvironment instances (one per episode)
        episode_configs: list of (game_key, strategy, first_action)
        model, tokenizer, device: for batched generation

    Returns:
        list of episode result dicts (or None for failed episodes)
    """
    from env.models import GameAction as LocalGameAction

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
            # Fallback: random action
            for i in active_indices:
                actions[i] = random.choice(obs_list[i].available_actions)

    return results


def make_reward_fn(base_url: str, model=None, tokenizer=None):
    """Returns a GRPO reward function that plays INTERACTIVE episodes
    using the LOCAL environment (no network round-trips).

    The model plays adaptively round-by-round instead of repeating a
    fixed action, enabling learning of conditional strategies.
    Uses local KantEnvironment for ~100x faster episode play.
    """
    from env.environment import KantEnvironment
    local_env = KantEnvironment()

    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Create a pool of local envs (one per concurrent episode)
    from env.environment import KantEnvironment as _KantEnv
    env_pool = [_KantEnv() for _ in range(len(REWARD_STRATEGIES) * 32)]

    def reward_fn(
        completions: list[str],
        prompts: list[str],
        **kwargs: Any,
    ) -> list[float]:
        game_keys = kwargs.get("game_key", ["prisoners_dilemma"] * len(completions))
        variants = kwargs.get("variant", [""] * len(completions))
        available_moves_batch = kwargs.get(
            "available_moves", [["cooperate", "defect"]] * len(completions)
        )

        # Parse all first actions
        first_actions = []
        for i, (completion, moves) in enumerate(zip(completions, available_moves_batch)):
            action = parse_action(completion.strip(), moves)
            first_actions.append(action)
            if i < 3:
                logger.info(
                    "Completion [%d] game=%s moves=%s -> parsed=%s | raw=%r",
                    i, game_keys[i], moves, action, completion[:200],
                )

        # Build ALL episode configs: each completion × 3 strategies
        episode_configs = []  # (game_key, strategy, first_action)
        completion_map = []   # maps episode index → completion index
        for i, (game_key, first_action) in enumerate(zip(game_keys, first_actions)):
            for strat in REWARD_STRATEGIES:
                episode_configs.append((game_key, strat, first_action))
                completion_map.append(i)

        # Play ALL episodes in batched mode
        episode_results = _play_batch_interactive_episodes(
            env_pool, episode_configs, model, tokenizer, device,
        )

        # Group results by completion and compute 5 metrics
        rewards = []
        n_strats = len(REWARD_STRATEGIES)
        for i in range(len(completions)):
            # Gather this completion's 3 episode results
            episodes = {}
            for j in range(n_strats):
                ep_idx = i * n_strats + j
                strat = REWARD_STRATEGIES[j]
                if episode_results[ep_idx] is not None:
                    episodes[strat] = episode_results[ep_idx]

            if not episodes:
                rewards.append(-1.0)
                continue

            # --- All 5 metrics with real cross-strategy data ---
            coop_rates = [ep["cooperation_rate"] for ep in episodes.values()]
            cooperation = sum(coop_rates) / len(coop_rates)

            pareto_scores = []
            for ep in episodes.values():
                joint = ep["player_score"] + ep["opponent_score"]
                if ep["rounds"] > 0:
                    pareto_scores.append(
                        max(0.0, min(1.0, joint / ep["rounds"]))
                    )
            pareto = sum(pareto_scores) / len(pareto_scores) if pareto_scores else 0.0

            fairness_scores = []
            for ep in episodes.values():
                denom = abs(ep["player_score"]) + abs(ep["opponent_score"])
                if denom > 0:
                    fairness_scores.append(
                        1.0 - abs(ep["player_score"] - ep["opponent_score"]) / denom
                    )
                else:
                    fairness_scores.append(1.0)
            fairness = sum(fairness_scores) / len(fairness_scores)

            scores_by_strat = {
                s: ep["player_score"] for s, ep in episodes.items()
            }
            if "always_defect" in scores_by_strat and len(scores_by_strat) > 1:
                best = max(scores_by_strat.values())
                worst = min(scores_by_strat.values())
                spread = best - worst
                if spread > 0:
                    exploit_resist = (scores_by_strat["always_defect"] - worst) / spread
                else:
                    exploit_resist = 0.5
            else:
                exploit_resist = 0.5

            if len(coop_rates) > 1:
                mean_c = sum(coop_rates) / len(coop_rates)
                var_c = sum((c - mean_c) ** 2 for c in coop_rates) / len(coop_rates)
                adaptability = min(var_c / 0.5, 1.0)
            else:
                adaptability = 0.0

            reward = (
                cooperation * 0.2
                + pareto * 0.2
                + fairness * 0.2
                + exploit_resist * 0.2
                + adaptability * 0.2
            )
            rewards.append(reward)

        return rewards

    return reward_fn


def _play_fixed_episode(env, game_key, strategy, action_str, variant=""):
    """Fallback: play episode with fixed action (no model). Used when
    model is not available to the reward function."""
    for attempt in range(3):
        try:
            reset_kwargs = {"game": game_key, "strategy": strategy}
            if variant:
                reset_kwargs["variant"] = variant
            result = env.reset(**reset_kwargs)
            while not result.done:
                result = env.step(KantBenchAction(move=action_str))
            obs = result.observation
            opp_score = sum(
                h.get("opponent_payoff", 0.0) for h in obs.history
            )
            return {
                "player_score": obs.cumulative_score,
                "opponent_score": opp_score,
                "cooperation_rate": _obs_cooperation_rate(obs),
                "rounds": obs.round_number,
                "strategy": strategy,
            }
        except (ConnectionError, RuntimeError, OSError) as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
        except (ValueError, KeyError):
            break
    return None


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

    # --- Model loading ---
    # Always pre-load so the reward function can use the model for
    # interactive episode play (generating actions round-by-round).
    peft_config = None
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

    # Use eager attention for compatibility with batched left-padded generation
    load_kwargs["attn_implementation"] = "eager"
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

    # Pass model + tokenizer so reward function can play interactive episodes
    reward_model = model_or_path if not isinstance(model_or_path, str) else None
    reward_fn = make_reward_fn(args.env_url, model=reward_model, tokenizer=tokenizer)

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
