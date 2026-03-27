"""KantBench PPO Training Script.

PPO alternative to GRPO that doesn't suffer from mode collapse on
short-completion tasks. Uses a value head (critic) for advantage
estimation instead of within-group reward comparison.

Usage:
    python -m train.ppo_train --model meta-llama/Llama-3.2-1B-Instruct --max-steps 500
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.games import GAMES
from common.strategies import STRATEGIES as STRATEGY_REGISTRY
from env.environment import KantEnvironment
from env.models import GameAction as LocalGameAction
from train.agent import parse_action
from train.splits import get_train_eval_split
from train.train import (
    SYSTEM_PROMPT,
    REWARD_STRATEGIES,
    _build_local_prompt,
    _batch_generate_actions,
    _play_batch_interactive_episodes,
    _local_coop_rate,
)

logger = logging.getLogger(__name__)


def build_local_dataset(n_samples=1000, games=None, strategies=None):
    """Build training prompts using local environment."""
    game_keys = games or list(GAMES.keys())
    strat_names = strategies or list(STRATEGY_REGISTRY.keys())
    local_env = KantEnvironment()
    samples = []

    while len(samples) < n_samples:
        game_key = random.choice(game_keys)
        strategy = random.choice(strat_names)
        try:
            obs = local_env.reset(game=game_key, strategy=strategy)
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
                "query": prompt,
                "game_key": game_key,
                "available_moves": list(obs.available_actions),
            })
        except Exception:
            continue

    return Dataset.from_list(samples)


def compute_reward(completion: str, game_key: str, moves: list[str],
                   model, tokenizer, device, env_pool) -> float:
    """Compute reward for a single completion by playing interactive episodes."""
    first_action = parse_action(completion.strip(), moves)

    episode_configs = [(game_key, strat, first_action) for strat in REWARD_STRATEGIES]
    results = _play_batch_interactive_episodes(
        env_pool, episode_configs, model, tokenizer, device,
    )

    episodes = {}
    for j, strat in enumerate(REWARD_STRATEGIES):
        if results[j] is not None:
            episodes[strat] = results[j]

    if not episodes:
        return -1.0

    coop_rates = [ep["cooperation_rate"] for ep in episodes.values()]
    cooperation = sum(coop_rates) / len(coop_rates)

    pareto_scores = []
    for ep in episodes.values():
        joint = ep["player_score"] + ep["opponent_score"]
        if ep["rounds"] > 0:
            pareto_scores.append(max(0.0, min(1.0, joint / ep["rounds"])))
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

    scores_by_strat = {s: ep["player_score"] for s, ep in episodes.items()}
    if "always_defect" in scores_by_strat and len(scores_by_strat) > 1:
        best = max(scores_by_strat.values())
        worst = min(scores_by_strat.values())
        spread = best - worst
        exploit_resist = (scores_by_strat["always_defect"] - worst) / spread if spread > 0 else 0.5
    else:
        exploit_resist = 0.5

    if len(coop_rates) > 1:
        mean_c = sum(coop_rates) / len(coop_rates)
        var_c = sum((c - mean_c) ** 2 for c in coop_rates) / len(coop_rates)
        adaptability = min(var_c / 0.5, 1.0)
    else:
        adaptability = 0.0

    return (cooperation * 0.2 + pareto * 0.2 + fairness * 0.2
            + exploit_resist * 0.2 + adaptability * 0.2)


def parse_args():
    p = argparse.ArgumentParser(description="KantBench PPO Training")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output-dir", default="./kantbench-ppo")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--use-train-split", action="store_true")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    print(f"PPO Training: {args.model}")
    print(f"Output: {args.output_dir}")

    # Import PPO from TRL
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except ImportError:
        print("ERROR: trl with PPO support required. Install with: pip install trl>=0.12.0")
        raise

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model with value head for PPO
    print("Loading model with value head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            peft_config=peft_config,
        )

    # Dataset
    train_games = None
    if args.use_train_split:
        train_set, _ = get_train_eval_split()
        train_games = sorted(train_set)

    dataset = build_local_dataset(args.episodes, games=train_games)
    print(f"Dataset: {len(dataset)} prompts")

    # Format with chat template
    def format_query(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["query"]},
        ]
        return {
            "query": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }
    dataset = dataset.map(format_query)

    # PPO config — note: no need for high temp or kl_beta
    # PPO doesn't have the mode collapse problem
    config = PPOConfig(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        ppo_epochs=4,
        init_kl_coef=args.kl_coef,
        log_with=args.report_to if args.report_to != "none" else None,
    )

    trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Env pool for reward computation
    env_pool = [KantEnvironment() for _ in range(len(REWARD_STRATEGIES) * args.batch_size)]
    device = next(model.parameters()).device

    # Training loop
    print(f"Starting PPO training ({args.max_steps} steps)...")
    generation_kwargs = {
        "max_new_tokens": 16,
        "temperature": args.temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    step = 0
    for epoch, batch in enumerate(trainer.dataloader):
        if step >= args.max_steps:
            break

        # Generate responses
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze() for q in batch["query"]]
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)

        # Decode and compute rewards
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        rewards = []
        for i, response in enumerate(responses):
            game_key = batch["game_key"][i]
            moves = batch["available_moves"][i]
            r = compute_reward(response, game_key, moves, model, tokenizer, device, env_pool)
            rewards.append(torch.tensor(r))

        # PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        step += 1
        if step % 10 == 0:
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            print(f"Step {step}/{args.max_steps}: reward={mean_reward:.4f} kl={stats.get('objective/kl', 0):.4f}")

        if step % args.save_steps == 0:
            trainer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{step}"))

    trainer.save_pretrained(args.output_dir)
    print(f"PPO training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
