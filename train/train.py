"""KantBench GRPO Training Script.

Trains a language model to play 2-player game theory games optimally
using Group Relative Policy Optimization (GRPO) via TRL.

The KantBench environment runs locally as the reward oracle:
  - Each GRPO completion is a single move
  - The reward function plays a FULL multi-round episode using that move
    as the agent's consistent strategy
  - The composite reward (payoff + cooperation + Pareto efficiency + fairness)
    becomes the GRPO signal

Uses the existing training infrastructure:
  - PromptBuilder for structured, strategy-blind prompts
  - episode_reward() for multi-metric reward decomposition
  - TrajectoryCollector for full episode rollouts
  - Stratified train/eval splits with curriculum scheduling

Usage:
    python train.py --model Qwen/Qwen2.5-3B-Instruct --max-steps 200
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any, List

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from common.games import GAMES
from common.strategies import STRATEGIES as STRATEGY_REGISTRY
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import PromptBuilder, parse_action
from train.rewards import episode_reward
from train.splits import get_train_eval_split
from train.trajectory import _compute_cooperation_rate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KANTBENCH_URL = "https://openenv-community-kantbench.hf.space"

SYSTEM_PROMPT = (
    "You are playing a game-theory game. Analyse the situation and choose "
    "the best action. Respond with ONLY the action name, nothing else."
)

# ---------------------------------------------------------------------------
# Dataset generation using PromptBuilder
# ---------------------------------------------------------------------------


def build_dataset(
    n_samples: int = 1000,
    games: list[str] | None = None,
    strategies: list[str] | None = None,
) -> Dataset:
    """Generate diverse game theory prompts for GRPO training.

    Uses PromptBuilder for structured prompts (same format the model sees
    during episode rollouts) and simulates partial game histories so the
    model trains on various game states, not just round 1.
    """
    env = KantEnvironment()
    game_keys = games or list(GAMES.keys())
    strat_names = strategies or list(STRATEGY_REGISTRY.keys())
    prompt_builder = PromptBuilder()
    samples = []

    for _ in range(n_samples):
        game_key = random.choice(game_keys)
        strategy = random.choice(strat_names)

        # Reset env to get a real observation
        obs = env.reset(game=game_key, strategy=strategy)

        # Play 0..N-1 random rounds to create diverse game states
        max_rounds = obs.total_rounds
        rounds_to_play = random.randint(0, max(max_rounds - 1, 0))
        for _ in range(rounds_to_play):
            random_action = GameAction(action=random.choice(obs.available_actions))
            obs = env.step(random_action)
            if obs.done:
                break

        if obs.done:
            # Replay without filling all rounds
            obs = env.reset(game=game_key, strategy=strategy)

        # Build the structured prompt from the real observation
        prompt = prompt_builder.build(obs)

        samples.append({
            "prompt": prompt,
            "game_key": game_key,
            "strategy": strategy,
            "available_moves": list(obs.available_actions),
            "rounds_remaining": obs.total_rounds - obs.current_round,
        })

    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Reward function — full episode rollout
# ---------------------------------------------------------------------------


def make_reward_fn():
    """Returns a GRPO reward function that plays full episodes locally.

    For each completion:
    1. Parse the move from the LLM output
    2. Reset a local KantEnvironment with the correct game/strategy
    3. Play the FULL episode using the parsed move as a consistent strategy
    4. Compute composite reward: payoff + cooperation + Pareto + fairness
    """
    env = KantEnvironment()

    def reward_fn(
        completions: list[str],
        prompts: list[str],
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        game_keys = kwargs.get("game_key", ["prisoners_dilemma"] * len(completions))
        strategies = kwargs.get("strategy", ["tit_for_tat"] * len(completions))
        available_moves_batch = kwargs.get(
            "available_moves", [["cooperate", "defect"]] * len(completions)
        )

        for completion, game_key, strategy, moves in zip(
            completions, game_keys, strategies, available_moves_batch
        ):
            # Parse move from LLM output
            action_str = parse_action(completion.strip(), moves)

            try:
                # Play a full episode using this move as the agent's strategy
                obs = env.reset(game=game_key, strategy=strategy)
                while not obs.done:
                    obs = env.step(GameAction(action=action_str))

                # Compute cooperation rate
                coop_rate = _compute_cooperation_rate(obs)

                # Composite reward from the reward module
                reward = episode_reward(
                    player_score=obs.player_score,
                    opponent_score=obs.opponent_score,
                    cooperation_rate=coop_rate,
                    total_rounds=obs.current_round,
                )
                rewards.append(reward)

            except (ValueError, KeyError, RuntimeError) as exc:
                logger.debug("Reward error for %s/%s: %s", game_key, action_str, exc)
                rewards.append(-1.0)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="KantBench GRPO Training")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="./kantbench-grpo")
    p.add_argument("--episodes", type=int, default=1000, help="Training dataset size")
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--report-to", default="wandb", help="wandb, tensorboard, or none")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default="jtowarek/kantbench-qwen2.5-7b")
    p.add_argument("--use-train-split", action="store_true",
                    help="Use stratified train/eval split (eval games held out)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    print(f"Loading model: {args.model}")
    print(f"Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally use stratified train/eval split
    train_games = None
    if args.use_train_split:
        train_set, eval_set = get_train_eval_split()
        train_games = sorted(train_set)
        print(f"Using stratified split: {len(train_games)} train, {len(eval_set)} eval games")

    dataset = build_dataset(args.episodes, games=train_games)
    print(f"Dataset: {len(dataset)} prompts across {len(GAMES)} games")

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

    reward_fn = make_reward_fn()

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=16,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    print(f"  Reward: composite (payoff + cooperation + Pareto + fairness)")
    print(f"  Episode: full multi-round rollout per completion")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
