"""KantBench GRPO Training Script.

Trains a language model to play 2-player game theory games optimally
using Group Relative Policy Optimization (GRPO) via TRL.

The KantBench OpenEnv Space acts as the reward oracle:
  - Each LLM completion is parsed as a game move
  - The move is submitted to the environment
  - The payoff becomes the GRPO reward signal

Usage on Northflank H100:
    pip install -r requirements.txt
    python train.py

Or with custom settings:
    python train.py --model Qwen/Qwen2.5-3B-Instruct --episodes 2000
"""

from __future__ import annotations

import argparse
import random
from typing import Any

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from common.games import GAMES
from common.strategies import STRATEGIES as STRATEGY_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KANTBENCH_URL = "https://openenv-community-kantbench.hf.space"

GAME_THEORY_SYSTEM_PROMPT = """You are an expert game theory player. You will be given the current state of a 2-player strategic game and must choose your move to maximize your long-term cumulative payoff.

Rules:
- Read the game description carefully
- Consider your opponent's strategy and history
- Respond with ONLY the move name, nothing else
- Your response must be exactly one of the available moves listed"""

# Pull games dynamically from the registry (90+ games)
GAMES_META = {
    key: {
        "name": cfg.name,
        "moves": list(cfg.actions),
        "description": cfg.description,
        "default_rounds": cfg.default_rounds,
    }
    for key, cfg in GAMES.items()
}

# All opponent strategy names from the registry
STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _make_history_str(history: list[dict]) -> str:
    if not history:
        return "No rounds played yet."
    lines = []
    for r in history:
        lines.append(
            f"  Round {r['round']}: you={r['your_move']}, "
            f"opponent={r['opponent_move']}, "
            f"your payoff={r['your_payoff']:+.1f}"
        )
    return "\n".join(lines)


def _simulate_history(moves: list[str], strategy: str, n: int) -> list[dict]:
    """Simulate n rounds of history for dataset prompt variety."""
    history = []
    for i in range(n):
        your_move = random.choice(moves)
        if strategy == "always_cooperate":
            opp_move = moves[0]
        elif strategy == "always_defect":
            opp_move = moves[-1]
        elif strategy == "tit_for_tat":
            opp_move = history[-1]["your_move"] if history else moves[0]
        else:
            opp_move = random.choice(moves)
        history.append({
            "round": i + 1,
            "your_move": your_move,
            "opponent_move": opp_move,
            "your_payoff": random.uniform(-1, 5),
        })
    return history


def build_dataset(n_samples: int = 1000) -> Dataset:
    """Generate diverse game theory prompts for GRPO training."""
    samples = []
    game_keys = list(GAMES_META.keys())
    for _ in range(n_samples):
        game_key = random.choice(game_keys)
        game = GAMES_META[game_key]
        strategy = random.choice(STRATEGY_NAMES)
        max_rounds = game["default_rounds"]
        round_num = random.randint(0, max(max_rounds - 1, 0))
        history = _simulate_history(game["moves"], strategy, round_num)
        cumulative = sum(r["your_payoff"] for r in history)

        prompt = (
            f"Game: {game['name']}\n"
            f"Description: {game['description']}\n"
            f"Available moves: {', '.join(game['moves'])}\n"
            f"Opponent strategy: {strategy}\n"
            f"Round: {round_num + 1}/{max_rounds}\n"
            f"Your cumulative score: {cumulative:.1f}\n"
            f"History:\n{_make_history_str(history)}\n\n"
            f"Your move:"
        )
        samples.append({
            "prompt": prompt,
            "game_key": game_key,
            "available_moves": game["moves"],
        })
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_reward_fn(env_url: str):
    """Returns a GRPO reward function that queries the KantBench environment."""
    try:
        from openenv.core.env_client import EnvClient
        from openenv.core.client_types import StepResult

        class KantBenchClient(EnvClient):
            def _step_payload(self, action):
                return action

            def _parse_result(self, payload):
                return StepResult(
                    observation=payload,
                    reward=float(payload.get("reward", 0.0)),
                    done=bool(payload.get("done", False)),
                )

            def _parse_state(self, payload):
                return payload

        _has_openenv = True
    except ImportError:
        _has_openenv = False

    def openenv_reward(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
        rewards = []
        available_moves_batch = kwargs.get("available_moves", [["cooperate", "defect"]] * len(completions))

        for completion, moves in zip(completions, available_moves_batch):
            # Parse the move from the LLM output
            text = completion.strip().lower()
            move = None
            for m in moves:
                if m in text:
                    move = m
                    break
            if move is None:
                # Invalid move — penalize
                rewards.append(-2.0)
                continue

            if _has_openenv:
                try:
                    with KantBenchClient(base_url=env_url) as env:
                        env.reset()
                        result = env.step({"move": move})
                        rewards.append(float(result.reward))
                except Exception:
                    # Fallback if env unreachable
                    rewards.append(0.0)
            else:
                # Fallback: simple heuristic reward (for testing without openenv)
                rewards.append(1.0 if move == moves[0] else 0.0)

        return rewards

    return openenv_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="./kantbench-grpo")
    p.add_argument("--episodes", type=int, default=1000, help="Training dataset size")
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--env-url", default=KANTBENCH_URL)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default="jtowarek/kantbench-qwen2.5-7b")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    print(f"KantBench env: {args.env_url}")
    print(f"Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.episodes)
    print(f"Dataset: {len(dataset)} prompts across {len(GAMES_META)} games")

    # Format prompts with chat template
    def format_prompt(example):
        messages = [
            {"role": "system", "content": GAME_THEORY_SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        return {"prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )}

    dataset = dataset.map(format_prompt)

    reward_fn = make_reward_fn(args.env_url)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=16,        # moves are short (1-2 words)
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        report_to="none",
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
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
