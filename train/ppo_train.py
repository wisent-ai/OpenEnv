"""KantBench REINFORCE Training Script.

REINFORCE with running-mean baseline as structural alternative to GRPO.
Fixes mode collapse by decoupling the advantage baseline from within-group
reward variance — the baseline is a global EMA, not a within-batch mean.
This means the model gets a learning signal even when all completions in a
batch share the same reward (the GRPO failure mode).

Architecture:
  - Policy model (fine-tuned) + frozen reference model for KL penalty
  - Per-step: generate → reward → log_prob forward pass → update
  - Advantage = reward - baseline;  baseline = EMA(rewards, alpha=0.01)
  - Loss = -mean(log_prob_sum * advantage) + kl_coef * KL(policy || ref)

Usage:
    python -m train.ppo_train --model meta-llama/Llama-3.2-1B-Instruct --max-steps 500
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
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

    # Adaptability: reward conditional behavior (cooperate with cooperators,
    # resist defectors) rather than raw variance of cooperation rates.
    # Variance never fires when the model already always-cooperates, creating
    # a dead gradient that lets the model stay at adaptability=0 indefinitely.
    # The new formula is maximised by TFT-style play: coop_vs_coop=1, coop_vs_defect=0.
    if "always_cooperate" in episodes and "always_defect" in episodes:
        coop_vs_coop = episodes["always_cooperate"]["cooperation_rate"]
        coop_vs_defect = episodes["always_defect"]["cooperation_rate"]
        adaptability = coop_vs_coop * (1.0 - coop_vs_defect)
    elif len(coop_rates) > 1:
        mean_c = sum(coop_rates) / len(coop_rates)
        var_c = sum((c - mean_c) ** 2 for c in coop_rates) / len(coop_rates)
        adaptability = min(var_c / 0.5, 1.0)
    else:
        adaptability = 0.0

    # Weight rebalance: reduce pure-cooperation incentive (drives always-cooperate
    # collapse), raise exploitation-resistance (underpowered at 20% previously).
    return (cooperation * 0.1 + pareto * 0.2 + fairness * 0.2
            + exploit_resist * 0.3 + adaptability * 0.2)


def compute_log_probs(model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Forward pass to get sum of log probs for the generated tokens only."""
    with torch.no_grad():
        pass  # will be called with grad enabled in training
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, seq_len, vocab]

    # Shift: logits[i] predicts token[i+1]
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)  # [seq_len-1, vocab]
    tokens = input_ids[0, 1:]  # [seq_len-1]

    # Only score the response tokens (after the prompt)
    response_log_probs = log_probs[response_start - 1:]  # prompt ends at response_start
    response_tokens = tokens[response_start - 1:]

    per_token = response_log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
    return per_token.sum()


def compute_kl(model, ref_model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Compute KL(policy || ref) for the response tokens."""
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids)
        ref_logits = ref_outputs.logits[0, :-1]  # [seq_len-1, vocab]

    policy_outputs = model(input_ids=input_ids)
    policy_logits = policy_outputs.logits[0, :-1]

    # KL per token position in response
    start = response_start - 1
    ref_lp = F.log_softmax(ref_logits[start:], dim=-1)
    pol_lp = F.log_softmax(policy_logits[start:], dim=-1)

    # KL(pol || ref) = sum(pol * (log_pol - log_ref))
    kl_per_token = (pol_lp.exp() * (pol_lp - ref_lp)).sum(-1)
    return kl_per_token.sum(), policy_logits


def parse_args():
    p = argparse.ArgumentParser(description="KantBench REINFORCE Training")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output-dir", default="./kantbench-reinforce")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--baseline-ema", type=float, default=0.01,
                   help="EMA decay for running reward baseline")
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

    print(f"REINFORCE Training: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"KL coef: {args.kl_coef}, Temperature: {args.temperature}")
    print("Mode collapse fix: running-mean baseline (no group variance dependency)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading policy model...")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
    }

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Frozen reference model for KL constraint
    print("Loading frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Dataset
    train_games = None
    if args.use_train_split:
        train_set, _ = get_train_eval_split()
        train_games = sorted(train_set)

    dataset = build_local_dataset(args.episodes, games=train_games)
    print(f"Dataset: {len(dataset)} prompts")

    # Format with chat template
    formatted_queries = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["query"]},
        ]
        formatted_queries.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # Wandb setup
    if args.report_to == "wandb":
        try:
            import wandb
            wandb.init(
                project="kantbench-grpo",
                name=args.wandb_run_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"Wandb init failed: {e}")

    device = next(model.parameters()).device
    env_pool = [KantEnvironment() for _ in range(len(REWARD_STRATEGIES) * args.batch_size)]

    # Running baseline — EMA of observed rewards
    baseline = 0.5

    generation_kwargs = {
        "max_new_tokens": 16,
        "temperature": args.temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    step = 0
    indices = list(range(len(dataset)))
    epoch = 0

    print(f"Starting REINFORCE training ({args.max_steps} steps)...", flush=True)
    step_start = time.time()

    while step < args.max_steps:
        if epoch * args.batch_size >= len(indices):
            random.shuffle(indices)
            epoch = 0

        batch_idx = indices[epoch * args.batch_size: (epoch + 1) * args.batch_size]
        epoch += 1

        # --- Generate completions ---
        model.eval()
        query_ids_list = []
        for i in batch_idx:
            q = formatted_queries[i]
            ids = tokenizer.encode(q, return_tensors="pt").to(device)
            query_ids_list.append(ids)

        completions = []
        full_ids_list = []
        with torch.no_grad():
            for ids in query_ids_list:
                out = model.generate(ids, **generation_kwargs)
                full_ids_list.append(out)
                new_tokens = out[0, ids.shape[1]:]
                completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        if step == 0:
            print(f"[step 0] Generated completions: {completions[:2]}", flush=True)

        # --- Compute rewards ---
        rewards = []
        t_rew = time.time()
        for i, completion in zip(batch_idx, completions):
            game_key = dataset[i]["game_key"]
            moves = dataset[i]["available_moves"]
            r = compute_reward(completion, game_key, moves, model, tokenizer, device, env_pool)
            rewards.append(r)

        if step == 0:
            print(f"[step 0] Rewards computed in {time.time()-t_rew:.1f}s: {rewards}", flush=True)

        mean_reward = sum(rewards) / len(rewards)

        # --- REINFORCE update ---
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_kl = 0.0

        for i, (ids, full_ids, reward) in enumerate(zip(query_ids_list, full_ids_list, rewards)):
            prompt_len = ids.shape[1]

            # Advantage with running baseline (structural fix — no group variance needed)
            advantage = reward - baseline

            # KL + policy log probs in one forward pass
            kl, policy_logits = compute_kl(model, ref_model, full_ids, prompt_len)
            total_kl += kl.item()

            # Log probs of generated tokens (from the policy forward pass we already did)
            log_probs_tokens = F.log_softmax(policy_logits[prompt_len - 1:], dim=-1)
            gen_tokens = full_ids[0, prompt_len:]
            if len(gen_tokens) == 0:
                continue
            log_prob_sum = log_probs_tokens[:len(gen_tokens)].gather(
                -1, gen_tokens.unsqueeze(-1)
            ).squeeze(-1).sum()

            # REINFORCE loss: maximize log_prob * advantage
            loss = -(log_prob_sum * advantage) + args.kl_coef * kl
            total_loss = total_loss + loss / len(batch_idx)

        # Update baseline (EMA)
        baseline = (1 - args.baseline_ema) * baseline + args.baseline_ema * mean_reward

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0
        )
        optimizer.step()

        step += 1
        elapsed = time.time() - step_start
        step_start = time.time()

        print(
            f"Step {step}/{args.max_steps}: "
            f"reward={mean_reward:.4f} "
            f"baseline={baseline:.4f} "
            f"kl={total_kl / max(len(batch_idx), 1):.4f} "
            f"loss={total_loss.item():.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
            if args.report_to == "wandb":
                try:
                    import wandb
                    wandb.log({
                        "reward": mean_reward,
                        "baseline": baseline,
                        "kl": total_kl / max(len(batch_idx), 1),
                        "loss": total_loss.item(),
                    }, step=step)
                except Exception:
                    pass

        if step % args.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint to {ckpt}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"REINFORCE training complete. Model saved to {args.output_dir}")

    if args.report_to == "wandb":
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
