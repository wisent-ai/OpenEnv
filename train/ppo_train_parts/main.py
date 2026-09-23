"""Parts of ppo_train.py, split by the tama size splitter; ppo_train.py imports every name back."""

from __future__ import annotations
import logging
import os
import random
import time
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from env.environment import KantEnvironment
from train.rewards_to_trajectory.splits import get_train_eval_split
from train.train import SYSTEM_PROMPT, REWARD_STRATEGIES, load_staged_dataset, require_local_artifact, require_stado_workload
from ppo_train_parts.compute_reward import compute_kl, compute_reward, parse_args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    require_stado_workload()

    model_path = require_local_artifact(
        args.model_path, "training model", directory=True
    )
    print(f"REINFORCE Training: {model_path}")
    print(f"Staged dataset: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"KL coef: {args.kl_coef}, Temperature: {args.temperature}")
    print("Mode collapse fix: running-mean baseline (no group variance dependency)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading policy model...")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
        "local_files_only": True,
    }

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
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
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Frozen reference model for KL constraint
    print("Loading frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True,
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

    dataset = load_staged_dataset(
        args.data_path, args.episodes, games=train_games
    )
    print(f"Dataset: {len(dataset)} prompts")

    # Format with chat template
    formatted_queries = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        formatted_queries.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )


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

        if step % args.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint to {ckpt}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"REINFORCE training complete. Model saved to {args.output_dir}")
