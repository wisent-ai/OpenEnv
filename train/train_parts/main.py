"""Parts of train.py, split by the tama size splitter; train.py imports every name back."""

from __future__ import annotations
import logging
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common.games import GAMES
from train.rewards_to_trajectory.splits import get_train_eval_split
from train_parts.system_prompt import SYSTEM_PROMPT, format_reward_fn, load_staged_dataset, require_local_artifact, require_stado_workload
from train_parts.parse_args import parse_args
from train_parts.make_reward_fn import make_reward_fn


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    require_stado_workload()

    model_path = require_local_artifact(
        args.model_path, "training model", directory=True
    )
    print(f"Loading staged model: {model_path}")
    print(f"Staged dataset: {args.data_path}")
    print(f"Output: {args.output_dir}")


    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model loading ---
    # Always pre-load so the reward function can use the model for
    # interactive episode play (generating actions round-by-round).
    peft_config = None
    # Pick dtype to match the GRPOConfig precision flags below.
    # bf16 needs compute capability >= 8 (A100/H100); T4/V100 are fp16 only.
    # Loading bf16 weights on a T4 then using fp16 GradScaler causes
    # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    # BFloat16" because the scaler unscales bf16 grads it cannot handle.
    # Use bf16 unconditionally on GPU. bf16 has the same exponent range
    # as fp32 so no GradScaler is needed (no AMP scaler.unscale_ calls
    # that error out on fp16 grads vs fp16 master, or on bf16 grads vs
    # the cuda kernel that does not implement bf16 unscale). On T4 (cap
    # 7.5) bf16 is software-emulated and slightly slower, but the run
    # completes correctly. For the dynamics study which only runs 100
    # GRPO steps this is the right trade — slower steps, no crashes.
    if torch.cuda.is_available():
        _train_dtype = torch.bfloat16
    else:
        _train_dtype = torch.float32
    load_kwargs = {
        "torch_dtype": _train_dtype,
        "device_map": "auto",
        "local_files_only": True,
    }

    if args.quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_train_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading with 4-bit quantization")

    # Use eager attention for compatibility with batched left-padded generation
    load_kwargs["attn_implementation"] = "eager"
    model_or_path = AutoModelForCausalLM.from_pretrained(
        model_path, **load_kwargs
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
    if args.games:
        train_games = [g.strip() for g in args.games.split(",") if g.strip()]
        unknown = [g for g in train_games if g not in GAMES]
        if unknown:
            raise SystemExit(f"--games unknown keys: {unknown}")
        print(f"Restricting training to {len(train_games)} game(s): {train_games}")
    if args.use_train_split:
        train_set, eval_set = get_train_eval_split()
        train_games = sorted(train_set)
        print(f"Using stratified split: {len(train_games)} train, {len(eval_set)} eval games")

    dataset = load_staged_dataset(
        args.data_path,
        args.episodes,
        games=train_games,
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
    reward_fn = make_reward_fn(model=reward_model, tokenizer=tokenizer)

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
        bf16=torch.cuda.is_available(),
        # 8-bit AdamW (bitsandbytes) instead of fp32 AdamW saves ~10 GB
        # of optimizer state for a 1B model — required to fit GRPO on a
        # 15 GB T4. Falls back to standard adamw_torch on CPU (where the
        # bnb backend would not work).
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        fp16=False,
        report_to=args.report_to,
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
    # Resolve a local resume target to the newest complete checkpoint.
    from train.rewards_to_trajectory.splits import resolve_resume_checkpoint as _resolve_resume
    resume_ckpt = _resolve_resume(args.resume_from_checkpoint, args.output_dir)

    print("Starting GRPO training...")
    print(f"  Reward: composite (payoff + cooperation + Pareto + fairness)")
    print("  Episode: full multi-round rollout in the local environment")
    if resume_ckpt:
        print(f"  Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")
