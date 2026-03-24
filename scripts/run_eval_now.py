"""Run KantBench eval on the trained model in /workspace/kantbench-output."""
import json
import logging
import os
import torch

logging.basicConfig(level=logging.INFO)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/kantbench-output"
EVAL_OUTPUT = "/workspace/eval-results/metrics.json"

os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Running KantBench tournament on held-out eval games...")
from train.grpo.trainer import KantGRPOTrainer
from train.grpo.config import GRPOConfig
from train.splits import get_train_eval_split
from common.games import GAMES

_, eval_games = get_train_eval_split()
available = set(GAMES.keys())
playable_eval = sorted(eval_games & available)
skipped = sorted(eval_games - available)
print(f"Eval games: {len(playable_eval)} playable, {len(skipped)} skipped (N-player/coalition)")

config = GRPOConfig(model_name="meta-llama/Llama-3.2-1B-Instruct")
trainer = KantGRPOTrainer(config=config, model=model, tokenizer=tokenizer)
metrics = trainer.evaluate(games=playable_eval, run_external=False)

print()
print("=== KantBench Eval Results ===")
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

with open(EVAL_OUTPUT, "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\nSaved to {EVAL_OUTPUT}")
