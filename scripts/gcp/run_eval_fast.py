"""Fast KantBench eval: tournament only, no external benchmarks."""
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/kantbench-output"
base_model_name = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B-Instruct"
output_file = sys.argv[3] if len(sys.argv) > 3 else "/workspace/eval-results/reinforce_eval.json"

print(f"Loading model from {model_path}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Running KantBench tournament (no external benchmarks)...", flush=True)
from train.grpo.trainer import KantGRPOTrainer
from train.grpo.config import GRPOConfig

config = GRPOConfig(model_name=base_model_name)
trainer = KantGRPOTrainer(config=config, model=model, tokenizer=tokenizer)
metrics = trainer.evaluate(run_external=False)

print("\n=== Results ===", flush=True)
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}", flush=True)
    else:
        print(f"  {k}: {v}", flush=True)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\nSaved to {output_file}", flush=True)
