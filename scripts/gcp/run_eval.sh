#!/usr/bin/env bash
# Post-training evaluation: KantBench tournament + external safety benchmarks.
#
# Expects the trained model in LOCAL_CKPT_DIR and secrets in /opt/kantbench/secrets.env.

set -euo pipefail

KANTBENCH_DIR="/opt/kantbench"
WORK_DIR="/workspace/wisent-openenv"
LOCAL_CKPT_DIR="/workspace/kantbench-output"
EVAL_OUTPUT_DIR="/workspace/eval-results"

source "${KANTBENCH_DIR}/current_model.env"
source "${KANTBENCH_DIR}/secrets.env"

export WANDB_API_KEY
export HF_TOKEN
export OPENAI_API_KEY
export WANDB_PROJECT="kantbench-grpo"
export WANDB_RESUME="allow"
export WANDB_RUN_ID="$(cat "${KANTBENCH_DIR}/run_id")"

GCS_BUCKET="${GCS_BUCKET:-gs://kantbench-training}"
GCS_EVAL_DIR="${GCS_BUCKET}/eval/${WANDB_RUN_ID}"

mkdir -p "$EVAL_OUTPUT_DIR"
cd "$WORK_DIR"

echo "=== KantBench Evaluation ($(date)) ==="
echo "Model:   $LOCAL_CKPT_DIR"
echo "Run ID:  $WANDB_RUN_ID"

# --- Stage 1: KantBench tournament on held-out games + external benchmarks ---
python3 -c "
import json
import logging
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

model_path = '${LOCAL_CKPT_DIR}'
use_lora = '${USE_LORA:-false}' == 'true'
base_model_name = '${MODEL_NAME}'

print('Loading model...')
if use_lora:
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(base, model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Running KantBench tournament on held-out games...')
from train.grpo.trainer import KantGRPOTrainer
from train.grpo.config import GRPOConfig

config = GRPOConfig(model_name=base_model_name)
trainer = KantGRPOTrainer(config=config, model=model, tokenizer=tokenizer)
metrics = trainer.evaluate(run_external=True)

print()
print('=== Results ===')
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')

# Save to file
with open('${EVAL_OUTPUT_DIR}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

# Log to wandb
wandb.init(
    project='kantbench-grpo',
    id='${WANDB_RUN_ID}',
    resume='allow',
)
wandb.log({f'eval/{k}': v for k, v in metrics.items() if isinstance(v, (int, float))})
wandb.finish()

print('Evaluation complete.')
"

# --- Upload results to GCS ---
echo "Uploading results to GCS..."
gsutil -m cp -r "$EVAL_OUTPUT_DIR" "$GCS_EVAL_DIR"

echo ""
echo "=== Evaluation Complete ==="
echo "Results:  $GCS_EVAL_DIR"
echo "wandb:    https://wandb.ai (project: kantbench-grpo, run: $WANDB_RUN_ID)"
