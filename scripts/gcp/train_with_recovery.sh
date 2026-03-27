#!/usr/bin/env bash
# Preemption-resilient GRPO training wrapper.
#
# Handles:
#   - Restoring checkpoints from GCS on startup
#   - Syncing checkpoints to GCS periodically (every 60s)
#   - Graceful shutdown on SIGTERM (GCP preemption signal)
#   - Running eval after training completes
#
# Expects:
#   /opt/kantbench/current_model.env - model hyperparameters
#   /opt/kantbench/secrets.env       - API keys
#   /opt/kantbench/run_id            - persistent wandb run ID

set -euo pipefail

KANTBENCH_DIR="/opt/kantbench"
WORK_DIR="/workspace/wisent-openenv"
LOCAL_CKPT_DIR="/workspace/kantbench-output"

# --- Load config ---
source "${KANTBENCH_DIR}/current_model.env"
source "${KANTBENCH_DIR}/secrets.env"

export WANDB_API_KEY
export HF_TOKEN
export OPENAI_API_KEY
export WANDB_PROJECT="kantbench-grpo"
export WANDB_RESUME="allow"
export WANDB_RUN_ID="$(cat "${KANTBENCH_DIR}/run_id")"

GCS_BUCKET="${GCS_BUCKET:-gs://kantbench-training}"
GCS_CKPT_DIR="${GCS_BUCKET}/checkpoints/${WANDB_RUN_ID}"
GCS_EVAL_DIR="${GCS_BUCKET}/eval/${WANDB_RUN_ID}"

TRAIN_PID=""
WATCHER_PID=""

echo "=== KantBench Training ($(date)) ==="
echo "Run ID:    $WANDB_RUN_ID"
echo "Model:     $MODEL_NAME"
echo "Steps:     $MAX_STEPS"
echo "GCS:       $GCS_CKPT_DIR"

# --- Sync helper ---
sync_to_gcs() {
    echo "[sync] Uploading checkpoints to GCS..."
    # -d flag: delete GCS files not present locally (prevents restoring deleted checkpoints)
    gsutil -m -q rsync -r -d "$LOCAL_CKPT_DIR" "$GCS_CKPT_DIR" 2>/dev/null || true
    echo "[sync] Done."
}

sync_from_gcs() {
    echo "[sync] Restoring checkpoints from GCS..."
    mkdir -p "$LOCAL_CKPT_DIR"
    LATEST=$(gsutil ls "$GCS_CKPT_DIR/" 2>/dev/null | grep "checkpoint-" | sort -t- -k2 -n | tail -2 || true)
    if [ -n "$LATEST" ]; then
        for ckpt in $LATEST; do
            gsutil -m -q rsync -r "$ckpt" "$LOCAL_CKPT_DIR/$(basename $ckpt)/" 2>/dev/null || true
        done
    else
        echo "[sync] No checkpoints in GCS (fresh run)"
    fi
    echo "[sync] Contents:"
    ls -la "$LOCAL_CKPT_DIR" 2>/dev/null || echo "  (empty)"
}

# --- Preemption handler ---
cleanup() {
    echo ""
    echo "[preemption] SIGTERM received at $(date). Syncing and shutting down..."
    sync_to_gcs
    [[ -n "$WATCHER_PID" ]] && kill "$WATCHER_PID" 2>/dev/null || true
    [[ -n "$TRAIN_PID" ]] && kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
    echo "[preemption] Shutdown complete."
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- Step 1: Restore from GCS ---
sync_from_gcs

# --- Step 2: Background checkpoint watcher ---
(
    while true; do
        sleep 60
        sync_to_gcs
    done
) &
WATCHER_PID=$!

# --- Step 3: Build training command ---
cd "$WORK_DIR"

# Choose GRPO or PPO training script
if [[ "${USE_PPO:-false}" == "true" ]]; then
    TRAIN_MODULE="train.ppo_train"
else
    TRAIN_MODULE="train.train"
fi

TRAIN_CMD=(
    python3 -m "$TRAIN_MODULE"
    --model "$MODEL_NAME"
    --output-dir "$LOCAL_CKPT_DIR"
    --episodes "${EPISODES:-1000}"
    --num-generations "${NUM_GEN:-8}"
    --batch-size "${BATCH_SIZE:-4}"
    --grad-accum "${GRAD_ACCUM:-4}"
    --lr "${LR:-5e-6}"
    --max-steps "${MAX_STEPS:-500}"
    --save-steps "${SAVE_STEPS:-50}"
    --temperature "${TEMPERATURE:-0.8}"
    --use-train-split
    --resume-from-checkpoint latest
    --report-to wandb
    --wandb-run-name "kantbench-${WANDB_RUN_ID}"
)

# LoRA flags
if [[ "${USE_LORA:-false}" == "true" ]]; then
    TRAIN_CMD+=(--use-lora --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}")
fi
if [[ "${QUANTIZE_4BIT:-false}" == "true" ]]; then
    TRAIN_CMD+=(--quantize-4bit)
fi
# KL penalty (higher = fights mode collapse)
if [[ -n "${KL_BETA:-}" ]]; then
    TRAIN_CMD+=(--kl-beta "${KL_BETA}")
fi

echo ""
echo "Command: ${TRAIN_CMD[*]}"
echo ""

# --- Step 4: Run training ---
"${TRAIN_CMD[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
EXIT_CODE=$?

# --- Step 5: Post-training ---
kill "$WATCHER_PID" 2>/dev/null || true

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "=== Training complete! ==="
    sync_to_gcs

    # Write completion flag
    echo "$(date)" | gsutil cp - "${GCS_CKPT_DIR}/TRAINING_COMPLETE"

    # Run eval
    echo "Starting evaluation..."
    bash "${KANTBENCH_DIR}/run_eval.sh"
else
    echo "Training exited with code $EXIT_CODE. Syncing checkpoints..."
    sync_to_gcs
    exit $EXIT_CODE
fi
