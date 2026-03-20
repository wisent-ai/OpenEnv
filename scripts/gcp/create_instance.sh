#!/usr/bin/env bash
# Create a GCP spot A100 VM for KantBench GRPO training.
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   bash scripts/gcp/create_instance.sh [model_config]
#
# Examples:
#   bash scripts/gcp/create_instance.sh                    # uses llama1b.env
#   bash scripts/gcp/create_instance.sh qwen9b             # uses qwen9b.env
#   bash scripts/gcp/create_instance.sh model_configs/qwen9b.env

set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_NAME="${INSTANCE_NAME:-kantbench-train}"
SA_EMAIL="kantbench-training@${GCP_PROJECT}.iam.gserviceaccount.com"
BUCKET="${GCS_BUCKET:-gs://kantbench-training}"

# Resolve model config
MODEL_ARG="${1:-llama1b}"
if [[ -f "$MODEL_ARG" ]]; then
    MODEL_CONFIG="$MODEL_ARG"
elif [[ -f "${SCRIPT_DIR}/model_configs/${MODEL_ARG}.env" ]]; then
    MODEL_CONFIG="${SCRIPT_DIR}/model_configs/${MODEL_ARG}.env"
else
    echo "Error: Cannot find model config '$MODEL_ARG'"
    echo "Available configs:"
    ls "${SCRIPT_DIR}/model_configs/"*.env 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "=== Creating KantBench Training Instance ==="
echo "Instance:  $INSTANCE_NAME"
echo "Config:    $MODEL_CONFIG"
echo ""

# Try zones in order of typical A100 spot availability
ZONES=("us-central1-a" "us-central1-c" "us-east4-c" "us-west1-b")

# Upload setup script and model config to GCS for the startup script to fetch
gsutil cp "${SCRIPT_DIR}/setup.sh" "${BUCKET}/scripts/setup.sh"
gsutil cp "${SCRIPT_DIR}/train_with_recovery.sh" "${BUCKET}/scripts/train_with_recovery.sh"
gsutil cp "${SCRIPT_DIR}/run_eval.sh" "${BUCKET}/scripts/run_eval.sh"
gsutil cp "${SCRIPT_DIR}/kantbench-train.service" "${BUCKET}/scripts/kantbench-train.service"
gsutil cp "$MODEL_CONFIG" "${BUCKET}/scripts/current_model.env"

# Startup script that fetches the real setup from GCS
STARTUP_SCRIPT="#!/bin/bash
set -euo pipefail
export GCS_BUCKET='${BUCKET}'
export GCP_PROJECT='${GCP_PROJECT}'
mkdir -p /opt/kantbench
gsutil cp '${BUCKET}/scripts/setup.sh' /opt/kantbench/setup.sh
chmod +x /opt/kantbench/setup.sh
bash /opt/kantbench/setup.sh
"

for ZONE in "${ZONES[@]}"; do
    echo "Trying zone: $ZONE ..."
    if gcloud compute instances create "$INSTANCE_NAME" \
        --project="$GCP_PROJECT" \
        --zone="$ZONE" \
        --machine-type=a2-highgpu-1g \
        --accelerator=type=nvidia-tesla-a100,count=1 \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --maintenance-policy=TERMINATE \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --scopes=storage-full,cloud-platform \
        --service-account="$SA_EMAIL" \
        --metadata=startup-script="$STARTUP_SCRIPT" \
        2>&1; then
        echo ""
        echo "=== Instance created in $ZONE ==="
        echo "SSH:      gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
        echo "Monitor:  bash scripts/gcp/monitor.sh --zone=$ZONE"
        echo "Stop:     gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
        echo "Delete:   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
        exit 0
    else
        echo "  Failed in $ZONE, trying next..."
    fi
done

echo "ERROR: Could not create instance in any zone. A100 spot capacity may be exhausted."
echo "Try again later or use a different region."
exit 1
