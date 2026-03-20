#!/usr/bin/env bash
# Idempotent VM bootstrap for KantBench training.
# Runs on first boot and after preemption restart via startup-script metadata.
#
# Expects:
#   GCS_BUCKET  - GCS bucket URL (e.g., gs://kantbench-training)
#   GCP_PROJECT - GCP project ID

set -euo pipefail

: "${GCS_BUCKET:?GCS_BUCKET not set}"
: "${GCP_PROJECT:?GCP_PROJECT not set}"

WORK_DIR="/workspace/wisent-openenv"
KANTBENCH_DIR="/opt/kantbench"

mkdir -p "$KANTBENCH_DIR"

echo "=== KantBench VM Setup ($(date)) ==="

# --- Fetch project from GCS ---
if [[ -d "$WORK_DIR/train" ]]; then
    echo "Project already extracted, skipping..."
else
    echo "Downloading project from GCS..."
    mkdir -p "$WORK_DIR"
    gsutil cp "${GCS_BUCKET}/code/wisent-openenv.tar.gz" /tmp/wisent-openenv.tar.gz
    tar xzf /tmp/wisent-openenv.tar.gz -C "$WORK_DIR"
    rm /tmp/wisent-openenv.tar.gz
fi
cd "$WORK_DIR"

# --- Install Python dependencies ---
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel --quiet
pip install ".[train]" --quiet
pip install google-cloud-storage wandb "jinja2>=3.1.0" "anthropic[vertex]" --quiet

# --- Fetch secrets ---
echo "Fetching secrets from Secret Manager..."
export WANDB_API_KEY="$(gcloud secrets versions access latest --secret=wandb-api-key --project="$GCP_PROJECT")"
export HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token --project="$GCP_PROJECT")"
export OPENAI_API_KEY="$(gcloud secrets versions access latest --secret=openai-api-key --project="$GCP_PROJECT" 2>/dev/null || echo "")"

# Persist secrets for the systemd service
cat > "${KANTBENCH_DIR}/secrets.env" <<EOF
WANDB_API_KEY=${WANDB_API_KEY}
HF_TOKEN=${HF_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY}
EOF
chmod 600 "${KANTBENCH_DIR}/secrets.env"

# --- HuggingFace login for gated models ---
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# --- wandb setup ---
export WANDB_PROJECT="kantbench-grpo"
export WANDB_RESUME="allow"

# Generate or reuse a persistent run ID (survives preemption)
if [[ ! -f "${KANTBENCH_DIR}/run_id" ]]; then
    python3 -c "import uuid; print(uuid.uuid4().hex[:12])" > "${KANTBENCH_DIR}/run_id"
    echo "Generated new run ID: $(cat "${KANTBENCH_DIR}/run_id")"
else
    echo "Reusing run ID: $(cat "${KANTBENCH_DIR}/run_id")"
fi

# --- Fetch scripts and model config from GCS ---
echo "Fetching scripts from GCS..."
gsutil cp "${GCS_BUCKET}/scripts/train_with_recovery.sh" "${KANTBENCH_DIR}/train_with_recovery.sh"
gsutil cp "${GCS_BUCKET}/scripts/run_eval.sh" "${KANTBENCH_DIR}/run_eval.sh"
gsutil cp "${GCS_BUCKET}/scripts/current_model.env" "${KANTBENCH_DIR}/current_model.env"
gsutil cp "${GCS_BUCKET}/scripts/kantbench-train.service" /etc/systemd/system/kantbench-train.service

chmod +x "${KANTBENCH_DIR}/train_with_recovery.sh"
chmod +x "${KANTBENCH_DIR}/run_eval.sh"

# --- Install and start systemd service ---
echo "Setting up systemd service..."
systemctl daemon-reload
systemctl enable kantbench-train.service
systemctl restart kantbench-train.service

echo "=== Setup complete. Training service started. ==="
echo "Check status: systemctl status kantbench-train"
echo "View logs:    journalctl -u kantbench-train -f"
