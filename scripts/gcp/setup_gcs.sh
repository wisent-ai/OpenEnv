#!/usr/bin/env bash
# One-time setup: GCS bucket, service account, and secrets for KantBench training.
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   bash scripts/gcp/setup_gcs.sh
#
# Prerequisites:
#   - gcloud CLI authenticated with project owner/editor permissions
#   - Billing enabled on the project

set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project ID}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCS_BUCKET:-gs://kantbench-training}"
SA_NAME="kantbench-training"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"

echo "=== KantBench GCS Setup ==="
echo "Project:  $GCP_PROJECT"
echo "Region:   $REGION"
echo "Bucket:   $BUCKET"
echo ""

gcloud config set project "$GCP_PROJECT"

# --- GCS Bucket ---
echo "Creating GCS bucket..."
if gcloud storage buckets describe "$BUCKET" &>/dev/null; then
    echo "  Bucket $BUCKET already exists, skipping."
else
    gcloud storage buckets create "$BUCKET" \
        --location="$REGION" \
        --uniform-bucket-level-access \
        --public-access-prevention
    echo "  Created $BUCKET"
fi

# --- Service Account ---
echo "Creating service account..."
if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null 2>&1; then
    echo "  Service account $SA_EMAIL already exists, skipping."
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="KantBench Training"
    echo "  Created $SA_EMAIL"
fi

# Grant storage access to the bucket
echo "Granting storage access..."
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet

# Grant Secret Manager access
echo "Granting Secret Manager access..."
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

# --- Secrets ---
echo ""
echo "Creating secrets (you will be prompted for values if they don't exist)..."

create_secret() {
    local name="$1"
    local description="$2"
    if gcloud secrets describe "$name" &>/dev/null 2>&1; then
        echo "  Secret '$name' already exists, skipping."
    else
        echo "  Enter value for $name ($description):"
        read -rs secret_value
        echo -n "$secret_value" | gcloud secrets create "$name" \
            --data-file=- \
            --replication-policy=automatic
        echo "  Created secret '$name'"
    fi
}

create_secret "wandb-api-key" "Weights & Biases API key"
create_secret "hf-token" "HuggingFace access token (for gated models)"
create_secret "openai-api-key" "OpenAI API key (for MT-Bench judge)"

# --- Budget Alert ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Bucket:           $BUCKET"
echo "Service Account:  $SA_EMAIL"
echo "Secrets:          wandb-api-key, hf-token, openai-api-key"
echo ""
echo "REMINDER: Set a budget alert at \$20 on your GCP billing account."
echo "  https://console.cloud.google.com/billing"
