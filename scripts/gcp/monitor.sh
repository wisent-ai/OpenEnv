#!/usr/bin/env bash
# Local monitoring script for KantBench training on GCP.
#
# Usage:
#   bash scripts/gcp/monitor.sh [--zone ZONE] [command]
#
# Commands:
#   status    - Show VM status and latest checkpoint (default)
#   logs      - Stream training logs via SSH
#   restart   - Restart a stopped VM (after preemption)
#   delete    - Delete the VM and clean up

set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project ID}"

INSTANCE="${INSTANCE_NAME:-kantbench-train}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET="${GCS_BUCKET:-gs://kantbench-training}"

# Parse --zone flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --zone) ZONE="$2"; shift 2 ;;
        --zone=*) ZONE="${1#*=}"; shift ;;
        *) break ;;
    esac
done

COMMAND="${1:-status}"

case "$COMMAND" in
    status)
        echo "=== VM Status ==="
        VM_STATUS=$(gcloud compute instances describe "$INSTANCE" \
            --zone="$ZONE" --project="$GCP_PROJECT" \
            --format='value(status)' 2>/dev/null || echo "NOT_FOUND")
        echo "Instance: $INSTANCE ($ZONE)"
        echo "Status:   $VM_STATUS"
        echo ""

        # Find run ID from latest checkpoint
        echo "=== Latest Checkpoints ==="
        gsutil ls "${BUCKET}/checkpoints/" 2>/dev/null | tail -5 || echo "  (none)"
        echo ""

        # Check for completion
        RUNS=$(gsutil ls "${BUCKET}/checkpoints/" 2>/dev/null | head -5)
        for RUN_DIR in $RUNS; do
            RUN_ID=$(basename "$RUN_DIR")
            if gsutil -q stat "${RUN_DIR}TRAINING_COMPLETE" 2>/dev/null; then
                echo "  [DONE] $RUN_ID"
            else
                echo "  [RUNNING] $RUN_ID"
                echo "  Checkpoints:"
                gsutil ls "${RUN_DIR}" 2>/dev/null | grep "checkpoint-" | tail -3 || echo "    (none yet)"
            fi
        done
        ;;

    logs)
        echo "Streaming logs from $INSTANCE..."
        gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$GCP_PROJECT" \
            -- journalctl -u kantbench-train -f
        ;;

    restart)
        echo "Restarting $INSTANCE in $ZONE..."
        gcloud compute instances start "$INSTANCE" \
            --zone="$ZONE" --project="$GCP_PROJECT"
        echo "Instance starting. The startup script will resume training automatically."
        ;;

    delete)
        echo "WARNING: This will delete $INSTANCE and its boot disk."
        echo "Checkpoints in GCS ($BUCKET) will be preserved."
        read -rp "Continue? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            gcloud compute instances delete "$INSTANCE" \
                --zone="$ZONE" --project="$GCP_PROJECT" --quiet
            echo "Instance deleted."
        else
            echo "Cancelled."
        fi
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: $0 [--zone ZONE] {status|logs|restart|delete}"
        exit 1
        ;;
esac
