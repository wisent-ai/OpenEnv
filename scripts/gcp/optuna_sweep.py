"""Optuna hyperparameter sweep for KantBench GRPO training.

Runs on GCP — each trial spawns a spot A100, trains for 100 steps,
evaluates step-100 metrics as proxy, and reports back.

Usage:
    python scripts/gcp/optuna_sweep.py --n-trials 20 --n-parallel 3
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import uuid

import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCP_PROJECT = "wisent-480400"
GCS_BUCKET = "gs://kantbench-training"
SA_EMAIL = f"kantbench-training@{GCP_PROJECT}.iam.gserviceaccount.com"

# Zones to try for spot A100
ZONES = ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f", "us-west1-b"]


def run_cmd(cmd: str, timeout: int = 300) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def create_trial_instance(trial_id: str, params: dict) -> str | None:
    """Create a spot A100 instance for a single trial. Returns zone or None."""
    import tempfile

    startup = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"export GCS_BUCKET={GCS_BUCKET}\n"
        f"export GCP_PROJECT={GCP_PROJECT}\n"
        "\n"
        "WORK_DIR=/workspace/wisent-openenv\n"
        "mkdir -p /workspace/output /workspace/eval-results $WORK_DIR\n"
        "\n"
        f"gsutil cp {GCS_BUCKET}/code/wisent-openenv.tar.gz /tmp/wisent-openenv.tar.gz\n"
        "tar xzf /tmp/wisent-openenv.tar.gz -C $WORK_DIR\n"
        "cd $WORK_DIR\n"
        "\n"
        "pip install --upgrade pip setuptools wheel -q\n"
        'pip install ".[train]" "jinja2>=3.1.0" -q\n'
        "\n"
        f"export HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token --project={GCP_PROJECT})\n"
        f"export WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project={GCP_PROJECT})\n"
        "huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true\n"
        "\n"
        "export WANDB_PROJECT=kantbench-optuna\n"
        "\n"
        "python3 -m train.train \\\n"
        "    --model meta-llama/Llama-3.2-1B-Instruct \\\n"
        "    --output-dir /workspace/output \\\n"
        "    --episodes 500 \\\n"
        f"    --num-generations {params['num_gen']} \\\n"
        "    --batch-size 4 \\\n"
        "    --grad-accum 4 \\\n"
        f"    --lr {params['lr']} \\\n"
        "    --max-steps 100 \\\n"
        "    --save-steps 50 \\\n"
        f"    --temperature {params['temperature']} \\\n"
        "    --use-train-split \\\n"
        "    --report-to wandb \\\n"
        f"    --wandb-run-name optuna-{trial_id} \\\n"
        f"    --kl-beta {params['kl_beta']} \\\n"
        "    2>&1 | tee /workspace/train.log\n"
        "\n"
        "# Extract metrics\n"
        "python3 /workspace/wisent-openenv/scripts/gcp/extract_trial_metrics.py\n"
        "\n"
        f"gsutil cp /workspace/trial_result.json {GCS_BUCKET}/optuna/{trial_id}/result.json\n"
        "\n"
        "shutdown -h now\n"
    )

    instance_name = f"optuna-{trial_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(startup)
        startup_file = f.name

    for zone in ZONES:
        cmd = (
            f"gcloud compute instances create {instance_name} "
            f"--project={GCP_PROJECT} --zone={zone} "
            f"--machine-type=a2-highgpu-1g "
            f"--accelerator=type=nvidia-tesla-a100,count=1 "
            f"--provisioning-model=SPOT "
            f"--instance-termination-action=STOP "
            f"--maintenance-policy=TERMINATE "
            f"--boot-disk-size=200GB --boot-disk-type=pd-ssd "
            f"--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 "
            f"--image-project=deeplearning-platform-release "
            f"--scopes=storage-full,cloud-platform "
            f"--service-account={SA_EMAIL} "
            f"--metadata-from-file=startup-script={startup_file} "
            f"2>&1"
        )
        result = run_cmd(cmd, timeout=300)
        if "Created" in result or "RUNNING" in result:
            logger.info(f"Trial {trial_id} created in {zone}")
            import os
            os.unlink(startup_file)
            return zone
        logger.debug(f"Failed in {zone}: {result[:200]}")

    import os
    os.unlink(startup_file)
    logger.error(f"Could not create instance for trial {trial_id}")
    return None


def wait_for_trial(trial_id: str, timeout_min: int = 30) -> dict | None:
    """Poll GCS for trial result."""
    result_path = f"{GCS_BUCKET}/optuna/{trial_id}/result.json"
    deadline = time.time() + timeout_min * 60

    while time.time() < deadline:
        try:
            output = run_cmd(f"gsutil cat {result_path} 2>/dev/null", timeout=15)
            if output and "{" in output:
                return json.loads(output)
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        time.sleep(30)

    return None


def cleanup_trial(trial_id: str):
    """Delete the trial instance."""
    instance_name = f"optuna-{trial_id}"
    for zone in ZONES:
        run_cmd(
            f"gcloud compute instances delete {instance_name} "
            f"--zone={zone} --project={GCP_PROJECT} --quiet 2>/dev/null",
            timeout=60,
        )


def objective(trial: optuna.Trial) -> float:
    """Single Optuna trial: train 100 steps, return composite score."""
    trial_id = f"t{trial.number:03d}-{uuid.uuid4().hex[:6]}"

    params = {
        "lr": trial.suggest_float("lr", 1e-6, 2e-5, log=True),
        "kl_beta": trial.suggest_float("kl_beta", 0.02, 0.5, log=True),
        "temperature": trial.suggest_float("temperature", 0.8, 1.5),
        "num_gen": trial.suggest_categorical("num_gen", [8, 16]),
    }

    logger.info(f"Trial {trial_id}: {params}")

    zone = create_trial_instance(trial_id, params)
    if zone is None:
        raise optuna.TrialPruned("Could not create instance")

    try:
        result = wait_for_trial(trial_id, timeout_min=25)
    finally:
        cleanup_trial(trial_id)

    if result is None or "error" in result:
        raise optuna.TrialPruned("Trial did not produce results")

    # Composite score: we want high reward_fn, low zero_std, positive reward_trend
    reward_fn = result.get("reward_fn", 0)
    zero_std = result.get("zero_std", 1)
    reward_trend = result.get("reward_trend", 0)
    entropy = result.get("entropy", 0)

    # Score: reward quality + learning signal + not collapsed + improving
    score = (
        reward_fn * 0.3                          # absolute reward quality
        + (1 - zero_std) * 0.3                   # has gradient signal
        + max(0, reward_trend) * 0.2             # improving over time
        + min(entropy / 3.0, 1.0) * 0.2          # maintaining diversity
    )

    logger.info(
        f"Trial {trial_id}: score={score:.4f} "
        f"(rfn={reward_fn:.3f} zero_std={zero_std:.3f} "
        f"trend={reward_trend:+.3f} entropy={entropy:.3f})"
    )

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-parallel", type=int, default=3)
    parser.add_argument("--study-name", default="kantbench-grpo-v1")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///optuna_{args.study_name}.db",
        load_if_exists=True,
    )

    # Run sequentially for now (parallel requires joblib + careful instance management)
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best Trial ===")
    print(f"Score: {study.best_value:.4f}")
    print(f"Params: {study.best_params}")

    # Save best params
    best = {"score": study.best_value, "params": study.best_params}
    with open("optuna_best.json", "w") as f:
        json.dump(best, f, indent=2)
    run_cmd(f"gsutil cp optuna_best.json {GCS_BUCKET}/optuna/best.json")

    print(f"\nResults saved to {GCS_BUCKET}/optuna/best.json")


if __name__ == "__main__":
    main()
