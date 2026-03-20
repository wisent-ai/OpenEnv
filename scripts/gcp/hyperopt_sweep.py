"""Hyperopt sweep for KantBench GRPO training.

Uses unbounded distributions centered on best-known params from Optuna.
Runs each trial as a GCP instance (A100 spot or on-demand).

Usage:
    python scripts/gcp/hyperopt_sweep.py --max-evals 40
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
import time
import uuid

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCP_PROJECT = "wisent-480400"
GCS_BUCKET = "gs://kantbench-training"
SA_EMAIL = f"kantbench-training@{GCP_PROJECT}.iam.gserviceaccount.com"

# GPU config — H100 preemptible (4x quota in us-central1 and us-west1)
MACHINE_TYPE = "a3-highgpu-1g"
ACCELERATOR = "type=nvidia-h100-80gb,count=1"
PROVISIONING = "SPOT"

ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c",
    "us-west1-b",
]

# Search space — unbounded distributions centered on Optuna's best (T4)
SPACE = {
    # T4 best: lr=2e-5. lognormal centered there, can explore 1e-6 to 1e-3
    "lr": hp.lognormal("lr", math.log(2e-5), 0.8),
    # T4 best: beta=0.4. lognormal centered there, can explore 0.01 to 5.0
    "kl_beta": hp.lognormal("kl_beta", math.log(0.4), 0.8),
    # T4 best: temp=1.5. normal centered there, can go 0.5 to 3.0
    "temperature": hp.normal("temperature", 1.5, 0.4),
    # Fixed at 8 — clear winner from Optuna
    "num_gen": 8,
}


def run_cmd(cmd: str, timeout: int = 300) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def create_instance(trial_id: str, params: dict) -> str | None:
    """Create a GCP instance for one trial. Returns zone or None."""
    # Clamp temperature to reasonable range
    temp = max(0.5, min(3.0, params["temperature"]))

    startup_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export GCS_BUCKET={GCS_BUCKET}",
        f"export GCP_PROJECT={GCP_PROJECT}",
        "",
        "WORK_DIR=/workspace/wisent-openenv",
        "mkdir -p /workspace/output /workspace/eval-results $WORK_DIR",
        "",
        f"gsutil cp {GCS_BUCKET}/code/wisent-openenv.tar.gz /tmp/wisent-openenv.tar.gz",
        "tar xzf /tmp/wisent-openenv.tar.gz -C $WORK_DIR",
        "cd $WORK_DIR",
        "",
        "pip install --upgrade pip setuptools wheel -q",
        'pip install ".[train]" "jinja2>=3.1.0" -q',
        "",
        f"export HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token --project={GCP_PROJECT})",
        f"export WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project={GCP_PROJECT})",
        "huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true",
        "",
        "export WANDB_PROJECT=kantbench-hyperopt",
        "",
        "python3 -m train.train \\",
        "    --model meta-llama/Llama-3.2-1B-Instruct \\",
        "    --output-dir /workspace/output \\",
        "    --episodes 500 \\",
        f"    --num-generations {params['num_gen']} \\",
        "    --batch-size 4 \\",
        "    --grad-accum 4 \\",
        f"    --lr {params['lr']:.8f} \\",
        "    --max-steps 100 \\",
        "    --save-steps 50 \\",
        f"    --temperature {temp:.4f} \\",
        "    --use-train-split \\",
        "    --report-to wandb \\",
        f"    --wandb-run-name hyperopt-{trial_id} \\",
        f"    --kl-beta {params['kl_beta']:.8f} \\",
        "    2>&1 | tee /workspace/train.log",
        "",
        "python3 /workspace/wisent-openenv/scripts/gcp/extract_trial_metrics.py",
        "",
        f"gsutil cp /workspace/trial_result.json {GCS_BUCKET}/hyperopt/{trial_id}/result.json",
        "",
        "shutdown -h now",
    ]

    startup = "\n".join(startup_lines)
    instance_name = f"hopt-{trial_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(startup)
        startup_file = f.name

    try:
        for zone in ZONES:
            prov_flag = f"--provisioning-model={PROVISIONING} " if PROVISIONING else ""
            cmd = (
                f"gcloud compute instances create {instance_name} "
                f"--project={GCP_PROJECT} --zone={zone} "
                f"--machine-type={MACHINE_TYPE} "
                f"--accelerator={ACCELERATOR} "
                f"{prov_flag}"
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
                return zone
            logger.debug(f"Failed in {zone}: {result[:200]}")
    finally:
        os.unlink(startup_file)

    logger.error(f"Could not create instance for trial {trial_id}")
    return None


def wait_for_result(trial_id: str, timeout_min: int = 30) -> dict | None:
    """Poll GCS for trial result."""
    result_path = f"{GCS_BUCKET}/hyperopt/{trial_id}/result.json"
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


def cleanup_instance(trial_id: str):
    """Delete the trial instance."""
    instance_name = f"hopt-{trial_id}"
    for zone in ZONES:
        run_cmd(
            f"gcloud compute instances delete {instance_name} "
            f"--zone={zone} --project={GCP_PROJECT} --quiet 2>/dev/null",
            timeout=60,
        )


def compute_score(result: dict) -> float:
    """Composite score from trial metrics."""
    reward_fn = result.get("reward_fn", 0)
    zero_std = result.get("zero_std", 1)
    reward_trend = result.get("reward_trend", 0)
    entropy = result.get("entropy", 0)

    score = (
        reward_fn * 0.3
        + (1 - zero_std) * 0.3
        + max(0, reward_trend) * 0.2
        + min(entropy / 3.0, 1.0) * 0.2
    )
    return score


def objective(params: dict) -> dict:
    """Single Hyperopt trial."""
    trial_id = f"{uuid.uuid4().hex[:8]}"

    logger.info(
        f"Trial {trial_id}: lr={params['lr']:.2e} "
        f"beta={params['kl_beta']:.4f} temp={params['temperature']:.3f}"
    )

    zone = create_instance(trial_id, params)
    if zone is None:
        return {"loss": 1.0, "status": STATUS_FAIL}

    try:
        result = wait_for_result(trial_id, timeout_min=30)
    finally:
        cleanup_instance(trial_id)

    if result is None or "error" in result:
        return {"loss": 1.0, "status": STATUS_FAIL}

    score = compute_score(result)

    logger.info(
        f"Trial {trial_id}: score={score:.4f} "
        f"(rfn={result.get('reward_fn', 0):.3f} "
        f"zero_std={result.get('zero_std', 0):.3f} "
        f"trend={result.get('reward_trend', 0):+.3f} "
        f"entropy={result.get('entropy', 0):.3f})"
    )

    # Hyperopt minimizes, so negate score
    return {
        "loss": -score,
        "status": STATUS_OK,
        "score": score,
        "result": result,
        "params": {k: v for k, v in params.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=40)
    args = parser.parse_args()

    trials = Trials()

    best = fmin(
        fn=objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
    )

    print("\n=== Hyperopt Sweep Complete ===")
    print(f"Best params: {best}")

    # Find the best trial result
    best_trial = min(trials.results, key=lambda r: r.get("loss", 1.0))
    print(f"Best score: {best_trial.get('score', 'N/A')}")
    print(f"Best result: {json.dumps(best_trial.get('result', {}), indent=2)}")

    # Save
    output = {
        "best_params": best,
        "best_score": best_trial.get("score"),
        "best_result": best_trial.get("result"),
        "all_trials": [
            {
                "params": r.get("params"),
                "score": r.get("score"),
                "result": r.get("result"),
            }
            for r in trials.results
            if r.get("status") == STATUS_OK
        ],
    }
    with open("hyperopt_results.json", "w") as f:
        json.dump(output, f, indent=2)
    run_cmd(f"gsutil cp hyperopt_results.json {GCS_BUCKET}/hyperopt/best.json")
    print(f"\nSaved to hyperopt_results.json and {GCS_BUCKET}/hyperopt/best.json")


if __name__ == "__main__":
    main()
