"""Parse GRPO training metrics from a wisent-compute GCS command_output.log
and emit per-step CSV.

Both in-flight free-chat-PD runs were submitted with --report-to none, so wandb
is not collecting metrics. TRL still prints the standard metrics dict to stdout
every logging_steps iterations, interleaved with the tqdm progress bar. This
script reconstructs the wandb-equivalent table by regex-extracting both signals
from the captured log.

Usage:
    python3 parse_training_metrics.py JOB_ID [JOB_ID ...] \\
        --out-dir /tmp/training_metrics

Writes <out_dir>/<job_id>.csv with one row per logged step.
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


GCS_LOG_TMPL = "gs://wisent-compute/status/{job_id}/output/command_output.log"

ADC_PATH = (
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    or f"{os.path.expanduser('~')}/.config/gcloud/legacy_credentials/"
       "droid-441@wisent-480400.iam.gserviceaccount.com/adc.json"
)


STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
DICT_RE = re.compile(r"(\{'loss':\s*[^}]+\})")
METRIC_COLUMNS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "num_tokens",
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "completions/min_terminated_length",
    "completions/max_terminated_length",
    "rewards/reward_fn/mean",
    "rewards/reward_fn/std",
    "rewards/format_reward_fn/mean",
    "rewards/format_reward_fn/std",
    "reward",
    "reward_std",
    "frac_reward_zero_std",
    "kl",
    "entropy",
    "clip_ratio/low_mean",
    "clip_ratio/low_min",
    "clip_ratio/high_mean",
    "clip_ratio/high_max",
    "clip_ratio/region_mean",
    "step_time",
    "epoch",
]


def fetch_log(job_id: str) -> bytes:
    """Stream the full command_output.log from GCS to memory."""
    uri = GCS_LOG_TMPL.format(job_id=job_id)
    env = dict(os.environ)
    env["GOOGLE_APPLICATION_CREDENTIALS"] = ADC_PATH
    result = subprocess.run(
        ["gsutil", "cat", uri],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not result.stdout:
        raise RuntimeError(
            f"empty log for job {job_id}: "
            f"stderr={result.stderr.decode(errors='replace')[:400]}"
        )
    return result.stdout


def parse_metrics(log_bytes: bytes):
    """Yield (step, total_steps, metrics_dict) for every metric line in the log.

    tqdm uses \\r to overwrite its progress bar, so the captured log is full of
    \\r-separated segments. After splitlines() the metrics dict typically lands
    on its own line (no embedded step counter), while the step counter is on
    the immediately preceding segment. We carry forward the most recent
    (step, total) seen and attach it to the next dict.
    """
    text = log_bytes.decode("utf-8", errors="replace")
    last_step = None
    last_total = None
    for line in text.splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            last_step = int(step_match.group(1))
            last_total = int(step_match.group(2))
        dict_match = DICT_RE.search(line)
        if not dict_match:
            continue
        if last_step is None:
            continue
        try:
            metrics = ast.literal_eval(dict_match.group(1))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(metrics, dict):
            continue
        yield last_step, last_total, metrics


def write_csv(rows, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total_steps", *METRIC_COLUMNS])
        for step, total, metrics in rows:
            writer.writerow(
                [step, total, *[metrics.get(col, "") for col in METRIC_COLUMNS]]
            )
            written += 1
    return written


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_ids", nargs="+", help="wisent-compute job IDs (8-char prefix)")
    p.add_argument(
        "--out-dir",
        default="/tmp/training_metrics",
        help="Output directory for per-job CSVs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    exit_code = 0
    for jid in args.job_ids:
        log = fetch_log(jid)
        rows = list(parse_metrics(log))
        csv_path = out_dir / f"{jid}.csv"
        n = write_csv(rows, csv_path)
        if rows:
            last_step, total, last_metrics = rows[-1]
            print(
                f"{jid}: wrote {n} rows -> {csv_path} | "
                f"last step={last_step}/{total} "
                f"loss={last_metrics.get('loss', '?')} "
                f"reward={last_metrics.get('reward', '?')} "
                f"kl={last_metrics.get('kl', '?')} "
                f"epoch={last_metrics.get('epoch', '?')}"
            )
        else:
            print(f"{jid}: 0 rows (no metric dicts found in log)", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
