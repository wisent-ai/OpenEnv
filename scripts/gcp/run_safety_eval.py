"""Safety benchmark evaluation: REINFORCE 1B vs baseline.

Runs HarmBench, ETHICS, XSTest, MT-Bench on both models and saves
comparison results to /workspace/safety-eval-results/.

Usage (on VM):
    python3 scripts/gcp/run_safety_eval.py
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_GCS = "gs://kantbench-training/checkpoints/64408bdbe416"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LOCAL_CHECKPOINT = "/workspace/reinforce-checkpoint"
RESULTS_DIR = "/workspace/safety-eval-results"

BENCHMARKS = ["xstest", "ethics", "harmbench", "truthfulqa"]
# mtbench excluded: requires API judge (Vertex AI unavailable, no OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOCAL_CHECKPOINT, exist_ok=True)

    # Ensure HF_TOKEN is set for gated datasets
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN set.", flush=True)
    else:
        print("WARNING: HF_TOKEN not set — gated datasets may fail.", flush=True)

    ckpt_marker = os.path.join(LOCAL_CHECKPOINT, "checkpoint-500", "config.json")
    if os.path.exists(ckpt_marker):
        print("Checkpoint already downloaded — skipping GCS sync.", flush=True)
    else:
        print("=== Downloading REINFORCE checkpoint from GCS ===", flush=True)
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", CHECKPOINT_GCS, LOCAL_CHECKPOINT],
            check=True,
        )

    # Use checkpoint-500 (final training step) — the root dir may contain
    # artifacts from a different run.
    import glob
    ckpts = sorted(
        glob.glob(os.path.join(LOCAL_CHECKPOINT, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    if ckpts:
        model_path = ckpts[-1]
        print(f"Using checkpoint: {model_path}", flush=True)
    elif os.path.exists(os.path.join(LOCAL_CHECKPOINT, "config.json")):
        model_path = LOCAL_CHECKPOINT
        print(f"Using root checkpoint: {model_path}", flush=True)
    else:
        raise RuntimeError(f"No HuggingFace model found in {LOCAL_CHECKPOINT}")

    # Verify it's the right architecture
    import json
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", ["?"])[0]
        print(f"Model architecture: {arch}", flush=True)

    return model_path


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _free_gpu():
    """Release cached GPU memory between runs."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_benchmark(bench_name: str, model_path: str, label: str) -> dict:
    """Run a single benchmark and return result dict."""
    sys.path.insert(0, "/workspace/wisent-openenv")

    from bench.external._model_handle import ModelHandle
    from bench.external.runner import ExternalBenchmarkRunner

    print(f"\n--- {bench_name.upper()} | {label} ---", flush=True)
    t0 = time.time()

    handle = ModelHandle(model_name_or_path=model_path, max_new_tokens=128)
    runner = ExternalBenchmarkRunner(model_handle=handle, benchmarks=[bench_name])
    results = runner.run_all()
    result = results.get(bench_name)

    elapsed = time.time() - t0

    if result is None:
        print(f"  No result returned.", flush=True)
        return {"error": "no result", "elapsed_seconds": elapsed}

    if result.error:
        print(f"  ERROR: {result.error}", flush=True)
        return {"error": result.error, "elapsed_seconds": elapsed}

    print(f"  scores: {result.scores}", flush=True)
    print(f"  primary ({result.primary_metric}): {result.scores.get(result.primary_metric)}", flush=True)
    print(f"  elapsed: {elapsed:.0f}s", flush=True)

    # Free model from memory before next run
    del handle
    _free_gpu()

    return {
        "scores": result.scores,
        "primary_metric": result.primary_metric,
        "primary_score": result.scores.get(result.primary_metric),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trained_path = setup()

    models = [
        ("baseline_1b", BASE_MODEL),
        ("reinforce_1b", trained_path),
    ]

    all_results: dict = {}

    for label, model_path in models:
        all_results[label] = {}
        for bench in BENCHMARKS:
            partial_path = os.path.join(RESULTS_DIR, f"{label}_{bench}.json")

            # Skip if already succeeded in a previous run
            if os.path.exists(partial_path):
                try:
                    with open(partial_path) as f:
                        existing = json.load(f)
                    if existing.get("scores"):
                        print(f"  Skipping {bench} for {label} (already have scores)", flush=True)
                        all_results[label][bench] = existing
                        continue
                except Exception:
                    pass

            result = run_benchmark(bench, model_path, label)
            all_results[label][bench] = result

            # Save partial results after each benchmark
            with open(partial_path, "w") as f:
                json.dump(result, f, indent=2)
            _free_gpu()

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    comparison: dict = {"benchmarks": {}}
    for bench in BENCHMARKS:
        base = all_results.get("baseline_1b", {}).get(bench, {})
        trained = all_results.get("reinforce_1b", {}).get(bench, {})
        comparison["benchmarks"][bench] = {
            "primary_metric": trained.get("primary_metric") or base.get("primary_metric"),
            "baseline_score": base.get("primary_score"),
            "reinforce_score": trained.get("primary_score"),
            "baseline_scores": base.get("scores", {}),
            "reinforce_scores": trained.get("scores", {}),
        }

    comp_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    full_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print("\n\n=== SAFETY BENCHMARK RESULTS ===", flush=True)
    print(f"{'Benchmark':<20} {'Metric':<30} {'Baseline':<12} {'REINFORCE 1B':<14} {'Delta'}", flush=True)
    print("-" * 90, flush=True)
    for bench, data in comparison["benchmarks"].items():
        metric = data["primary_metric"] or "?"
        b = data["baseline_score"]
        r = data["reinforce_score"]
        if b is not None and r is not None:
            delta = f"{(r - b):+.4f}"
        else:
            delta = "N/A"
        b_str = f"{b:.4f}" if b is not None else "ERR"
        r_str = f"{r:.4f}" if r is not None else "ERR"
        print(f"{bench:<20} {metric:<30} {b_str:<12} {r_str:<14} {delta}", flush=True)

    print(f"\nAll results saved to {RESULTS_DIR}/", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
