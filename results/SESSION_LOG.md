# KantBench GRPO Training Session Log

## Goal

The paper "Kant: Teaching Ethical Reasoning to Language Models via Comprehensive Game-Theoretic Training" claims that training LLMs on game theory improves AI safety benchmarks. The results section is empty. Our goal: run the experiments to fill that table.

**Central hypothesis**: GRPO training on game-theoretic environments → improved cooperation, fairness, exploitation resistance, adaptability → transfers to safety benchmarks (HarmBench, ETHICS, TruthfulQA, XSTest, MT-Bench).

---

## Infrastructure Setup

### GCP Resources Created
- **GCS bucket**: `gs://kantbench-training` (us-central1)
- **Service account**: `kantbench-training@wisent-480400.iam.gserviceaccount.com`
  - Roles: `storage.objectAdmin`, `secretmanager.secretAccessor`, `aiplatform.user`
- **Secrets in Secret Manager**: `wandb-api-key`, `hf-token`
- **Quota increases**: A100 4→5 (approved), L4 8→9 (denied), preemptible H100 already had 4

### Scripts Created (`scripts/gcp/`)
- `setup_gcs.sh` — one-time bucket + IAM + secrets setup
- `create_instance.sh` — provisions spot A100 VM with zone fallback
- `setup.sh` — idempotent VM bootstrap (pulls code from GCS, installs deps, starts systemd)
- `train_with_recovery.sh` — preemption-resilient wrapper (SIGTERM trap, GCS checkpoint sync every 60s)
- `run_eval.sh` — post-training eval (KantBench tournament + external benchmarks)
- `kantbench-train.service` — systemd unit with auto-restart
- `monitor.sh` — local monitoring utility (status, logs, restart, delete)
- `model_configs/*.env` — per-experiment hyperparameter configs
- `optuna_sweep.py` — Optuna hyperparameter sweep (spawns GCP instances per trial)
- `hyperopt_sweep.py` — Hyperopt sweep with unbounded distributions
- `extract_trial_metrics.py` — runs on trial VMs to extract checkpoint metrics

### Key Code Changes

#### `train/train.py` (major rewrites)
1. Added LoRA/QLoRA support (`--use-lora`, `--quantize-4bit`, `--lora-r`, `--lora-alpha`)
2. Added `--save-steps`, `--temperature`, `--kl-beta`, `--wandb-run-name` args
3. Added retry logic (3 attempts, exponential backoff) in reward function
4. Smart checkpoint resume (no crash when no checkpoints exist)
5. Always pre-load model (needed for interactive reward)
6. **V5**: Reward function plays 3 strategies per completion (always_defect, tit_for_tat, always_cooperate) to compute all 5 metrics
7. **V6**: Interactive episodes — model generates actions round-by-round instead of fixed-action replay. First uses remote HF Space, then switched to local `KantEnvironment` for ~100x speedup
8. **V7**: Upweighted pareto (0.25) and fairness (0.25) in reward, downweighted cooperation (0.15) and adaptability (0.15)

#### `common/games.py`
- Fixed `_matrix_payoff_fn` to return (0,0) on missing payoff pairs instead of crashing

#### `common/strategies.py`
- Fixed `_parse_amount` to handle unparseable action strings (returned 0 instead of IndexError)

#### `train/grpo/trainer.py`
- Fixed CUDA device mismatch in `evaluate()` — inputs now moved to model device

#### `bench/evaluation/tournament.py`
- Added logging import and logger
- Added try/except around episode runs (later reverted — errors should crash properly)

#### `bench/external/_model_handle.py`
- Switched `_generate_anthropic` from `anthropic.Anthropic()` to `anthropic.AnthropicVertex()` for Vertex AI

#### `bench/external/constants.py`
- Changed MT-Bench judge from GPT-5.4 to `claude-sonnet-4-6` via Vertex AI

---

## Experiments

### Baseline
- **Model**: Llama 3.2-1B-Instruct (untrained)
- **Eval**: 21 held-out games × 11 strategies × 5 episodes
- **Results**: `results/baseline_llama1b.json`
  - cooperation_rate: 0.024
  - exploitation_resistance: 0.458
  - pareto_efficiency: 0.359
  - fairness_index: 0.616
  - adaptability: 0.013
  - **strategic_reasoning: 0.294**

### V1 — First GRPO run
- **Config**: lr=5e-6, temp=0.8, kl_beta=0.04 (TRL default), num_gen=8, 500 steps
- **Reward**: Fixed-action replay, single strategy, 3 metrics (exploit_resist/adaptability hardcoded to 0.5)
- **Result**: MODE COLLAPSE. Entropy dropped 0.57→0.001 by step 100. frac_reward_zero_std hit 0.95. Model became deterministic, GRPO got no gradient signal.
- **wandb**: `kantbench-3adba57f52bc`

### V2 — Anti-collapse attempt 1
- **Changes**: temp=1.1, num_gen=16, kl_beta=0.1
- **Result**: Still collapsed by step 50. Entropy dropped to 0.017. Higher temp/gen wasn't enough.

### V3 — Anti-collapse attempt 2 (aggressive)
- **Changes**: lr=1e-6, temp=1.5, kl_beta=0.5
- **Result**: NO COLLAPSE (zero_std=0.00 through all 500 steps, entropy stable ~3.0). But reward flat — model explored without learning. KL penalty too strong.
- **Eval**: strategic_reasoning=0.274 (vs baseline 0.294). Barely moved.

### Optuna Hyperparameter Sweep
- **Setup**: 15 trials on spot A100, 100 steps each
- **Search space**: lr [1e-6, 2e-5], kl_beta [0.02, 0.5], temp [0.8, 1.5], num_gen {8, 16}
- **Best trial (T4)**: lr=2e-5, kl_beta=0.406, temp=1.49, gen=8 → score 0.622
- **Key finding**: High lr + high beta + high temp + gen=8 is optimal. Best params were at search space ceiling for lr and temp.

### Hyperopt Sweep (unbounded)
- **Setup**: 40 trials on spot H100, 100 steps each
- **Search space**: hp.lognormal centered on Optuna best (lr~2e-5, beta~0.4), hp.normal for temp (~1.5), gen fixed at 8
- **Completed**: ~20/40 trials
- **Best score**: 0.666 — scores clustered 0.63-0.67, search converged
- **Confirmed**: lr~2e-5, beta~0.4, temp~1.5, gen=8 is the sweet spot

### V4 — Optimized hyperparams
- **Config**: lr=2e-5, temp=1.5, kl_beta=0.4, gen=8, 500 steps
- **Reward**: Fixed-action replay, 3 metrics
- **Training**: Reward climbed 0.536→0.610 (+0.074 trend), no collapse
- **Eval**: strategic_reasoning=0.260. Cooperation +137%, but pareto -18%, fairness -18%.

### V5 — Full 5-metric reward
- **Changes**: Reward function plays 3 strategies per completion, computes all 5 metrics (no hardcoded 0.5)
- **Training**: reward_fn ~0.42, slight positive trend
- **Eval**: strategic_reasoning=0.253. Worse — the 5-metric reward was noisy without interactive play.

### V6 — Interactive episodes (THE KEY FIX)
- **Changes**: Reward function plays episodes interactively — model generates actions round-by-round. First action from GRPO completion, subsequent from model.generate(). Enables learning conditional strategies (tit-for-tat, retaliation).
- **Implementation**: Initially used remote HF Space (too slow, ~13s/completion). Switched to local `KantEnvironment` (~3s/completion).
- **Training**: 500 steps, reward 0.39→0.46, positive trend +0.062
- **Eval — BEST RESULT**:
  - cooperation_rate: 0.047 (+96% vs baseline)
  - exploitation_resistance: 0.486 (+6% vs baseline — FIRST IMPROVEMENT)
  - adaptability: 0.020 (+53% vs baseline)
  - pareto_efficiency: 0.311 (-13%)
  - fairness_index: 0.514 (-17%)
  - **strategic_reasoning: 0.276 (-6% vs baseline)**
- **Saved**: `results/v6_trained_llama1b.json`

### V7 — Longer training + reward reweighting
- **Changes**: 1000 steps (2x), pareto weight 0.20→0.25, fairness 0.20→0.25
- **Training**: Steepest reward improvement (+0.133 trend at step 300, +0.118 final)
- **Eval**: strategic_reasoning=0.256. WORSE than V6. Overtrained — training reward diverged from eval metric.
- **Lesson**: More steps ≠ better when reward and eval are misaligned.

---

## Key Discoveries

### 1. Mode Collapse in GRPO
GRPO on short-completion tasks (1-2 token actions) collapses within 50-100 steps. Fix: high KL beta (0.4) + high lr (2e-5) + high temp (1.5). This keeps the model exploring while learning fast.

### 2. Fixed-Action Replay Can't Learn Strategy
Replaying the same action for a full episode means the model can only learn "always cooperate" or "always defect". It can never learn "cooperate first, then retaliate" (tit-for-tat). The fix is interactive episodes where the model generates actions round-by-round.

### 3. Reward-Eval Misalignment
V1-V4: exploit_resist and adaptability were hardcoded to 0.5 in training reward but computed properly in eval. The model was optimizing the wrong objective.

### 4. Training Reward ≠ Eval Score
V7 had the highest training reward trend (+0.118) but worst eval score (0.256). The model learned to game the training setup (single-strategy rotation) in ways that didn't transfer to the full multi-strategy tournament.

### 5. Interactive Episodes Work
V6's interactive play was the only approach that improved exploitation_resistance (the hardest metric — requires different behavior against different opponents). This was the architectural fix that mattered most.

---

## Current Best Results (V6)

| Metric | Baseline | V6 Trained | Change |
|---|---|---|---|
| cooperation_rate | 0.024 | 0.047 | **+96%** |
| exploitation_resistance | 0.458 | 0.486 | **+6%** |
| pareto_efficiency | 0.359 | 0.311 | -13% |
| fairness_index | 0.616 | 0.514 | -17% |
| adaptability | 0.013 | 0.020 | **+53%** |
| strategic_reasoning | 0.294 | 0.276 | -6% |

Three of five metrics improved. Composite still 6% below baseline due to pareto/fairness degradation.

---

## What's Not Done Yet

1. **Beat baseline on composite score** — pareto and fairness still dropping
2. **Safety transfer benchmarks** — HarmBench, ETHICS, TruthfulQA, XSTest, MT-Bench not run yet
3. **Larger model** — all experiments on Llama 1B. Qwen 9B with V6 settings untested
4. **DPO training** — only GRPO tried so far
5. **Self-play** — model vs model training not attempted

---

## Approximate GCP Costs
- ~15 A100 spot hours ($1.10/hr) ≈ $17
- ~5 H100 spot hours ($3.50/hr) ≈ $18
- ~3 L4 on-demand hours ($0.70/hr) ≈ $2
- Misc (T4, n2-highmem, storage) ≈ $3
- **Total: ~$40**

---

## Files Changed/Created This Session

### New files
- `scripts/gcp/` — entire directory (11 files)
- `results/baseline_llama1b.json`
- `results/v3_trained_llama1b.json`
- `results/v4_trained_llama1b.json`
- `results/v5_trained_llama1b.json`
- `results/v6_trained_llama1b.json`
- `results/v7_trained_llama1b.json`

### Modified files
- `train/train.py` — major rewrites (LoRA, interactive reward, local env)
- `train/grpo/trainer.py` — device fix
- `common/games.py` — payoff error handling
- `common/strategies.py` — parse_amount fix
- `bench/evaluation/tournament.py` — logging
- `bench/external/_model_handle.py` — Vertex AI
- `bench/external/constants.py` — Claude judge

### Commits
1. `9e615a6` — GCP infra, GRPO fixes, first eval results
2. `44ecdcf` — Interactive episode reward + v5/v6 configs
3. `2ec47ea` — V6 results + V7 config
