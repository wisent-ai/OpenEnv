# KantBench GRPO Session Log — Reproduction

_Continued from [SESSION_LOG.md](SESSION_LOG.md)._

## How to Reproduce

### Quick start (run V6 — the best config)
```bash
# 1. Auth
gcloud auth login
export GCP_PROJECT=wisent-480400

# 2. Upload code
tar czf /tmp/wisent-openenv.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='RL results' --exclude='paper' --exclude='notebooks' --exclude='.claude' .
gsutil cp /tmp/wisent-openenv.tar.gz gs://kantbench-training/code/wisent-openenv.tar.gz
gsutil cp scripts/gcp/model_configs/llama1b_v6.env gs://kantbench-training/scripts/current_model.env

# 3. Start/restart the train VM
gcloud compute instances start kantbench-train --zone=us-central1-a --project=wisent-480400

# 4. SSH in and setup
gcloud compute ssh kantbench-train --zone=us-central1-a --project=wisent-480400
sudo bash -c '
  systemctl stop kantbench-train.service 2>/dev/null
  rm -f /opt/kantbench/run_id
  rm -rf /workspace/kantbench-output/* /workspace/wisent-openenv
  export GCS_BUCKET=gs://kantbench-training GCP_PROJECT=wisent-480400
  gsutil cp gs://kantbench-training/scripts/setup.sh /opt/kantbench/setup.sh
  chmod +x /opt/kantbench/setup.sh
  bash /opt/kantbench/setup.sh
'

# 5. Monitor training
gcloud compute ssh kantbench-train --zone=us-central1-a --project=wisent-480400 --command="
  sudo journalctl -u kantbench-train --no-pager -n 10
"

# 6. Check metrics (replace checkpoint-N with latest)
gcloud compute ssh kantbench-train --zone=us-central1-a --project=wisent-480400 --command="python3 -c \"
import json, glob, re
ckpts = glob.glob('/workspace/kantbench-output/checkpoint-*/trainer_state.json')
ckpts.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
if ckpts:
    with open(ckpts[-1]) as f:
        d = json.load(f)
    logs = [e for e in d.get('log_history', []) if 'rewards/reward_fn/mean' in e]
    print(f'Step: {d[\\\"global_step\\\"]}')
    if logs:
        print(f'rfn={logs[-1][\\\"rewards/reward_fn/mean\\\"]:.4f} zero_std={logs[-1][\\\"frac_reward_zero_std\\\"]:.4f}')
\""

# 7. Run eval after training completes
gcloud compute scp /tmp/run_eval_now.py kantbench-train:/workspace/wisent-openenv/
gcloud compute ssh kantbench-train --zone=us-central1-a --project=wisent-480400 --command="
sudo bash -c '
  systemctl stop kantbench-train.service 2>/dev/null
  rm -rf /workspace/kantbench-output/checkpoint-*
  source /opt/kantbench/secrets.env
  export WANDB_API_KEY HF_TOKEN GCP_PROJECT=wisent-480400
  cd /workspace/wisent-openenv
  sed -i \"s/max_new_tokens=self._config.max_completion_length/max_new_tokens=16/\" train/grpo/trainer.py
  nohup python3 run_eval_now.py > /workspace/eval.log 2>&1 &
'
"

# 8. Get results (~20 min later)
gcloud compute ssh kantbench-train --zone=us-central1-a --project=wisent-480400 --command="cat /workspace/eval-results/metrics.json"

# 9. Stop VM when done
gcloud compute instances stop kantbench-train --zone=us-central1-a --project=wisent-480400
```

### Run baseline eval
```bash
# Upload run_baseline_eval.py to a GPU instance and run
# The script loads base Llama-3.2-1B-Instruct (no training) and runs
# the same tournament on the same held-out eval games
# Results saved to /workspace/eval-results/baseline_metrics.json
```

### Gotchas
- `gcloud auth` expires every ~12 hours. Run `gcloud auth login` when you see reauth errors
- The HF Space (`openenv-community-kantbench.hf.space`) sleeps after inactivity. Hit the health endpoint to wake it: `curl https://openenv-community-kantbench.hf.space/health`
- Spot A100s get preempted frequently. The systemd service auto-restarts and resumes from GCS checkpoints
- Disk fills up if checkpoints aren't cleaned. `save_total_limit=3` in GRPOConfig, but GCS sync restores old ones. Manually `rm -rf checkpoint-*` before eval
- `run_eval_now.py` is not in the tarball — must be scp'd separately
- The eval `max_new_tokens` must be patched to 16 via sed (base model generates 64 tokens of garbage per action otherwise)
- 5 eval games are skipped (coalition_commons, coalition_rule_voting, nplayer_el_farol, nplayer_volunteer_dilemma, trust_erosion) — they need N-player/coalition env not in base GAMES registry
- V6 interactive episodes: ~80s/step on A100 (10 rounds × model.generate() per episode). 500 steps ≈ 11 hours

### Wandb
- Project: `kantbench-grpo` at https://wandb.ai/3qax-jakub-towarek-technologies/kantbench-grpo
- Hyperopt project: `kantbench-optuna`

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
- `results/evals/baseline_llama1b.json`
- `results/evals/v3_trained_llama1b.json`
- `results/evals/v4_trained_llama1b.json`
- `results/evals/v5_trained_llama1b.json`
- `results/evals/v6_trained_llama1b.json`
- `results/evals/v7_trained_llama1b.json`

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
4. `f9e8dd3` — Qwen 2.5-7B BEATS BASELINE: strategic_reasoning=0.304
5. `9923160` — Add PPO/REINFORCE training script as GRPO alternative
6. `3063ec2` — Add REINFORCE 1B eval results: strategic_reasoning=0.348

---

