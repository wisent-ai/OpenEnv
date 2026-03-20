"""Extract training metrics from the latest checkpoint for Optuna scoring."""
import glob
import json

ckpts = sorted(glob.glob("/workspace/output/checkpoint-*/trainer_state.json"))
if not ckpts:
    print("NO CHECKPOINTS")
    with open("/workspace/trial_result.json", "w") as f:
        json.dump({"error": "no checkpoints"}, f)
    raise SystemExit(0)

with open(ckpts[-1]) as f:
    d = json.load(f)

logs = d.get("log_history", [])
result = {}

if logs:
    last = logs[-1]
    result["reward_fn"] = last.get("rewards/reward_fn/mean", 0)
    result["format_fn"] = last.get("rewards/format_reward_fn/mean", 0)
    result["zero_std"] = last.get("frac_reward_zero_std", 1)
    result["entropy"] = last.get("entropy", 0)
    result["step"] = last.get("step", 0)

rfn_values = [
    e.get("rewards/reward_fn/mean", 0)
    for e in logs
    if "rewards/reward_fn/mean" in e
]
if len(rfn_values) >= 2:
    result["reward_trend"] = rfn_values[-1] - rfn_values[0]
else:
    result["reward_trend"] = 0

with open("/workspace/trial_result.json", "w") as f:
    json.dump(result, f)

print(json.dumps(result, indent=2))
