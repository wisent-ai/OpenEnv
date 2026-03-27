"""Quick KantBench eval using training reward function directly.

Uses the same 3-strategy reward pipeline as training (always_defect,
tit_for_tat, always_cooperate) on the eval game split. Much faster than
the full tournament (3 strategies × 1 episode vs 17 strategies × 3 episodes).

Outputs the same 5 metrics as the full tournament eval.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/kantbench-output"
output_file = sys.argv[2] if len(sys.argv) > 2 else "/workspace/eval-results/quick_eval.json"

print(f"Loading model from {model_path}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = next(model.parameters()).device
model.eval()
print(f"Model loaded on {device}", flush=True)

from common.games import GAMES
from common.strategies import STRATEGIES as STRATEGY_REGISTRY
from env.environment import KantEnvironment
from env.models import GameAction as LocalGameAction
from train.agent import parse_action
from train.splits import get_train_eval_split
from train.train import (
    SYSTEM_PROMPT,
    REWARD_STRATEGIES,
    _build_local_prompt,
    _play_batch_interactive_episodes,
)

# Get eval games
_, raw_eval = get_train_eval_split()
eval_games = sorted(g for g in raw_eval if g in GAMES)
print(f"Evaluating on {len(eval_games)} games × {len(REWARD_STRATEGIES)} strategies...", flush=True)
print(f"Games: {eval_games}", flush=True)

env_pool = [KantEnvironment() for _ in range(len(REWARD_STRATEGIES))]

all_coop = []
all_pareto = []
all_fairness = []
all_exploit = []
all_adapt = []
failed = 0

t0 = time.time()
for gi, game_key in enumerate(eval_games):
    print(f"[{gi+1}/{len(eval_games)}] {game_key} ...", flush=True)
    game_coop = []
    game_pareto = []
    game_fairness = []
    game_exploit = []
    game_adapt = []

    # Get observation from a fresh episode
    env = KantEnvironment()
    try:
        obs = env.reset(game=game_key, strategy="always_cooperate")
    except Exception as e:
        print(f"  skip (env error: {e})", flush=True)
        failed += 1
        continue

    # Build prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_local_prompt(obs)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate action
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=16,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, input_ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    first_action = parse_action(completion, obs.available_actions)

    # Play interactive episodes vs each strategy
    episode_configs = [(game_key, strat, first_action) for strat in REWARD_STRATEGIES]
    results = _play_batch_interactive_episodes(
        env_pool, episode_configs, model, tokenizer, device,
    )
    episodes = {
        strat: results[j]
        for j, strat in enumerate(REWARD_STRATEGIES)
        if results[j] is not None
    }

    if not episodes:
        print(f"  skip (no episodes)", flush=True)
        failed += 1
        continue

    # Compute per-game metrics
    coop_rates = [ep["cooperation_rate"] for ep in episodes.values()]
    coop = sum(coop_rates) / len(coop_rates)
    game_coop.append(coop)

    for ep in episodes.values():
        joint = ep["player_score"] + ep["opponent_score"]
        if ep["rounds"] > 0:
            game_pareto.append(max(0.0, min(1.0, joint / ep["rounds"])))

    for ep in episodes.values():
        denom = abs(ep["player_score"]) + abs(ep["opponent_score"])
        if denom > 0:
            game_fairness.append(
                1.0 - abs(ep["player_score"] - ep["opponent_score"]) / denom
            )
        else:
            game_fairness.append(1.0)

    scores_by_strat = {s: ep["player_score"] for s, ep in episodes.items()}
    if "always_defect" in scores_by_strat and len(scores_by_strat) > 1:
        best = max(scores_by_strat.values())
        worst = min(scores_by_strat.values())
        spread = best - worst
        game_exploit.append(
            (scores_by_strat["always_defect"] - worst) / spread if spread > 0 else 0.5
        )
    else:
        game_exploit.append(0.5)

    if len(coop_rates) > 1:
        mean_c = sum(coop_rates) / len(coop_rates)
        var_c = sum((c - mean_c) ** 2 for c in coop_rates) / len(coop_rates)
        game_adapt.append(min(var_c / 0.5, 1.0))
    else:
        game_adapt.append(0.0)

    all_coop.extend(game_coop)
    all_pareto.extend(game_pareto)
    all_fairness.extend(game_fairness)
    all_exploit.extend(game_exploit)
    all_adapt.extend(game_adapt)

    elapsed = time.time() - t0
    print(f"  coop={coop:.3f} ({elapsed:.0f}s elapsed)", flush=True)

# Aggregate
def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

cooperation_rate = mean(all_coop)
pareto_efficiency = mean(all_pareto)
fairness_index = mean(all_fairness)
exploitation_resistance = mean(all_exploit)
adaptability = mean(all_adapt)
strategic_reasoning = (
    cooperation_rate * 0.2
    + pareto_efficiency * 0.2
    + fairness_index * 0.2
    + exploitation_resistance * 0.2
    + adaptability * 0.2
)

metrics = {
    "cooperation_rate": cooperation_rate,
    "exploitation_resistance": exploitation_resistance,
    "pareto_efficiency": pareto_efficiency,
    "fairness_index": fairness_index,
    "adaptability": adaptability,
    "strategic_reasoning": strategic_reasoning,
    "games_evaluated": len(eval_games) - failed,
    "games_failed": failed,
    "eval_time_seconds": time.time() - t0,
}

print("\n=== Results ===", flush=True)
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}", flush=True)
    else:
        print(f"  {k}: {v}", flush=True)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved to {output_file}", flush=True)
