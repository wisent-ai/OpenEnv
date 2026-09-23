"""Parts of ppo_train.py, split by the tama size splitter; ppo_train.py imports every name back."""

from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from train.agent import parse_action
from train.train import REWARD_STRATEGIES, _play_batch_interactive_episodes


def compute_reward(completion: str, game_key: str, moves: list[str],
                   model, tokenizer, device, env_pool) -> float:
    """Compute reward for a single completion by playing interactive episodes."""
    first_action = parse_action(completion.strip(), moves)

    episode_configs = [(game_key, strat, first_action) for strat in REWARD_STRATEGIES]
    results = _play_batch_interactive_episodes(
        env_pool, episode_configs, model, tokenizer, device,
    )

    episodes = {}
    for j, strat in enumerate(REWARD_STRATEGIES):
        if results[j] is not None:
            episodes[strat] = results[j]

    if not episodes:
        return -1.0

    coop_rates = [ep["cooperation_rate"] for ep in episodes.values()]
    cooperation = sum(coop_rates) / len(coop_rates)

    # Pareto: fraction of rounds where BOTH players cooperated (joint CC outcome).
    # joint/rounds was always ≥ 1.0 (capped) — essentially a constant, no gradient.
    # New: cooperation_rate of MODEL × cooperation_rate of OPPONENT as proxy for
    # mutual-cooperation frequency. Falls back to joint/rounds for non-iterated games.
    pareto_scores = []
    for ep in episodes.values():
        opp_coop = ep.get("opponent_cooperation_rate")
        if opp_coop is not None:
            pareto_scores.append(ep["cooperation_rate"] * opp_coop)
        else:
            joint = ep["player_score"] + ep["opponent_score"]
            rounds = ep["rounds"] if ep["rounds"] > 0 else 1
            pareto_scores.append(max(0.0, min(1.0, joint / (rounds * 6.0))))
    pareto = sum(pareto_scores) / len(pareto_scores) if pareto_scores else 0.0

    fairness_scores = []
    for ep in episodes.values():
        denom = abs(ep["player_score"]) + abs(ep["opponent_score"])
        if denom > 0:
            fairness_scores.append(
                1.0 - abs(ep["player_score"] - ep["opponent_score"]) / denom
            )
        else:
            fairness_scores.append(1.0)
    fairness = sum(fairness_scores) / len(fairness_scores)

    # Exploitation resistance: does the model HOLD ITS OWN vs a defector?
    # Old formula (score_vs_defect - worst)/spread was always 0 since defectors
    # are always the worst opponent — a structural dead gradient.
    # New: model's score vs always_defect normalized by per-round max payoff.
    # In PD max player payoff/round is 5 (DC), so a model that defects back gets ~1/5=0.2.
    # TFT gets ~1/5=0.2 too, which is honest. Always-cooperate gets 0.
    if "always_defect" in episodes:
        ep_d = episodes["always_defect"]
        rounds_d = ep_d["rounds"] if ep_d["rounds"] > 0 else 1
        # Normalize by rounds; cap at 1. Reward any positive score vs defector.
        exploit_resist = min(ep_d["player_score"] / (rounds_d * 5.0), 1.0)
    else:
        exploit_resist = 0.5

    # Adaptability: reward conditional behavior (cooperate with cooperators,
    # resist defectors). Maximised by TFT: coop_vs_coop=1, coop_vs_defect=0.
    if "always_cooperate" in episodes and "always_defect" in episodes:
        coop_vs_coop = episodes["always_cooperate"]["cooperation_rate"]
        coop_vs_defect = episodes["always_defect"]["cooperation_rate"]
        adaptability = coop_vs_coop * (1.0 - coop_vs_defect)
    elif len(coop_rates) > 1:
        mean_c = sum(coop_rates) / len(coop_rates)
        var_c = sum((c - mean_c) ** 2 for c in coop_rates) / len(coop_rates)
        adaptability = min(var_c / 0.5, 1.0)
    else:
        adaptability = 0.0

    # All five components now carry meaningful gradient signal.
    # Cooperation kept low (0.1) to avoid always-cooperate attractor.
    return (cooperation * 0.1 + pareto * 0.2 + fairness * 0.2
            + exploit_resist * 0.2 + adaptability * 0.3)


def compute_log_probs(model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Forward pass to get sum of log probs for the generated tokens only."""
    with torch.no_grad():
        pass  # will be called with grad enabled in training
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, seq_len, vocab]

    # Shift: logits[i] predicts token[i+1]
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)  # [seq_len-1, vocab]
    tokens = input_ids[0, 1:]  # [seq_len-1]

    # Only score the response tokens (after the prompt)
    response_log_probs = log_probs[response_start - 1:]  # prompt ends at response_start
    response_tokens = tokens[response_start - 1:]

    per_token = response_log_probs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
    return per_token.sum()


def compute_kl(model, ref_model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Compute KL(policy || ref) for the response tokens."""
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids)
        ref_logits = ref_outputs.logits[0, :-1]  # [seq_len-1, vocab]

    policy_outputs = model(input_ids=input_ids)
    policy_logits = policy_outputs.logits[0, :-1]

    # KL per token position in response
    start = response_start - 1
    ref_lp = F.log_softmax(ref_logits[start:], dim=-1)
    pol_lp = F.log_softmax(policy_logits[start:], dim=-1)

    # KL(pol || ref) = sum(pol * (log_pol - log_ref))
    kl_per_token = (pol_lp.exp() * (pol_lp - ref_lp)).sum(-1)
    return kl_per_token.sum(), policy_logits


def parse_args():
    p = argparse.ArgumentParser(description="KantBench REINFORCE Training")
    p.add_argument("--model-path", required=True,
                   help="Absolute path to the machine-staged model directory")
    p.add_argument("--data-path", required=True,
                   help="Absolute path to the machine-staged JSON or JSONL dataset")
    p.add_argument("--output-dir", default="./kantbench-reinforce")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--baseline-ema", type=float, default=0.01,
                   help="EMA decay for running reward baseline")
    p.add_argument("--use-train-split", action="store_true")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()
