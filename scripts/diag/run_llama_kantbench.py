"""End-to-end run: meta-llama/Llama-3.2-1B-Instruct vs the KantBench env.

Loads Llama-3.2-1B-Instruct on Apple MPS, wraps it in the existing
LLMAgent + PromptBuilder, runs a TournamentRunner across six base games
(PD, Stag Hunt, Hawk-Dove, Ultimatum, Trust, Public Goods) against three
baseline opponent strategies, and emits the agent's mean per-round
self-payoff per (game, strategy) and aggregated per game.

Run:

    PYTHONPATH=. HF_TOKEN=<token> python3 scripts/diag/run_llama_kantbench.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project imports go via PYTHONPATH=.
from bench.evaluation.tournament import TournamentRunner
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import APIAgent, LLMAgent, PromptBuilder, parse_action  # noqa: F401


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
GAMES_TO_RUN = (
    "prisoners_dilemma",
    "stag_hunt",
    "hawk_dove",
    "ultimatum",
    "trust",
    "public_goods",
)
STRATEGIES_TO_RUN = ("tit_for_tat", "always_defect", "always_cooperate")
EPISODES_PER_PAIR = 5
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.7

# Module-level counters: track how often parse_action could not match the
# completion to any available action and resorted to random.choice.
_PARSE_MISS_COUNT = 0
_PARSE_TOTAL_COUNT = 0


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_generate_fn(model, tokenizer, device):
    """Wrap (model, tokenizer) into the prompt -> completion callable LLMAgent expects."""
    def _generate(prompt: str) -> str:
        messages = [
            {"role": "system",
             "content": "You play a game-theory round. Reply with exactly one of the listed actions."},
            {"role": "user", "content": prompt},
        ]
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return completion.strip()
    return _generate


def _wrap_parse_action_with_counters():
    """Replace train.agent.parse_action with a wrapper that counts misses.

    A "miss" is when the wrapper detects the original parse_action would
    have returned random.choice(...) because no action token actually
    appears in the completion. We don't change behavior; we only count.
    """
    import train.agent as _agent_mod

    _original = _agent_mod.parse_action

    def _wrapped(response: str, available_actions):
        global _PARSE_MISS_COUNT, _PARSE_TOTAL_COUNT
        _PARSE_TOTAL_COUNT += 1
        stripped = response.strip()
        lower = stripped.lower()
        matched = (
            stripped in available_actions
            or any(a.lower() == lower for a in available_actions)
            or any(a.lower() in lower for a in available_actions)
        )
        if not matched:
            _PARSE_MISS_COUNT += 1
        return _original(response, available_actions)

    _agent_mod.parse_action = _wrapped


def main() -> None:
    print(f"[run] device={_device()}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if _device() != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(_device())
    model.eval()
    print(f"[run] model loaded in {time.time() - t0:.1f}s", flush=True)

    _wrap_parse_action_with_counters()
    generate_fn = _build_generate_fn(model, tokenizer, _device())
    agent = LLMAgent(generate_fn=generate_fn)

    def _agent_fn(obs: GameObservation) -> GameAction:
        return agent(obs)

    env = KantEnvironment()
    runner = TournamentRunner(env=env, agent_fn=_agent_fn)

    print(
        f"[run] starting tournament on {len(GAMES_TO_RUN)} games "
        f"x {len(STRATEGIES_TO_RUN)} strategies x {EPISODES_PER_PAIR} episode(s)",
        flush=True,
    )
    t1 = time.time()
    results = runner.run_tournament(
        games=list(GAMES_TO_RUN),
        strategies=list(STRATEGIES_TO_RUN),
        num_episodes=EPISODES_PER_PAIR,
    )
    elapsed = time.time() - t1
    print(f"[run] tournament finished in {elapsed:.1f}s "
          f"(total episodes={results.total_episodes})", flush=True)

    # Per-(game, strategy) mean self-payoff -- the headline metric.
    print("\n=== Per (game, strategy) mean self-payoff ===", flush=True)
    for g_key, g_res in results.games.items():
        for s_key, s_res in g_res.strategy_results.items():
            rounds = sum(e.rounds_played for e in s_res.episodes) or 1
            print(
                f"  {g_key:20s}  vs {s_key:18s}  "
                f"player_score_total={s_res.total_player_score:7.2f}  "
                f"rounds={rounds}  per_round={s_res.total_player_score / rounds:6.3f}",
                flush=True,
            )

    # Aggregated mean self-payoff per game across all opponents.
    print("\n=== Mean self-payoff per game (averaged across opponents) ===",
          flush=True)
    for g_key, g_res in results.games.items():
        score_sum = 0.0
        rounds_sum = 0
        for s_res in g_res.strategy_results.values():
            score_sum += s_res.total_player_score
            for ep in s_res.episodes:
                rounds_sum += ep.rounds_played
        mean = score_sum / rounds_sum if rounds_sum else float("nan")
        print(f"  {g_key:20s}  mean_self_payoff_per_round={mean:6.3f}  "
              f"(rounds={rounds_sum})", flush=True)

    print(
        f"\n[run] parse misses: {_PARSE_MISS_COUNT}/{_PARSE_TOTAL_COUNT} "
        f"({(_PARSE_MISS_COUNT / max(1, _PARSE_TOTAL_COUNT)) * 100:.1f}% "
        "of LLM completions did not contain any valid action token "
        "and were resolved by random.choice in train.agent.parse_action)",
        flush=True,
    )
    print(f"[run] total wallclock = {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
