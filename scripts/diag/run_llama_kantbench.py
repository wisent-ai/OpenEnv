"""End-to-end Llama-vs-KantBench runner with three opponent modes.

Three opponent modes selectable via --mode:
  hardcoded  -- opponent is a scripted strategy from common/strategies.py
                (per-game appropriate: matrix strategies for matrix games,
                ultimatum_*/trust_*/public_goods_* for the asymmetric games)
  self       -- opponent is the same model as the player (self-play)
  cross      -- opponent is a different model loaded from --opponent-model

Run:

    PYTHONPATH=. HF_TOKEN=<token> python3 scripts/diag/run_llama_kantbench.py \\
        --mode hardcoded
    PYTHONPATH=. HF_TOKEN=<token> python3 scripts/diag/run_llama_kantbench.py \\
        --mode self --model meta-llama/Llama-3.2-1B-Instruct
    PYTHONPATH=. HF_TOKEN=<token> python3 scripts/diag/run_llama_kantbench.py \\
        --mode cross --model meta-llama/Llama-3.2-1B-Instruct \\
                     --opponent-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constant_definitions.game_constants import EVAL_DEFAULT_EPISODES
from constant_definitions.train.agent_constants import MAX_ACTION_TOKENS
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import LLMAgent, parse_action


GAMES_AND_STRATEGIES = (
    ("prisoners_dilemma", ("tit_for_tat", "always_defect", "always_cooperate")),
    ("stag_hunt",         ("tit_for_tat", "always_defect", "always_cooperate")),
    ("hawk_dove",         ("tit_for_tat", "always_defect", "always_cooperate")),
    ("ultimatum",         ("ultimatum_fair", "ultimatum_low")),
    ("trust",             ("trust_fair", "trust_generous")),
    ("public_goods",      ("public_goods_fair", "public_goods_free_rider")),
)
GAMES_FOR_LLM_OPPONENT = tuple(g for g, _ in GAMES_AND_STRATEGIES)
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE_NUMERATOR = 7
TEMPERATURE_DENOMINATOR = 10
TEMPERATURE = TEMPERATURE_NUMERATOR / TEMPERATURE_DENOMINATOR

_PARSE_MISS_COUNT = 0
_PARSE_TOTAL_COUNT = 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("hardcoded", "self", "cross"),
                   default="hardcoded")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF id of the player's model")
    p.add_argument("--opponent-model", default=None,
                   help="HF id of the opponent's model; required for --mode cross")
    p.add_argument("--episodes", type=int, default=EVAL_DEFAULT_EPISODES,
                   help="episodes per (game, opponent) pair")
    return p.parse_args()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_id: str):
    """Return (model, tokenizer, device) on the best available accelerator."""
    device = _device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    mdl.eval()
    return mdl, tok, device


def _build_generate_fn(model, tokenizer, device):
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
                max_new_tokens=MAX_ACTION_TOKENS,
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


def _agent_fn_from_llm(agent: LLMAgent):
    def _fn(obs: GameObservation) -> GameAction:
        return agent(obs)
    return _fn


def _play_one_episode(env: KantEnvironment, agent_fn, *, game: str, **reset_kw):
    """Loop env.step until done. Return (player_score, rounds_played)."""
    obs = env.reset(game=game, **reset_kw)
    while not obs.done:
        action = agent_fn(obs)
        obs = env.step(action)
    return obs.player_score, obs.current_round


def _run_hardcoded(env, agent_fn, episodes):
    rows = []
    for game, strategies in GAMES_AND_STRATEGIES:
        for strat in strategies:
            score_sum, round_sum = 0.0, 0
            for _ in range(episodes):
                ps, rounds = _play_one_episode(
                    env, agent_fn, game=game, strategy=strat,
                )
                score_sum += ps
                round_sum += rounds
            rows.append((game, strat, score_sum, round_sum))
    return rows


def _run_llm_opponent(env, agent_fn, opponent_fn, label, episodes):
    rows = []
    for game in GAMES_FOR_LLM_OPPONENT:
        score_sum, round_sum = 0.0, 0
        for _ in range(episodes):
            ps, rounds = _play_one_episode(
                env, agent_fn, game=game, opponent_fn=opponent_fn,
            )
            score_sum += ps
            round_sum += rounds
        rows.append((game, label, score_sum, round_sum))
    return rows


def _print_rows(rows):
    print("\n=== Per (game, opponent) mean self-payoff ===", flush=True)
    for game, opp, score, rounds in rows:
        per_round = score / rounds if rounds else float("nan")
        print(f"  {game:20s}  vs {opp:24s}  "
              f"player_score_total={score:7.2f}  rounds={rounds}  "
              f"per_round={per_round:6.3f}", flush=True)
    print("\n=== Per-game aggregate (sum across opponents) ===", flush=True)
    by_game = {}
    for game, _, score, rounds in rows:
        s, r = by_game.get(game, (0.0, 0))
        by_game[game] = (s + score, r + rounds)
    for game, (score, rounds) in by_game.items():
        per_round = score / rounds if rounds else float("nan")
        print(f"  {game:20s}  mean_self_payoff_per_round={per_round:6.3f}  "
              f"(rounds={rounds})", flush=True)


def main() -> None:
    args = _parse_args()
    if args.mode == "cross" and not args.opponent_model:
        raise SystemExit("--mode cross requires --opponent-model")

    print(f"[run] mode={args.mode}  player_model={args.model}  device={_device()}",
          flush=True)
    if args.mode == "cross":
        print(f"[run] opponent_model={args.opponent_model}", flush=True)

    t0 = time.time()
    player_model, player_tok, device = _load_model(args.model)
    print(f"[run] player model loaded in {time.time() - t0:.1f}s", flush=True)

    _wrap_parse_action_with_counters()
    player_agent = LLMAgent(
        generate_fn=_build_generate_fn(player_model, player_tok, device),
    )
    agent_fn = _agent_fn_from_llm(player_agent)

    opponent_fn = None
    if args.mode == "self":
        opponent_agent = LLMAgent(
            generate_fn=_build_generate_fn(player_model, player_tok, device),
        )
        opponent_fn = _agent_fn_from_llm(opponent_agent)
    elif args.mode == "cross":
        t1 = time.time()
        opp_model, opp_tok, _ = _load_model(args.opponent_model)
        print(f"[run] opponent model loaded in {time.time() - t1:.1f}s",
              flush=True)
        opponent_agent = LLMAgent(
            generate_fn=_build_generate_fn(opp_model, opp_tok, device),
        )
        opponent_fn = _agent_fn_from_llm(opponent_agent)

    env = KantEnvironment()
    t2 = time.time()
    if args.mode == "hardcoded":
        rows = _run_hardcoded(env, agent_fn, args.episodes)
    elif args.mode == "self":
        rows = _run_llm_opponent(env, agent_fn, opponent_fn, "self",
                                 args.episodes)
    else:
        rows = _run_llm_opponent(
            env, agent_fn, opponent_fn,
            f"cross[{args.opponent_model.split('/')[-1]}]",
            args.episodes,
        )
    print(f"[run] tournament finished in {time.time() - t2:.1f}s", flush=True)
    _print_rows(rows)

    miss_pct = (_PARSE_MISS_COUNT / max(1, _PARSE_TOTAL_COUNT)) * 100
    print(f"\n[run] parse misses: {_PARSE_MISS_COUNT}/{_PARSE_TOTAL_COUNT} "
          f"({miss_pct:.1f}%)", flush=True)
    print(f"[run] total wallclock = {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
