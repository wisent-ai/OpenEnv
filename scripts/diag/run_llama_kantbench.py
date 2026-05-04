"""End-to-end Llama-vs-KantBench runner with three opponent modes.

Modes selectable via --mode:
  hardcoded  -- scripted strategies from common/strategies.py
  self       -- opponent is the same model as the player
  cross      -- opponent is a different model (--opponent-model)

Default game set: every game in the live registry, routed through the
matching env class (KantEnvironment for 2-player, NPlayerEnvironment for
N-player). Filter with --games <comma-separated-keys>.

Run:

    PYTHONPATH=. HF_TOKEN=<token> python3 scripts/diag/run_llama_kantbench.py \\
        --mode self --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constant_definitions.game_constants import EVAL_DEFAULT_EPISODES
from constant_definitions.train.agent_constants import MAX_ACTION_TOKENS
from env.environment import KantEnvironment
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.coalition.environment import CoalitionEnvironment

import _episode_play as _ep  # type: ignore[import-not-found]


GAME_TYPE_STRATEGIES = {
    "ultimatum":    ("ultimatum_fair", "ultimatum_low"),
    "trust":        ("trust_fair", "trust_generous"),
    "public_goods": ("public_goods_fair", "public_goods_free_rider"),
}
MATRIX_STRATEGIES = ("tit_for_tat", "always_defect", "always_cooperate")
RANDOM_STRATEGIES = ("random",)
NPLAYER_DEFAULT_STRATEGIES = ("random",)
COALITION_DEFAULT_STRATEGIES = (
    "coalition_random", "coalition_loyal", "coalition_betrayer",
    "coalition_tit_for_tat", "coalition_grim_trigger",
)
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE_NUMERATOR = 7
TEMPERATURE_DENOMINATOR = 10
TEMPERATURE = TEMPERATURE_NUMERATOR / TEMPERATURE_DENOMINATOR


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("hardcoded", "self", "cross"),
                   default="hardcoded")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF id of the player's model")
    p.add_argument("--opponent-model", default=None,
                   help="HF id of the opponent model; required for --mode cross")
    p.add_argument("--lora-path", default=None,
                   help="Path to a PEFT/LoRA adapter dir; merged into the player model")
    p.add_argument("--episodes", type=int, default=EVAL_DEFAULT_EPISODES,
                   help="episodes per (game, opponent) pair")
    p.add_argument("--games", default=None,
                   help="comma-separated game keys (default: every game)")
    return p.parse_args()


def _strategies_for(cfg) -> tuple[str, ...]:
    """Pick opponents appropriate to cfg's game_type, falling back by arity."""
    if cfg.game_type in GAME_TYPE_STRATEGIES:
        return GAME_TYPE_STRATEGIES[cfg.game_type]
    return MATRIX_STRATEGIES if len(cfg.actions) == 2 else RANDOM_STRATEGIES


def _resolve_games(games_filter):
    """Return list of (game_key, env_kind, strategies) from every registry."""
    from common.games import GAMES, GAME_FACTORIES
    # Importing nplayer_games.py runs NPLAYER_GAMES.update(_BUILTIN_NPLAYER_GAMES)
    # at module load; without this side effect NPLAYER_GAMES is empty.
    import common.games_meta.nplayer_games  # noqa: F401
    from common.games_meta.nplayer_config import NPLAYER_GAMES

    requested = None
    if games_filter:
        requested = {g.strip() for g in games_filter.split(",") if g.strip()}

    rows: list[tuple[str, str, tuple[str, ...]]] = []
    keys_2p = sorted(set(GAMES.keys()) | set(GAME_FACTORIES.keys()))
    for key in keys_2p:
        if requested is not None and key not in requested:
            continue
        cfg = GAMES.get(key) or GAME_FACTORIES.get(key, lambda: None)()
        if cfg is None:
            continue
        rows.append((key, "2p", _strategies_for(cfg)))

    # Coalition games are also written into NPLAYER_GAMES by coalition_config:208;
    # claim them for the coalition env first so they don't double-count.
    from common.games_meta.coalition_config import COALITION_GAMES
    for key in sorted(COALITION_GAMES.keys()):
        if requested is not None and key not in requested:
            continue
        rows.append((key, "coalition", COALITION_DEFAULT_STRATEGIES))

    for key in sorted(NPLAYER_GAMES.keys()):
        if requested is not None and key not in requested:
            continue
        if key in COALITION_GAMES:
            continue
        rows.append((key, "nplayer", NPLAYER_DEFAULT_STRATEGIES))

    if requested is not None:
        seen = {k for k, _, _ in rows}
        missing = requested - seen
        if missing:
            raise SystemExit(f"--games unknown: {sorted(missing)}")
    if not rows:
        raise SystemExit("No games selected.")
    return rows


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_id: str):
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


def _print_rows(rows):
    print("\n=== Per (game, opponent) mean self-payoff ===", flush=True)
    for game, opp, score, rounds in rows:
        per_round = score / rounds if rounds else float("nan")
        print(f"  {game:28s}  vs {opp:24s}  "
              f"player_score_total={score:7.2f}  rounds={rounds}  "
              f"per_round={per_round:6.3f}", flush=True)
    print("\n=== Per-game aggregate (sum across opponents) ===", flush=True)
    by_game = {}
    for game, _, score, rounds in rows:
        s, r = by_game.get(game, (0.0, 0))
        by_game[game] = (s + score, r + rounds)
    for game, (score, rounds) in by_game.items():
        per_round = score / rounds if rounds else float("nan")
        print(f"  {game:28s}  mean_self_payoff_per_round={per_round:6.3f}  "
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
    if args.lora_path:
        from peft import PeftModel
        player_model = PeftModel.from_pretrained(player_model, args.lora_path)
        print(f"[run] LoRA adapter loaded from {args.lora_path}", flush=True)
    print(f"[run] player model loaded in {time.time() - t0:.1f}s", flush=True)

    _ep.install_parse_action_counter()
    p_gen = _build_generate_fn(player_model, player_tok, device)
    agent_fn_2p = _ep.make_2p_agent(p_gen)
    agent_fn_n = _ep.make_nplayer_agent(p_gen)

    opp_fn_2p = opp_fn_n = None
    opp_gen = None
    if args.mode == "self":
        opp_gen = _build_generate_fn(player_model, player_tok, device)
        opp_fn_2p = _ep.make_2p_agent(opp_gen)
        opp_fn_n = _ep.make_nplayer_agent(opp_gen)
        opp_label = "self"
    elif args.mode == "cross":
        t1 = time.time()
        opp_model, opp_tok, _dev = _load_model(args.opponent_model)
        print(f"[run] opponent model loaded in {time.time() - t1:.1f}s", flush=True)
        opp_gen = _build_generate_fn(opp_model, opp_tok, device)
        opp_fn_2p = _ep.make_2p_agent(opp_gen)
        opp_fn_n = _ep.make_nplayer_agent(opp_gen)
        opp_label = f"cross[{args.opponent_model.split('/')[-1]}]"
    else:
        opp_label = ""

    selected = _resolve_games(args.games)
    print(f"[run] {len(selected)} game(s) selected "
          f"({sum(1 for _, k, _ in selected if k == '2p')} 2P, "
          f"{sum(1 for _, k, _ in selected if k == 'nplayer')} N-player, "
          f"{sum(1 for _, k, _ in selected if k == 'coalition')} coalition)",
          flush=True)

    env_by_kind = {
        "2p": KantEnvironment(),
        "nplayer": NPlayerEnvironment(),
        "coalition": CoalitionEnvironment(),
    }
    agent_by_kind = {"2p": agent_fn_2p, "nplayer": agent_fn_n, "coalition": agent_fn_n}
    opp_by_kind = {"2p": opp_fn_2p, "nplayer": opp_fn_n, "coalition": opp_fn_n}
    t2 = time.time()
    rows = []
    for key, env_kind, strategies in selected:
        rows.extend(_ep.play_rows(
            env_kind, env_by_kind[env_kind], key, strategies, args.episodes,
            args.mode, agent_by_kind[env_kind], opp_by_kind[env_kind], opp_label,
            generate_fn=p_gen, opp_generate_fn=opp_gen,
        ))
    print(f"[run] tournament finished in {time.time() - t2:.1f}s", flush=True)
    _print_rows(rows)

    miss = _ep.PARSE_MISS_COUNT
    total = _ep.PARSE_TOTAL_COUNT
    pct = (miss / max(1, total)) * 100
    print(f"\n[run] parse misses: {miss}/{total} ({pct:.1f}%)", flush=True)
    print(f"[run] total wallclock = {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
