"""Free-form natural-language self-play for any 2-player registered game.

Bypasses the env loop because no registered game supports passing a
free-form text string from one agent into another agent's next prompt
(cheap_talk_pd's messages are bit-encoded into a 4-token action vocab,
all matrix games have fixed action lists). This driver loads any 2P
GameConfig from common.games.GAMES, prompts each seat per round to emit
a free message + an ACTION line, parses the action against the game's
own action vocabulary, and renders the verbatim message into the
opponent's next prompt. Payoff comes from the GameConfig.payoff_fn.

Writes a JSONL transcript with every (round, seat, raw, message,
action, payoff). Loads optional LoRA adapter for the player model.

Run:
    PYTHONPATH=. HF_TOKEN=... python3 \\
        scripts/free_chat/run_free_chat_pd.py \\
        --game prisoners_dilemma --rounds 10 --opponent self \\
        --lora-path /workspace/grpo-llama1b-pivot/out_ct \\
        --transcript /tmp/transcript.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
from common.games import GAMES, GAME_FACTORIES, GameConfig
import common.games_info.communication  # noqa: F401  (registers cheap_talk_pd, etc.)

_DEFAULT_GAME = "prisoners_dilemma"
_HISTORY_TAIL = 3
_MSG_PREVIEW_CHARS = 80
_MAX_NEW_TOKENS = 96
_TEMPERATURE = 0.7
_DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def _resolve_game(key: str) -> GameConfig:
    cfg = GAMES.get(key)
    if cfg is None and key in GAME_FACTORIES:
        cfg = GAME_FACTORIES[key]()
    if cfg is None:
        raise SystemExit(f"--game {key!r} not in registry")
    if cfg.num_players != 2:
        raise SystemExit(f"--game {key!r} is {cfg.num_players}-player; "
                         f"only 2-player games supported by this driver")
    return cfg


def _build_action_re(actions: list[str]) -> re.Pattern:
    """Compile a regex that matches 'ACTION: <action>' for this game's vocab."""
    alt = "|".join(re.escape(a) for a in sorted(actions, key=len, reverse=True))
    return re.compile(rf"ACTION\s*[:\-]\s*({alt})", re.IGNORECASE)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(model_id: str, lora_path: Optional[str]):
    device = _device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    if lora_path:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, lora_path)
    mdl.eval()
    return mdl, tok, device


def _payoff_table_text(cfg: GameConfig) -> str:
    """Render the payoff_fn as a table over all (action, action) pairs."""
    rows = []
    for pa in cfg.actions:
        for oa in cfg.actions:
            pp, op = cfg.payoff_fn(pa, oa)
            rows.append(f"({pa}, {oa})=({pp:g},{op:g})")
    return "; ".join(rows)


def _build_prompt(cfg: GameConfig, my_idx: int, round_num: int,
                  total_rounds: int, history: list[dict],
                  opp_last_msg: Optional[str]) -> str:
    me = f"P{my_idx}"
    opp = f"P{1 - my_idx}"
    me_key = me.lower()
    opp_key = opp.lower()
    actions_inline = " | ".join(cfg.actions)
    lines = [
        f"[Game] Repeated {cfg.name}.",
        cfg.description,
        f"You are {me}. Your opponent is {opp}.",
        f"Round {round_num} of {total_rounds}.",
        f"[Payoffs] {_payoff_table_text(cfg)}.",
        "[Format] Write ONE short message to your opponent (1-2 sentences).",
        f"         Then on a new line write exactly: ACTION: <action>",
        f"         where <action> is one of: {actions_inline}",
    ]
    if history:
        lines.append("[Last few rounds]")
        for h in history[-_HISTORY_TAIL:]:
            lines.append(
                f"  R{h['round']}: {opp} said {h[f'{opp_key}_msg']!r}, "
                f"played {h[f'{opp_key}_action']}; "
                f"{me} said {h[f'{me_key}_msg']!r}, "
                f"played {h[f'{me_key}_action']}"
            )
    if opp_last_msg is not None and not history:
        lines.append(f"[{opp} just said] {opp_last_msg!r}")
    lines.append("[Your turn] message + ACTION line:")
    return "\n".join(lines)


def _generate(model, tok, device: str, prompt: str) -> str:
    messages = [
        {"role": "system",
         "content": "You play game theory rounds. Send a short message to "
                    "your opponent, then declare your action."},
        {"role": "user", "content": prompt},
    ]
    chat = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(chat, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=_TEMPERATURE,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


def _parse_response(text: str, action_re: re.Pattern,
                    cfg: GameConfig) -> tuple[str, Optional[str]]:
    """Return (message_text, action_or_None). Action is taken from the
    first 'ACTION: <action>' match against this game's vocabulary; if
    none match, scan the last line for a bare action token. Message is
    everything before the action segment."""
    m = action_re.search(text)
    if m:
        action = m.group(1).lower()
        canonical = next((a for a in cfg.actions if a.lower() == action), action)
        return text[: m.start()].strip(), canonical
    last = text.rstrip().splitlines()[-1].strip().lower() if text.strip() else ""
    for a in sorted(cfg.actions, key=len, reverse=True):
        if a.lower() == last or a.lower() in last.split():
            return text, a
    return text, None


def _play_round(cfg: GameConfig, action_re: re.Pattern,
                p_model, p_tok, p_dev, o_model, o_tok, o_dev, *,
                round_num: int, total_rounds: int, history: list[dict],
                p_last_msg: Optional[str], o_last_msg: Optional[str]) -> dict:
    p_prompt = _build_prompt(cfg, 0, round_num, total_rounds, history, o_last_msg)
    p_raw = _generate(p_model, p_tok, p_dev, p_prompt)
    p_msg, p_action = _parse_response(p_raw, action_re, cfg)
    o_prompt = _build_prompt(cfg, 1, round_num, total_rounds, history, p_last_msg)
    o_raw = _generate(o_model, o_tok, o_dev, o_prompt)
    o_msg, o_action = _parse_response(o_raw, action_re, cfg)
    if p_action is None or o_action is None:
        return {
            "round": round_num,
            "p0_raw": p_raw, "p0_msg": p_msg, "p0_action": p_action,
            "p1_raw": o_raw, "p1_msg": o_msg, "p1_action": o_action,
            "parse_miss": True, "p0_payoff": None, "p1_payoff": None,
        }
    p_pay, o_pay = cfg.payoff_fn(p_action, o_action)
    return {
        "round": round_num,
        "p0_raw": p_raw, "p0_msg": p_msg, "p0_action": p_action,
        "p1_raw": o_raw, "p1_msg": o_msg, "p1_action": o_action,
        "parse_miss": False, "p0_payoff": p_pay, "p1_payoff": o_pay,
    }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--lora-path", default=None,
                   help="Optional PEFT/LoRA adapter dir merged into player+opponent")
    p.add_argument("--game", default=_DEFAULT_GAME,
                   help=f"registered 2P game key (default: {_DEFAULT_GAME})")
    p.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    p.add_argument("--opponent", choices=("self",), default="self",
                   help="self = both seats share weights")
    p.add_argument("--transcript", required=True,
                   help="JSONL output path; one row per round")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _resolve_game(args.game)
    action_re = _build_action_re(cfg.actions)
    print(f"[run] device={_device()} model={args.model} game={args.game} "
          f"actions={cfg.actions} rounds={args.rounds} "
          f"lora={args.lora_path or '(none)'}", flush=True)
    t0 = time.time()
    p_model, p_tok, p_dev = _load(args.model, args.lora_path)
    print(f"[run] player loaded in {time.time() - t0:.1f}s", flush=True)
    o_model, o_tok, o_dev = p_model, p_tok, p_dev

    history: list[dict] = []
    p_last_msg: Optional[str] = None
    o_last_msg: Optional[str] = None
    p_total = 0.0
    o_total = 0.0
    with open(args.transcript, "w", encoding="utf-8") as fh:
        for r in range(1, args.rounds + 1):
            row = _play_round(
                cfg, action_re,
                p_model, p_tok, p_dev, o_model, o_tok, o_dev,
                round_num=r, total_rounds=args.rounds, history=history,
                p_last_msg=p_last_msg, o_last_msg=o_last_msg,
            )
            history.append(row)
            p_last_msg = row["p0_msg"]
            o_last_msg = row["p1_msg"]
            if not row["parse_miss"]:
                p_total += row["p0_payoff"]
                o_total += row["p1_payoff"]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            preview_p = row["p0_msg"].replace("\n", " ")[:_MSG_PREVIEW_CHARS]
            preview_o = row["p1_msg"].replace("\n", " ")[:_MSG_PREVIEW_CHARS]
            tag = "MISS" if row["parse_miss"] else "    "
            print(f"[r{r:02d}] {tag} P0 {str(row['p0_action']):9s} :: {preview_p!r}",
                  flush=True)
            print(f"        {tag} P1 {str(row['p1_action']):9s} :: {preview_o!r}",
                  flush=True)
    print(f"[run] totals  P0={p_total:.1f}  P1={o_total:.1f}  "
          f"rounds={args.rounds}  wallclock={time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
