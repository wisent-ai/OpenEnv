"""Free-form natural-language Prisoner's Dilemma between two LLM seats.

Bypasses the env loop because no registered game supports passing a
free-form text string from one agent into another agent's next prompt;
cheap_talk_pd's "messages" are bit-encoded into a 4-token action
vocabulary (msg_<say>_<do>). This driver instead runs both seats from
the same generate_fn (or two), prompts each per round to emit a free
text message followed by an ACTION line, and renders the verbatim
message into the opponent's next prompt.

Writes a JSONL transcript with every (round, seat, raw, message,
action, payoff). Loads optional LoRA adapter for the player model.

Run:
    PYTHONPATH=. HF_TOKEN=... python3 \\
        scripts/free_chat/run_free_chat_pd.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --rounds 10 --opponent self \\
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

_PD_PAYOFF: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (3.0, 3.0),
    ("cooperate", "defect"):    (0.0, 5.0),
    ("defect",    "cooperate"): (5.0, 0.0),
    ("defect",    "defect"):    (1.0, 1.0),
}
_ACTION_RE = re.compile(r"ACTION\s*[:\-]\s*(cooperate|defect)", re.IGNORECASE)
_HISTORY_TAIL = 3
_MSG_PREVIEW_CHARS = 80
_MAX_NEW_TOKENS = 96
_TEMPERATURE = 0.7
_DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


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


def _build_prompt(my_idx: int, round_num: int, total_rounds: int,
                  history: list[dict], opp_last_msg: Optional[str]) -> str:
    me = f"P{my_idx}"
    opp = f"P{1 - my_idx}"
    me_key = me.lower()
    opp_key = opp.lower()
    lines = [
        "[Game] Repeated Prisoner's Dilemma.",
        f"You are {me}. Your opponent is {opp}.",
        f"Round {round_num} of {total_rounds}.",
        "[Payoffs] (cooperate, cooperate)=(3,3); (cooperate, defect)=(0,5);",
        "          (defect, cooperate)=(5,0); (defect, defect)=(1,1).",
        "[Format] Write ONE short message to your opponent (1-2 sentences).",
        "         Then on a new line write exactly: ACTION: cooperate",
        "         or:                              ACTION: defect",
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


def _parse_response(text: str) -> tuple[str, Optional[str]]:
    """Return (message_text, action_or_None). Action is taken from the
    first 'ACTION: <cooperate|defect>' match; the message is everything
    before that match."""
    m = _ACTION_RE.search(text)
    if not m:
        return text, None
    action = m.group(1).lower()
    msg = text[: m.start()].strip()
    return msg, action


def _play_round(p_model, p_tok, p_dev, o_model, o_tok, o_dev, *,
                round_num: int, total_rounds: int, history: list[dict],
                p_last_msg: Optional[str], o_last_msg: Optional[str]) -> dict:
    p_prompt = _build_prompt(0, round_num, total_rounds, history, o_last_msg)
    p_raw = _generate(p_model, p_tok, p_dev, p_prompt)
    p_msg, p_action = _parse_response(p_raw)
    o_prompt = _build_prompt(1, round_num, total_rounds, history, p_last_msg)
    o_raw = _generate(o_model, o_tok, o_dev, o_prompt)
    o_msg, o_action = _parse_response(o_raw)
    p_a = p_action or "defect"
    o_a = o_action or "defect"
    p_pay, o_pay = _PD_PAYOFF[(p_a, o_a)]
    return {
        "round": round_num,
        "p0_raw": p_raw, "p0_msg": p_msg, "p0_action": p_a,
        "p0_parsed": p_action is not None,
        "p1_raw": o_raw, "p1_msg": o_msg, "p1_action": o_a,
        "p1_parsed": o_action is not None,
        "p0_payoff": p_pay, "p1_payoff": o_pay,
    }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--lora-path", default=None,
                   help="Optional PEFT/LoRA adapter dir merged into player+opponent")
    p.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    p.add_argument("--opponent", choices=("self",), default="self",
                   help="self = both seats share weights")
    p.add_argument("--transcript", required=True,
                   help="JSONL output path; one row per round")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[run] device={_device()} model={args.model} "
          f"lora={args.lora_path or '(none)'} rounds={args.rounds}",
          flush=True)
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
                p_model, p_tok, p_dev, o_model, o_tok, o_dev,
                round_num=r, total_rounds=args.rounds, history=history,
                p_last_msg=p_last_msg, o_last_msg=o_last_msg,
            )
            history.append(row)
            p_last_msg = row["p0_msg"]
            o_last_msg = row["p1_msg"]
            p_total += row["p0_payoff"]
            o_total += row["p1_payoff"]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            preview_p = row["p0_msg"].replace("\n", " ")[:_MSG_PREVIEW_CHARS]
            preview_o = row["p1_msg"].replace("\n", " ")[:_MSG_PREVIEW_CHARS]
            print(f"[r{r:02d}] P0 {row['p0_action']:9s} :: {preview_p!r}",
                  flush=True)
            print(f"        P1 {row['p1_action']:9s} :: {preview_o!r}",
                  flush=True)
    print(f"[run] totals  P0={p_total:.1f}  P1={o_total:.1f}  "
          f"rounds={args.rounds}  wallclock={time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
