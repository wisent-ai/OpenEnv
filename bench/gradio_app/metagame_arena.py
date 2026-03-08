"""Metagame Arena -- full 5-phase arena with messaging between models."""
from __future__ import annotations
import random as _rand

from registry import (
    _ZERO, _ONE, _TWO, _TEN,
    _HAS_LLM_AGENT, _LLM_MODELS,
    PromptBuilder, parse_action, GameObservation, RoundResult,
    _SYS_PROMPT, get_env_api_key,
)

_MAX_TOKENS = _TEN * _TEN
_HISTORY_WINDOW = _TEN
_COMM_PROMPT_TEMPLATE = (
    "You are in a multi-round game theory arena. "
    "Before playing, you may send a message to your opponent. "
    "This message is free-form text (max 500 chars). Use it to "
    "negotiate, threaten, promise, or coordinate.\n\n"
    "Game: {game_name}\n"
    "Round: {round_num}\n"
    "Your score: {your_score:.1f} | Opponent score: {opp_score:.1f}\n"
    "{history_section}"
    "{incoming_messages}"
    "\nRespond with ONLY your message to the opponent. "
    "If you don't want to send a message, respond with: PASS"
)

_PLAY_PROMPT_TEMPLATE = (
    "You are playing a game theory game.\n\n"
    "Game: {game_name}\n"
    "Description: {description}\n"
    "Round: {round_num}\n"
    "Your score: {your_score:.1f} | Opponent score: {opp_score:.1f}\n"
    "Available actions: {actions}\n"
    "{history_section}"
    "{message_section}"
    "\nChoose your action. Respond with ONLY the action name."
)


def _model_provider(model_name):
    for prov, models in _LLM_MODELS.items():
        if model_name in models:
            return prov
    return "OpenAI"


def _call_llm(provider, model, prompt, max_tokens=None):
    token = get_env_api_key(provider)
    if not token:
        raise RuntimeError(f"No API key for {provider}")
    mt = max_tokens or _MAX_TOKENS
    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=token)
        resp = client.messages.create(
            model=model, max_tokens=mt, system=_SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[_ZERO].text
    if provider == "OpenAI":
        import openai
        client = openai.OpenAI(api_key=token)
        resp = client.chat.completions.create(
            model=model, max_tokens=mt,
            messages=[{"role": "system", "content": _SYS_PROMPT},
                      {"role": "user", "content": prompt}])
        return resp.choices[_ZERO].message.content
    return ""


def _format_history(p_hist, o_hist, limit=None):
    lim = limit or _HISTORY_WINDOW
    lines = []
    for ph, oh in zip(p_hist[-lim:], o_hist[-lim:]):
        lines.append(
            f"  Round {ph['round']}: You={ph['action']} Opp={oh['action']} "
            f"Pay={ph['payoff']:.1f}")
    if lines:
        return "\nRecent history:\n" + "\n".join(lines) + "\n"
    return ""


def _format_messages(msgs):
    if not msgs:
        return ""
    lines = ["\nMessages from opponent:"]
    for m in msgs:
        lines.append(f"  [Round {m['round']}] {m['content']}")
    return "\n".join(lines) + "\n"


def _init_matchups(models):
    matchups = []
    for i in range(len(models)):
        for j in range(i + _ONE, len(models)):
            p1, p2 = models[i], models[j]
            p1_prov, p2_prov = _model_provider(p1), _model_provider(p2)
            matchups.append({
                "p1_label": f"{p1_prov}/{p1}", "p2_label": f"{p2_prov}/{p2}",
                "p1_prov": p1_prov, "p1_model": p1,
                "p2_prov": p2_prov, "p2_model": p2,
                "p1_hist": [], "p2_hist": [],
                "p1_score": float(), "p2_score": float(),
                "p1_msgs": [], "p2_msgs": [],
                "recent": [],
            })
    return matchups


def run_metagame_arena(game_name, game_desc, actions, payoff_fn, models):
    """Generator: communication + play each round, yield markdown."""
    if len(models) < _TWO:
        yield "Select at least two models."
        return
    matchups = _init_matchups(models)
    if not matchups:
        yield "No valid matchups."
        return
    rnd = _ZERO
    while True:
        rnd += _ONE
        for m in matchups:
            # Phase 1: Communication
            p1_incoming = [msg for msg in m["p2_msgs"][-_HISTORY_WINDOW:]]
            p2_incoming = [msg for msg in m["p1_msgs"][-_HISTORY_WINDOW:]]

            hist_p1 = _format_history(m["p1_hist"], m["p2_hist"])
            hist_p2 = _format_history(m["p2_hist"], m["p1_hist"])

            comm_prompt_p1 = _COMM_PROMPT_TEMPLATE.format(
                game_name=game_name, round_num=rnd,
                your_score=m["p1_score"], opp_score=m["p2_score"],
                history_section=hist_p1,
                incoming_messages=_format_messages(p1_incoming))
            comm_prompt_p2 = _COMM_PROMPT_TEMPLATE.format(
                game_name=game_name, round_num=rnd,
                your_score=m["p2_score"], opp_score=m["p1_score"],
                history_section=hist_p2,
                incoming_messages=_format_messages(p2_incoming))

            try:
                msg1 = _call_llm(m["p1_prov"], m["p1_model"], comm_prompt_p1).strip()
                if msg1.upper() == "PASS":
                    msg1 = ""
            except Exception:
                msg1 = ""
            try:
                msg2 = _call_llm(m["p2_prov"], m["p2_model"], comm_prompt_p2).strip()
                if msg2.upper() == "PASS":
                    msg2 = ""
            except Exception:
                msg2 = ""

            # Truncate to 500 chars
            msg1 = msg1[:_TEN * _TEN * _HISTORY_WINDOW] if msg1 else ""
            msg2 = msg2[:_TEN * _TEN * _HISTORY_WINDOW] if msg2 else ""

            if msg1:
                m["p1_msgs"].append({"round": rnd, "content": msg1})
            if msg2:
                m["p2_msgs"].append({"round": rnd, "content": msg2})

            # Phase 2: Play
            # P1 sees P2's message from this round, and vice versa
            this_round_p1_msgs = [{"round": rnd, "content": msg2}] if msg2 else []
            this_round_p2_msgs = [{"round": rnd, "content": msg1}] if msg1 else []
            all_p1_msgs = p1_incoming + this_round_p1_msgs
            all_p2_msgs = p2_incoming + this_round_p2_msgs

            play_prompt_p1 = _PLAY_PROMPT_TEMPLATE.format(
                game_name=game_name, description=game_desc,
                round_num=rnd,
                your_score=m["p1_score"], opp_score=m["p2_score"],
                actions=", ".join(actions),
                history_section=hist_p1,
                message_section=_format_messages(all_p1_msgs[-_HISTORY_WINDOW:]))
            play_prompt_p2 = _PLAY_PROMPT_TEMPLATE.format(
                game_name=game_name, description=game_desc,
                round_num=rnd,
                your_score=m["p2_score"], opp_score=m["p1_score"],
                actions=", ".join(actions),
                history_section=hist_p2,
                message_section=_format_messages(all_p2_msgs[-_HISTORY_WINDOW:]))

            try:
                raw1 = _call_llm(m["p1_prov"], m["p1_model"], play_prompt_p1)
                act1 = parse_action(raw1, actions)
            except Exception:
                act1 = _rand.choice(actions)
            try:
                raw2 = _call_llm(m["p2_prov"], m["p2_model"], play_prompt_p2)
                act2 = parse_action(raw2, actions)
            except Exception:
                act2 = _rand.choice(actions)

            p1_pay, p2_pay = payoff_fn(act1, act2)
            m["p1_score"] += p1_pay
            m["p2_score"] += p2_pay
            m["p1_hist"].append({"round": rnd, "action": act1, "payoff": p1_pay})
            m["p2_hist"].append({"round": rnd, "action": act2, "payoff": p2_pay})
            m["recent"].append({
                "round": rnd, "p1_action": act1, "p2_action": act2,
                "p1_pay": p1_pay, "p2_pay": p2_pay,
                "p1_msg": msg1, "p2_msg": msg2,
            })
            if len(m["recent"]) > _TEN + _TEN:
                m["recent"] = m["recent"][-(_TEN + _TEN):]
            if len(m["p1_hist"]) > _TEN * _TEN:
                m["p1_hist"] = m["p1_hist"][-(_TEN * _TEN):]
                m["p2_hist"] = m["p2_hist"][-(_TEN * _TEN):]
            if len(m["p1_msgs"]) > _TEN * _TEN:
                m["p1_msgs"] = m["p1_msgs"][-(_TEN * _TEN):]
                m["p2_msgs"] = m["p2_msgs"][-(_TEN * _TEN):]

        yield _render_metagame(matchups, rnd)


def _render_metagame(matchups, current_round):
    lines = [f"## Metagame Arena -- Round {current_round}\n"]

    # Leaderboard
    scores = {}
    for m in matchups:
        scores.setdefault(m["p1_label"], float())
        scores.setdefault(m["p2_label"], float())
        scores[m["p1_label"]] += m["p1_score"]
        scores[m["p2_label"]] += m["p2_score"]
    lines.extend(["### Leaderboard\n",
                  "| Rank | Model | Total Score | Avg / Round |",
                  "|------|-------|-------------|-------------|"])
    for rank, (model, score) in enumerate(
            sorted(scores.items(), key=lambda x: -x[_ONE])):
        avg = score / max(current_round, _ONE)
        lines.append(f"| {rank + _ONE} | {model} | {score:.1f} | {avg:.2f} |")

    # Per-matchup details with messages
    for i, m in enumerate(matchups):
        recent = m["recent"]
        lines.append(
            f"\n### Match {i + _ONE}: {m['p1_label']} vs {m['p2_label']} "
            f"(last {len(recent)} rounds)\n")

        for rd in recent:
            lines.append(f"**Round {rd['round']}**")
            if rd.get("p1_msg"):
                lines.append(f"> **{m['p1_label']}** says: *{rd['p1_msg']}*")
            if rd.get("p2_msg"):
                lines.append(f"> **{m['p2_label']}** says: *{rd['p2_msg']}*")
            lines.append(
                f"| Action | Payoff |\n|--------|--------|\n"
                f"| {m['p1_label']}: **{rd['p1_action']}** | {rd['p1_pay']:.1f} |\n"
                f"| {m['p2_label']}: **{rd['p2_action']}** | {rd['p2_pay']:.1f} |")
            lines.append("")

    return "\n".join(lines)
