"""LLM Arena -- infinite spectator tournament."""
from __future__ import annotations
import random as _rand

from registry import (
    _ZERO, _ONE, _TWO, _TEN,
    _HAS_LLM_AGENT, _LLM_MODELS,
    PromptBuilder, parse_action, GameObservation, RoundResult,
    _SYS_PROMPT, get_oauth_token,
)
from callbacks import _get_game_info

_MAX_TOKENS = _TEN + _TEN
_DETAIL_LIMIT = _TEN + _TEN
_HISTORY_WINDOW = _TEN * _TEN
_INF_HORIZON = _TEN * _TEN * _TEN * _TEN

_HDR_MATCH = (f"| Match | Player {_ONE} | Player {_TWO} "
              f"| P{_ONE} Score | P{_TWO} Score | Leader |")
_SEP_MATCH = "|-------|----------|----------|----------|----------|--------|"
_HDR_ROUND = (f"| Round | P{_ONE} Action | P{_TWO} Action "
              f"| P{_ONE} Pay | P{_TWO} Pay | Rules |")
_SEP_ROUND = "|-------|-----------|-----------|--------|--------|-------|"

_CONST_PREFIX = "const"
_EXIT_ACTION = "exit"


def _parse_rule_status(p1_action, p2_action, locked_rule):
    """Parse actions and return (p1_base, p2_base, rule_status_str, new_locked_rule)."""
    sep = "_"
    p1_rule, p2_rule = "", ""
    p1_base, p2_base = p1_action, p2_action

    if p1_action == _EXIT_ACTION:
        p1_base = _EXIT_ACTION
    elif p1_action.startswith(_CONST_PREFIX + sep):
        parts = p1_action.split(sep, _TWO + _ONE)
        if len(parts) >= _TWO + _ONE:
            p1_rule = parts[_ONE]
            p1_base = parts[_TWO]

    if p2_action == _EXIT_ACTION:
        p2_base = _EXIT_ACTION
    elif p2_action.startswith(_CONST_PREFIX + sep):
        parts = p2_action.split(sep, _TWO + _ONE)
        if len(parts) >= _TWO + _ONE:
            p2_rule = parts[_ONE]
            p2_base = parts[_TWO]

    new_locked = locked_rule
    if locked_rule:
        status = f"LOCKED: {locked_rule}"
    elif p1_rule and p2_rule:
        if p1_rule == p2_rule and p1_rule != "none":
            status = f"AGREED: {p1_rule}"
            new_locked = p1_rule
        else:
            status = f"{p1_rule} vs {p2_rule}"
    elif p1_rule or p2_rule:
        status = f"{p1_rule or '-'} vs {p2_rule or '-'}"
    else:
        status = ""

    return p1_base, p2_base, status, new_locked


def _call_llm(provider, model, prompt):
    """Call an LLM provider using OAuth tokens and return raw text."""
    token = get_oauth_token(provider)
    if not token:
        raise RuntimeError(f"OAuth token unavailable for {provider}")
    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=token)
        resp = client.messages.create(
            model=model, max_tokens=_MAX_TOKENS, system=_SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[_ZERO].text
    if provider == "OpenAI":
        import openai
        client = openai.OpenAI(api_key=token)
        resp = client.chat.completions.create(
            model=model, max_tokens=_MAX_TOKENS,
            messages=[{"role": "system", "content": _SYS_PROMPT},
                      {"role": "user", "content": prompt}])
        return resp.choices[_ZERO].message.content
    return ""


def _build_obs(info, p_hist, o_hist, rnd, p_score, o_score):
    """Build GameObservation for one player in infinite mode."""
    history = []
    for ph, oh in zip(p_hist[-_HISTORY_WINDOW:], o_hist[-_HISTORY_WINDOW:]):
        history.append(RoundResult(
            round_number=ph["round"],
            player_action=ph["action"], opponent_action=oh["action"],
            player_payoff=ph["payoff"], opponent_payoff=oh["payoff"]))
    return GameObservation(
        game_name=info.get("key", ""),
        game_description=info.get("description", ""),
        available_actions=info["actions"], current_round=rnd,
        total_rounds=_INF_HORIZON, history=history,
        player_score=p_score, opponent_score=o_score,
        opponent_strategy="llm")


def _model_provider(model_name):
    """Determine provider from model name."""
    for prov, models in _LLM_MODELS.items():
        if model_name in models:
            return prov
    return "Anthropic"


def _init_matchups(models):
    """Build initial matchup state for all pairs."""
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
                "recent": [], "locked_rule": "",
            })
    return matchups


def run_infinite_tournament(game_name, variants, models):
    """Generator that runs forever, yielding markdown after each round."""
    if len(models) < _TWO:
        yield "Select at least two models."
        return
    if not _HAS_LLM_AGENT:
        yield "LLM agent not available."
        return
    info = _get_game_info(game_name, variants)
    if not info:
        yield "Game not found."
        return
    actions = info["actions"]
    matchups = _init_matchups(models)
    rnd = _ZERO
    while True:
        rnd += _ONE
        for m in matchups:
            obs1 = _build_obs(info, m["p1_hist"], m["p2_hist"],
                              rnd, m["p1_score"], m["p2_score"])
            obs2 = _build_obs(info, m["p2_hist"], m["p1_hist"],
                              rnd, m["p2_score"], m["p1_score"])
            prompt1 = PromptBuilder.build(obs1)
            prompt2 = PromptBuilder.build(obs2)
            try:
                raw1 = _call_llm(m["p1_prov"], m["p1_model"], prompt1)
                act1 = parse_action(raw1, actions)
            except Exception:
                act1 = _rand.choice(actions)
            try:
                raw2 = _call_llm(m["p2_prov"], m["p2_model"], prompt2)
                act2 = parse_action(raw2, actions)
            except Exception:
                act2 = _rand.choice(actions)
            p1_pay, p2_pay = info["payoff_fn"](act1, act2)
            m["p1_score"] += p1_pay
            m["p2_score"] += p2_pay
            p1_base, p2_base, rule_status, new_locked = _parse_rule_status(
                act1, act2, m.get("locked_rule", ""))
            m["locked_rule"] = new_locked
            m["p1_hist"].append({"round": rnd, "action": act1, "payoff": p1_pay})
            m["p2_hist"].append({"round": rnd, "action": act2, "payoff": p2_pay})
            m["recent"].append({"round": rnd, "p1_action": p1_base, "p2_action": p2_base,
                                "p1_pay": p1_pay, "p2_pay": p2_pay,
                                "rule_status": rule_status})
            if len(m["recent"]) > _DETAIL_LIMIT:
                m["recent"] = m["recent"][-_DETAIL_LIMIT:]
            if len(m["p1_hist"]) > _HISTORY_WINDOW:
                m["p1_hist"] = m["p1_hist"][-_HISTORY_WINDOW:]
                m["p2_hist"] = m["p2_hist"][-_HISTORY_WINDOW:]
        yield _render_state(matchups, rnd)


def _render_state(matchups, current_round):
    """Render current infinite tournament state as markdown."""
    lines = [f"## Infinite Tournament -- Round {current_round}\n"]
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
    lines.extend(["\n### Matchups\n", _HDR_MATCH, _SEP_MATCH])
    for i, m in enumerate(matchups):
        leader = m["p1_label"] if m["p1_score"] > m["p2_score"] else (
            m["p2_label"] if m["p2_score"] > m["p1_score"] else "Tied")
        locked = m.get("locked_rule", "")
        rule_col = f" **{locked}**" if locked else " negotiating..."
        lines.append(f"| {i + _ONE} | {m['p1_label']} | {m['p2_label']} | "
                     f"{m['p1_score']:.1f} | {m['p2_score']:.1f} | {leader} |")
    for i, m in enumerate(matchups):
        recent = m["recent"]
        locked = m.get("locked_rule", "")
        rule_note = f" -- Rule: **{locked}**" if locked else ""
        lines.extend([
            f"\n### Match {i + _ONE}: {m['p1_label']} vs {m['p2_label']} "
            f"(last {len(recent)} rounds){rule_note}\n",
            _HDR_ROUND, _SEP_ROUND])
        for rd in recent:
            rule_str = rd.get("rule_status", "")
            lines.append(
                f"| {rd['round']} | {rd['p1_action']} | {rd['p2_action']} | "
                f"{rd['p1_pay']:.1f} | {rd['p2_pay']:.1f} | {rule_str} |")
    return "\n".join(lines)
