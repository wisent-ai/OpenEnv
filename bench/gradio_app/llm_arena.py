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
              f"| P{_ONE} Pay | P{_TWO} Pay |")
_SEP_ROUND = "|-------|-----------|-----------|--------|--------|"


def _call_llm(provider, model, prompt, api_key):
    """Call an LLM provider and return raw text."""
    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=_MAX_TOKENS, system=_SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[_ZERO].text
    if provider == "OpenAI":
        import openai
        client = openai.OpenAI(api_key=api_key)
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


def _resolve_key(provider, manual_key):
    """Use manual key if provided, otherwise try OAuth."""
    if manual_key and manual_key.strip():
        return manual_key.strip()
    return get_oauth_token(provider)


def _init_matchups(models, anthropic_key, openai_key):
    """Build initial matchup state for all pairs."""
    matchups = []
    for i in range(len(models)):
        for j in range(i + _ONE, len(models)):
            p1, p2 = models[i], models[j]
            p1_prov, p2_prov = _model_provider(p1), _model_provider(p2)
            p1_key = _resolve_key(p1_prov,
                                  anthropic_key if p1_prov == "Anthropic" else openai_key)
            p2_key = _resolve_key(p2_prov,
                                  anthropic_key if p2_prov == "Anthropic" else openai_key)
            if not p1_key or not p2_key:
                continue
            matchups.append({
                "p1_label": f"{p1_prov}/{p1}", "p2_label": f"{p2_prov}/{p2}",
                "p1_prov": p1_prov, "p1_model": p1, "p1_key": p1_key,
                "p2_prov": p2_prov, "p2_model": p2, "p2_key": p2_key,
                "p1_hist": [], "p2_hist": [],
                "p1_score": float(), "p2_score": float(),
                "recent": [],
            })
    return matchups


def run_infinite_tournament(game_name, variants, models,
                            anthropic_key, openai_key):
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
    matchups = _init_matchups(models, anthropic_key, openai_key)
    if not matchups:
        yield "No valid matchups -- provide API keys or enable OAuth."
        return
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
                raw1 = _call_llm(m["p1_prov"], m["p1_model"], prompt1, m["p1_key"])
                act1 = parse_action(raw1, actions)
            except Exception:
                act1 = _rand.choice(actions)
            try:
                raw2 = _call_llm(m["p2_prov"], m["p2_model"], prompt2, m["p2_key"])
                act2 = parse_action(raw2, actions)
            except Exception:
                act2 = _rand.choice(actions)
            p1_pay, p2_pay = info["payoff_fn"](act1, act2)
            m["p1_score"] += p1_pay
            m["p2_score"] += p2_pay
            m["p1_hist"].append({"round": rnd, "action": act1, "payoff": p1_pay})
            m["p2_hist"].append({"round": rnd, "action": act2, "payoff": p2_pay})
            m["recent"].append({"round": rnd, "p1_action": act1, "p2_action": act2,
                                "p1_pay": p1_pay, "p2_pay": p2_pay})
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
        lines.append(f"| {i + _ONE} | {m['p1_label']} | {m['p2_label']} | "
                     f"{m['p1_score']:.1f} | {m['p2_score']:.1f} | {leader} |")
    for i, m in enumerate(matchups):
        recent = m["recent"]
        lines.extend([
            f"\n### Match {i + _ONE}: {m['p1_label']} vs {m['p2_label']} "
            f"(last {len(recent)} rounds)\n",
            _HDR_ROUND, _SEP_ROUND])
        for rd in recent:
            lines.append(
                f"| {rd['round']} | {rd['p1_action']} | {rd['p2_action']} | "
                f"{rd['p1_pay']:.1f} | {rd['p2_pay']:.1f} |")
    return "\n".join(lines)
