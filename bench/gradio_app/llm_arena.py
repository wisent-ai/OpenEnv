"""LLM Arena -- spectator round-robin for Infinite Mode."""
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

_HDR_MATCH = f"| Match | Player {_ONE} | Player {_TWO} | P{_ONE} Score | P{_TWO} Score | Winner |"
_SEP_MATCH = "|-------|----------|----------|----------|----------|--------|"
_HDR_ROUND = f"| Round | P{_ONE} Action | P{_TWO} Action | P{_ONE} Pay | P{_TWO} Pay |"
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


def _build_obs(info, p_hist, o_hist, rnd, total, p_score, o_score):
    """Build GameObservation for one player."""
    history = []
    for ph, oh in zip(p_hist, o_hist):
        history.append(RoundResult(
            round_number=ph["round"],
            player_action=ph["action"], opponent_action=oh["action"],
            player_payoff=ph["payoff"], opponent_payoff=oh["payoff"]))
    return GameObservation(
        game_name=info.get("key", ""), game_description=info.get("description", ""),
        available_actions=info["actions"], current_round=rnd,
        total_rounds=total, history=history,
        player_score=p_score, opponent_score=o_score,
        opponent_strategy="llm")


def _model_provider(model_name):
    """Determine provider from model name."""
    for prov, models in _LLM_MODELS.items():
        if model_name in models:
            return prov
    return "Anthropic"


def run_match(game_name, variants, num_rounds,
              p1_prov, p1_model, p1_key, p2_prov, p2_model, p2_key):
    """Run a full match between two LLMs. Returns result dict."""
    if not _HAS_LLM_AGENT:
        return {"error": "LLM agent not available"}
    info = _get_game_info(game_name, variants)
    if not info:
        return {"error": f"Game not found: {game_name}"}
    actions = info["actions"]
    p1_hist, p2_hist = [], []
    p1_score, p2_score = float(), float()
    rounds = []
    for rnd in range(_ONE, num_rounds + _ONE):
        obs1 = _build_obs(info, p1_hist, p2_hist, rnd, num_rounds, p1_score, p2_score)
        obs2 = _build_obs(info, p2_hist, p1_hist, rnd, num_rounds, p2_score, p1_score)
        prompt1, prompt2 = PromptBuilder.build(obs1), PromptBuilder.build(obs2)
        try:
            raw1 = _call_llm(p1_prov, p1_model, prompt1, p1_key)
            act1 = parse_action(raw1, actions)
        except Exception as exc:
            act1, raw1 = _rand.choice(actions), f"ERROR: {exc}"
        try:
            raw2 = _call_llm(p2_prov, p2_model, prompt2, p2_key)
            act2 = parse_action(raw2, actions)
        except Exception as exc:
            act2, raw2 = _rand.choice(actions), f"ERROR: {exc}"
        p1_pay, p2_pay = info["payoff_fn"](act1, act2)
        p1_score += p1_pay
        p2_score += p2_pay
        p1_hist.append({"round": rnd, "action": act1, "payoff": p1_pay})
        p2_hist.append({"round": rnd, "action": act2, "payoff": p2_pay})
        rounds.append({"round": rnd, "p1_action": act1, "p2_action": act2,
                        "p1_pay": p1_pay, "p2_pay": p2_pay,
                        "p1_raw": raw1.strip(), "p2_raw": raw2.strip()})
    return {"p1": f"{p1_prov}/{p1_model}", "p2": f"{p2_prov}/{p2_model}",
            "p1_score": p1_score, "p2_score": p2_score,
            "rounds": rounds, "total_rounds": num_rounds}


def _resolve_key(provider, manual_key):
    """Use manual key if provided, otherwise try OAuth."""
    if manual_key and manual_key.strip():
        return manual_key.strip()
    return get_oauth_token(provider)


def run_tournament(game_name, variants, num_rounds, models,
                   anthropic_key, openai_key):
    """Run round-robin tournament between selected models."""
    if len(models) < _TWO:
        return [], "Select at least two models."
    results = []
    for i in range(len(models)):
        for j in range(i + _ONE, len(models)):
            p1, p2 = models[i], models[j]
            p1_prov, p2_prov = _model_provider(p1), _model_provider(p2)
            p1_key = _resolve_key(p1_prov, anthropic_key if p1_prov == "Anthropic" else openai_key)
            p2_key = _resolve_key(p2_prov, anthropic_key if p2_prov == "Anthropic" else openai_key)
            if not p1_key or not p2_key:
                results.append({"error": "No OAuth token or API key available"})
                continue
            result = run_match(game_name, variants, num_rounds,
                               p1_prov, p1, p1_key, p2_prov, p2, p2_key)
            results.append(result)
    return results, ""


def render_tournament(results):
    """Render tournament results as markdown."""
    if not results:
        return "No results yet. Select models and run the tournament."
    lines = ["## Tournament Results\n", _HDR_MATCH, _SEP_MATCH]
    for i, r in enumerate(results):
        if "error" in r:
            lines.append(f"| {i + _ONE} | - | - | - | - | Error: {r['error']} |")
            continue
        winner = r["p1"] if r["p1_score"] > r["p2_score"] else (
            r["p2"] if r["p2_score"] > r["p1_score"] else "Draw")
        lines.append(f"| {i + _ONE} | {r['p1']} | {r['p2']} | "
                     f"{r['p1_score']:.1f} | {r['p2_score']:.1f} | {winner} |")
    scores = {}
    for r in results:
        if "error" in r:
            continue
        scores.setdefault(r["p1"], float())
        scores.setdefault(r["p2"], float())
        scores[r["p1"]] += r["p1_score"]
        scores[r["p2"]] += r["p2_score"]
    if scores:
        lines.extend(["\n## Leaderboard\n",
                      "| Rank | Model | Total Score |",
                      "|------|-------|-------------|"])
        for rank, (model, score) in enumerate(
                sorted(scores.items(), key=lambda x: -x[_ONE])):
            lines.append(f"| {rank + _ONE} | {model} | {score:.1f} |")
    for i, r in enumerate(results):
        if "error" in r or not r.get("rounds"):
            continue
        lines.extend([f"\n### Match {i + _ONE}: {r['p1']} vs {r['p2']}\n",
                      _HDR_ROUND, _SEP_ROUND])
        for rd in r["rounds"][:_DETAIL_LIMIT]:
            lines.append(f"| {rd['round']} | {rd['p1_action']} | {rd['p2_action']} | "
                         f"{rd['p1_pay']:.1f} | {rd['p2_pay']:.1f} |")
        if len(r["rounds"]) > _DETAIL_LIMIT:
            remaining = len(r["rounds"]) - _DETAIL_LIMIT
            lines.append(f"| ... | ({remaining} more rounds) | ... | ... | ... |")
    return "\n".join(lines)
