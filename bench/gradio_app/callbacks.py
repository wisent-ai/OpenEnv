"""State management, callbacks, and reference builder for the Kant Gradio app."""
from __future__ import annotations
import random as _rand
import gradio as gr

from registry import (
    _ZERO, _ONE, _TWO, _FOUR, _TEN,
    DEFAULT_NUM_ROUNDS,
    _HAS_REGISTRY, _HAS_VARIANTS, _HAS_NPLAYER_ENV, _HAS_FULL_STRATEGIES,
    _HAS_LLM_AGENT,
    _GAME_INFO, _KEY_TO_NAME, _CATEGORY_DIMS, _ALL_FILTER,
    compose_game, get_games_by_tag,
    STRATEGIES_2P, _strategies_for_game, _NPLAYER_STRAT_NAMES,
    _filter_game_names, _filter_by_mp,
    _HUMAN_VARIANTS, _2P_ONLY_VARIANTS,
    _GENERIC_STRATEGIES, _GAME_TYPE_STRATEGIES,
    NPlayerEnvironment, NPlayerAction,
    PromptBuilder, parse_action, GameObservation, RoundResult,
    _SYS_PROMPT, _LLM_OPPONENT_LABEL, _LLM_MODELS,
    _HAS_OAUTH, get_oauth_token,
)


def _get_game_info(gname, variants=None):
    base_info = _GAME_INFO.get(gname)
    if not base_info or not variants or not _HAS_VARIANTS:
        return base_info
    try:
        cfg = compose_game(base_info["key"], *variants)
        return {"actions": cfg.actions, "description": cfg.description,
                "payoff_fn": cfg.payoff_fn, "default_rounds": cfg.default_rounds,
                "key": base_info["key"], "num_players": cfg.num_players,
                "game_type": cfg.game_type, "opponent_actions": cfg.opponent_actions}
    except (KeyError, ValueError):
        return base_info


def _blank(gname, sname, variants=None, max_rounds=None):
    info = _get_game_info(gname, variants) or {}
    np = info.get("num_players", _TWO)
    mr = max_rounds if max_rounds is not None else info.get("default_rounds", DEFAULT_NUM_ROUNDS)
    return {"game": gname, "strategy": sname, "history": [], "llm_log": [],
            "p_score": _ZERO, "o_score": _ZERO, "round": _ZERO,
            "max_rounds": mr, "done": False, "num_players": np,
            "scores": [_ZERO] * np, "nplayer_env": None,
            "variants": list(variants or [])}


def _render(st):
    np = st.get("num_players", _TWO)
    is_mp = np > _TWO
    vlist = st.get("variants", [])
    vtag = f"  |  **Variants:** {', '.join(vlist)}" if vlist else ""
    lines = [f"**Game:** {st['game']}  |  **Players:** {np}  |  **Opponent:** {st['strategy']}{vtag}",
             f"**Round:** {st['round']} / {st['max_rounds']}"]
    if is_mp:
        scores = st.get("scores", [])
        lines.append(f"**Scores:** {' | '.join(f'P{i}: {s:.1f}' for i, s in enumerate(scores))}")
    else:
        lines.append(f"**Your score:** {st['p_score']}  |  **Opponent score:** {st['o_score']}")
    if st["done"]:
        lines.append("\n### Game Over")
    if is_mp:
        hc = ["Round"] + [f"P{i}" for i in range(np)] + [f"Pay{i}" for i in range(np)]
        lines.append("\n| " + " | ".join(hc) + " |")
        lines.append("|" + "|".join(["-------"] * len(hc)) + "|")
        for r in st["history"]:
            row = [str(r["round"])] + [str(a) for a in r.get("actions", [])]
            row.extend(f"{p:.1f}" for p in r.get("payoffs", []))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("\n| Round | You | Opponent | Your Pay | Opp Pay |")
        lines.append("|-------|-----|----------|----------|---------|")
        for r in st["history"]:
            lines.append(f"| {r['round']} | {r['player_action']} | "
                         f"{r['opponent_action']} | {r['p_pay']} | {r['o_pay']} |")
    for entry in st.get("llm_log", []):
        lines.append(f"- **Round {entry['round']}**: `{entry['raw']}`")
    return "\n".join(lines)


def _resolve_api_key(provider, api_key):
    """Return an API key: use provided key, or fall back to OAuth."""
    if api_key and api_key.strip():
        return api_key.strip()
    return get_oauth_token(provider)


def _llm_choose_action(state, info, provider, model, api_key):
    """Have the LLM choose an action."""
    if not _HAS_LLM_AGENT:
        return _rand.choice(info["actions"]), "(LLM agent not available)"
    history = []
    for r in state.get("history", []):
        history.append(RoundResult(
            round_number=r["round"], player_action=r["opponent_action"],
            opponent_action=r["player_action"],
            player_payoff=r.get("o_pay", float()), opponent_payoff=r.get("p_pay", float())))
    opp_actions = info.get("opponent_actions")
    actions = list(opp_actions) if opp_actions else info["actions"]
    obs = GameObservation(
        game_name=info.get("key", state["game"]),
        game_description=info.get("description", ""),
        available_actions=actions, current_round=state["round"],
        total_rounds=state["max_rounds"], history=history,
        player_score=state["o_score"], opponent_score=state["p_score"],
        opponent_strategy="human")
    prompt = PromptBuilder.build(obs)
    try:
        if provider == "Anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model, max_tokens=_TEN + _TEN, system=_SYS_PROMPT,
                messages=[{"role": "user", "content": prompt}])
            raw = resp.content[_ZERO].text
        elif provider == "OpenAI":
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model, max_tokens=_TEN + _TEN,
                messages=[{"role": "system", "content": _SYS_PROMPT},
                          {"role": "user", "content": prompt}])
            raw = resp.choices[_ZERO].message.content
        else:
            return _rand.choice(info["actions"]), f"Unknown provider: {provider}"
    except Exception as exc:
        return _rand.choice(info["actions"]), f"API error: {exc}"
    act_list = list(opp_actions) if opp_actions else info["actions"]
    return parse_action(raw, act_list), raw.strip()


def _finish_round(state, info, opp, p_pay, o_pay, action_str, raw=None):
    state["round"] += _ONE
    state["p_score"] += p_pay
    state["o_score"] += o_pay
    state["history"].append({"round": state["round"], "player_action": action_str,
                             "opponent_action": opp, "p_pay": p_pay, "o_pay": o_pay})
    if raw is not None:
        state.setdefault("llm_log", []).append({"round": state["round"], "raw": raw})
    if state["round"] >= state["max_rounds"]:
        state["done"] = True
    acts = info["actions"]
    return (state, _render(state), info["description"],
            gr.update(choices=acts, value=acts[_ZERO]))


def play_round(action_str, state, provider=None, model=None, api_key=None):
    if state is None or state["done"]:
        return state, "Reset the game to play again.", gr.update(), gr.update()
    info = _get_game_info(state["game"], state.get("variants"))
    np = state.get("num_players", _TWO)
    is_llm = state.get("strategy") == _LLM_OPPONENT_LABEL
    if np > _TWO and _HAS_NPLAYER_ENV:
        nenv = state.get("nplayer_env")
        if nenv is None:
            return state, "Error: N-player env not initialized.", gr.update(), gr.update()
        obs = nenv.step(NPlayerAction(action=action_str))
        state["round"] += _ONE
        state["scores"] = list(obs.scores)
        state["history"].append({"round": state["round"],
                                 "actions": list(obs.last_round.actions),
                                 "payoffs": list(obs.last_round.payoffs)})
        if obs.done:
            state["done"] = True
        acts = info["actions"]
        return (state, _render(state), info["description"],
                gr.update(choices=acts, value=acts[_ZERO]))
    if is_llm:
        resolved_key = _resolve_api_key(provider, api_key)
        if not resolved_key:
            return state, "No OAuth token available and no API key provided.", gr.update(), gr.update()
        opp, raw = _llm_choose_action(state, info, provider, model, resolved_key)
        p_pay, o_pay = info["payoff_fn"](action_str, opp)
        return _finish_round(state, info, opp, p_pay, o_pay, action_str, raw)
    opp_actions = info.get("opponent_actions")
    opp_act_list = list(opp_actions) if opp_actions else info["actions"]
    strat = STRATEGIES_2P[state["strategy"]]
    if _HAS_FULL_STRATEGIES:
        opp = strat.choose_action(info.get("game_type", "matrix"), opp_act_list, state["history"])
    else:
        opp = strat(opp_act_list, state["history"])
    p_pay, o_pay = info["payoff_fn"](action_str, opp)
    return _finish_round(state, info, opp, p_pay, o_pay, action_str)


def reset_game(gname, sname, variants=None, max_rounds=None):
    vlist = list(variants or [])
    info = _get_game_info(gname, vlist)
    np = info.get("num_players", _TWO)
    st = _blank(gname, sname, vlist, max_rounds)
    if np > _TWO and _HAS_NPLAYER_ENV:
        nenv = NPlayerEnvironment()
        nenv.reset(_GAME_INFO.get(gname, {}).get("key", ""),
                   opponent_strategies=[sname] * (np - _ONE))
        st["nplayer_env"] = nenv
    acts = info["actions"]
    return (st, _render(st), info["description"], gr.update(choices=acts, value=acts[_ZERO]))


def on_game_change(gname, sname, variants=None):
    return reset_game(gname, sname, variants)


def on_category_change(tag, mp_filter):
    names = _filter_game_names(tag)
    names = _filter_by_mp(mp_filter, names)
    if not names:
        names = sorted(_GAME_INFO.keys())
    return gr.update(choices=names, value=names[_ZERO])


def on_mp_filter_change(mp_filter, tag):
    return on_category_change(tag, mp_filter)


def on_game_select(gname):
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    if np > _TWO and _HAS_NPLAYER_ENV:
        strat_names = _NPLAYER_STRAT_NAMES
    else:
        strat_names = _strategies_for_game(gname) + [_LLM_OPPONENT_LABEL]
    label = f"Players: {np}" if np > _TWO else "Two-Player"
    return gr.update(choices=strat_names, value=strat_names[_ZERO]), gr.update(value=label)


def on_game_select_variant(gname):
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    if np > _TWO or not _HAS_VARIANTS:
        return gr.update(choices=[], value=[])
    available = [v for v in _HUMAN_VARIANTS if v not in _2P_ONLY_VARIANTS or np <= _TWO]
    return gr.update(choices=available, value=[])


def on_strategy_change(sname):
    is_llm = sname == _LLM_OPPONENT_LABEL
    show_key = is_llm and not _HAS_OAUTH
    return gr.update(visible=is_llm), gr.update(visible=show_key)


def on_provider_change(provider):
    models = _LLM_MODELS.get(provider, [])
    return gr.update(choices=models, value=models[_ZERO] if models else "")


def _build_reference_md():
    if not _HAS_REGISTRY:
        return "# Game Theory Reference\n\nFull registry not available."
    sections = []
    for dim_name, tags in sorted(_CATEGORY_DIMS.items()):
        sec = [f"## {dim_name.replace('_', ' ').title()}"]
        for tag in tags:
            names = sorted(_KEY_TO_NAME[k] for k in get_games_by_tag(tag) if k in _KEY_TO_NAME)
            if names:
                sec.append(f"**{tag}** ({len(names)}): {', '.join(names)}")
        sections.append("\n\n".join(sec))
    np_games = [(gn, gi) for gn, gi in _GAME_INFO.items() if gi.get("num_players", _TWO) > _TWO]
    if np_games:
        np_lines = ["## Multiplayer Games", "| Game | Players | Actions | Rounds |",
                     "|------|---------|---------|--------|"]
        for gn, gi in sorted(np_games):
            acts = gi["actions"]
            act_str = ", ".join(acts[:_FOUR]) + (f" ... ({len(acts)} total)" if len(acts) > _FOUR else "")
            np_lines.append(f"| {gn} | {gi['num_players']} | {act_str} | {gi['default_rounds']} |")
        sections.append("\n".join(np_lines))
    if _HUMAN_VARIANTS:
        sections.append("## Composable Variants\n" + "\n".join(f"- **{v}**" for v in _HUMAN_VARIANTS))
    slines = ["## Opponent Strategies",
              f"**Generic** ({len(_GENERIC_STRATEGIES)}): {', '.join(_GENERIC_STRATEGIES)}"]
    for gt, strats in sorted(_GAME_TYPE_STRATEGIES.items()):
        slines.append(f"**{gt}**: {', '.join(strats)}")
    if _HAS_NPLAYER_ENV:
        slines.append(f"**N-player**: {', '.join(_NPLAYER_STRAT_NAMES)}")
    slines.append(f"\n**LLM Opponents**: Select '{_LLM_OPPONENT_LABEL}' as strategy, "
                  "provide your Anthropic or OpenAI API key, and play against Claude or GPT.")
    sections.append("\n\n".join(slines))
    total, np_count = len(_GAME_INFO), len(np_games)
    return (f"# Game Theory Reference\n\n**{total} games** ({total - np_count} two-player, "
            f"{np_count} multiplayer)\n\n" + "\n\n---\n\n".join(sections))
