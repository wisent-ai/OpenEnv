"""Tournament-runner callback for the gradio Tournament Results tab.

Runs a chosen agent strategy against every other registered 2-player
strategy across the user-selected games and reports the agent's mean
per-round self-payoff plus cooperation rate as a descriptive secondary
column.

Avoids ``bench.evaluation.tournament.TournamentRunner`` because that pulls
in ``env.environment`` (and therefore the upstream ``openenv-core``
package); we simulate rounds directly via ``GameConfig.payoff_fn`` and
the ``STRATEGIES_2P`` registry, matching the pattern already used by
``callbacks.play_round``.
"""

from __future__ import annotations

from registry import (
    _ZERO, _ONE, _TWO,
    DEFAULT_NUM_ROUNDS,
    _HAS_FULL_STRATEGIES, STRATEGIES_2P, _GAME_INFO,
    _LLM_MODELS, _SYS_PROMPT, get_env_api_key, ANTHROPIC_OAUTH_BETA_HEADER,
    PromptBuilder, parse_action, GameObservation, RoundResult,
)


def _model_provider(model_name):
    """Return the OAuth provider name a model belongs to, or empty string."""
    for prov, models in _LLM_MODELS.items():
        if model_name in models:
            return prov
    return ""


def _llm_token(model_name):
    """Call an LLM via OAuth, return the raw text. Mirrors llm_arena._call_llm."""
    return None  # placeholder; the live caller is _llm_call below.


def _llm_call(model_name: str, prompt: str) -> str:
    provider = _model_provider(model_name)
    token = get_env_api_key(provider)
    if not token:
        raise RuntimeError(f"OAuth token unavailable for {provider}")
    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(
            api_key=None, auth_token=token,
            default_headers={"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER},
        )
        resp = client.messages.create(
            model=model_name, max_tokens=20, system=_SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[_ZERO].text
    if provider == "OpenAI":
        import openai
        client = openai.OpenAI(api_key=token)
        resp = client.chat.completions.create(
            model=model_name, max_tokens=20,
            messages=[{"role": "system", "content": _SYS_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[_ZERO].message.content
    raise RuntimeError(f"Unknown provider for model {model_name}")


def _build_obs_for_side(info, my_history, opp_history, rnd, my_score, opp_score, total):
    """Construct a GameObservation for one side of an LLM-vs-LLM match."""
    history = []
    for ph, oh in zip(my_history, opp_history):
        history.append(RoundResult(
            round_number=ph["round"],
            player_action=ph["action"], opponent_action=oh["action"],
            player_payoff=ph["payoff"], opponent_payoff=oh["payoff"]))
    return GameObservation(
        game_name=info.get("key", ""),
        game_description=info.get("description", ""),
        available_actions=info["actions"],
        current_round=rnd, total_rounds=total, history=history,
        player_score=my_score, opponent_score=opp_score,
        opponent_strategy="llm")


def _simulate_episode_llm_self(info, model_name, num_rounds):
    """LLM-vs-itself episode. Both sides run the same model; per-side history
    is flipped so each side sees its own moves as 'player_action'.
    Returns (p_score, action_counts, cooperative_count)."""
    p_hist, o_hist = [], []
    p_score = o_score = 0.0
    counts: dict = {}
    coop = _ZERO
    coop_set = {"cooperate", "stag", "dove"}
    actions = info["actions"]
    for rnum in range(num_rounds):
        rnd = rnum + _ONE
        prompt_p = PromptBuilder.build(_build_obs_for_side(
            info, p_hist, o_hist, rnd, p_score, o_score, num_rounds))
        prompt_o = PromptBuilder.build(_build_obs_for_side(
            info, o_hist, p_hist, rnd, o_score, p_score, num_rounds))
        a = parse_action(_llm_call(model_name, prompt_p), actions)
        b = parse_action(_llm_call(model_name, prompt_o), actions)
        p_pay, o_pay = info["payoff_fn"](a, b)
        p_score += p_pay
        o_score += o_pay
        counts[a] = counts.get(a, _ZERO) + _ONE
        if a in coop_set:
            coop += _ONE
        p_hist.append({"round": rnd, "action": a, "payoff": p_pay})
        o_hist.append({"round": rnd, "action": b, "payoff": o_pay})
    return p_score, counts, coop


def _flip_history(history):
    """Swap player_action / opponent_action in each entry, for the opponent's view."""
    return [
        {"player_action": h["opponent_action"], "opponent_action": h["player_action"]}
        for h in history
    ]


def _simulate_episode(info, agent_strategy, opponent_strategy, num_rounds):
    """Play *num_rounds* of *info* with both sides using STRATEGIES_2P entries.

    Returns ``(player_score, opponent_score, action_counts, total_actions,
    cooperation_count)``.
    """
    a_strat = STRATEGIES_2P[agent_strategy]
    o_strat = STRATEGIES_2P[opponent_strategy]
    opp_acts_field = info.get("opponent_actions")
    p_acts = info["actions"]
    o_acts = list(opp_acts_field) if opp_acts_field else p_acts
    history = []
    p_total = float()
    o_total = float()
    counts = {}
    coop = _ZERO
    coop_set = {"cooperate", "stag", "dove"}
    gtype = info.get("game_type", "matrix")
    for rnum in range(num_rounds):
        if _HAS_FULL_STRATEGIES:
            a = a_strat.choose_action(gtype, p_acts, history)
            o = o_strat.choose_action(gtype, o_acts, _flip_history(history))
        else:
            a = a_strat(p_acts, history)
            o = o_strat(o_acts, _flip_history(history))
        p_pay, o_pay = info["payoff_fn"](a, o)
        p_total += p_pay
        o_total += o_pay
        counts[a] = counts.get(a, _ZERO) + _ONE
        if a in coop_set:
            coop += _ONE
        history.append({
            "round": rnum + _ONE, "player_action": a, "opponent_action": o,
            "p_pay": p_pay, "o_pay": o_pay,
        })
    return p_total, o_total, counts, num_rounds, coop


def run_metrics_tournament(
    agent_strategy, episodes_per_pair, selected_games,
    mode="hardcoded", model="",
):
    """Run a tournament in scripted-strategy mode or LLM-self-play mode.

    mode="hardcoded": *agent_strategy* plays against every other base 2P
    strategy on *selected_games*; columns are mean self-payoff and
    cooperation rate.

    mode="llm_self": both sides of every match are *model* (an OAuth-
    backed LLM listed in registry._LLM_MODELS). Same columns. Only 2P
    games are scored; N-player games skipped because the gradio harness
    builds GameObservation, not NPlayerObservation.
    """
    if not selected_games:
        return "_Select at least one game._"
    if mode == "llm_self":
        if not model:
            return "_Pick an LLM model for self-play mode._"
        return _run_llm_self_table(model, episodes_per_pair, selected_games)
    if not _HAS_FULL_STRATEGIES:
        return "_STRATEGIES_2P registry not available; cannot run tournament._"
    n_eps = max(_ONE, int(episodes_per_pair))
    out = [
        f"# Tournament Results — agent = `{agent_strategy}`",
        f"*{n_eps} episode(s) per opponent. Headline: agent's mean per-round "
        "self-payoff (the quantity the training reward optimises).*",
        "",
        "| Game | Mean self-payoff | Cooperation rate |",
        "|------|------------------|------------------|",
    ]
    opponents = [s for s in STRATEGIES_2P if s != agent_strategy]
    for gname in selected_games:
        info = _GAME_INFO.get(gname)
        if not info or info.get("num_players", _TWO) > _TWO:
            continue
        rounds = info.get("default_rounds", DEFAULT_NUM_ROUNDS)
        agg_p_score = float()
        agg_rounds = _ZERO
        agg_coop = _ZERO
        for opp in opponents:
            for _ep in range(n_eps):
                ps, _os, _c, t, k = _simulate_episode(
                    info, agent_strategy, opp, rounds,
                )
                agg_p_score += ps
                agg_rounds += t
                agg_coop += k
        if agg_rounds == _ZERO:
            continue
        mean_pay = agg_p_score / agg_rounds
        coop_rate = agg_coop / agg_rounds
        out.append(f"| {gname} | {mean_pay:.3f} | {coop_rate:.3f} |")
    return "\n".join(out)


def _run_llm_self_table(model, episodes_per_pair, selected_games):
    n_eps = max(_ONE, int(episodes_per_pair))
    out = [
        f"# Tournament Results — agent = `{model}` (self-play)",
        f"*{n_eps} episode(s) per game. Both sides are the same model.*",
        "",
        "| Game | Mean self-payoff | Cooperation rate |",
        "|------|------------------|------------------|",
    ]
    for gname in selected_games:
        info = _GAME_INFO.get(gname)
        if not info or info.get("num_players", _TWO) > _TWO:
            continue
        rounds = info.get("default_rounds", DEFAULT_NUM_ROUNDS)
        agg_p_score = 0.0
        agg_rounds = _ZERO
        agg_coop = _ZERO
        try:
            for _ep in range(n_eps):
                ps, _c, k = _simulate_episode_llm_self(info, model, rounds)
                agg_p_score += ps
                agg_rounds += rounds
                agg_coop += k
        except RuntimeError as exc:
            out.append(f"| {gname} | _error: {exc}_ | — |")
            continue
        if agg_rounds == _ZERO:
            continue
        mean_pay = agg_p_score / agg_rounds
        coop_rate = agg_coop / agg_rounds
        out.append(f"| {gname} | {mean_pay:.3f} | {coop_rate:.3f} |")
    return "\n".join(out)
