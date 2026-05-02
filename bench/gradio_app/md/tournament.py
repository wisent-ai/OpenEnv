"""Tournament-runner callback for the gradio Tournament Results tab.

Runs a chosen agent strategy against every other registered 2-player
strategy across the user-selected games, computes per-game Nash distance
to the nearest declared equilibrium, mean self-payoff, and cooperation
rate, then renders the result as a markdown table.

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
)


def _tv(p, q):
    """Total variation distance over the union of supports of *p* and *q*."""
    keys = set(p) | set(q)
    if not keys:
        return float()
    return float(0.5) * sum(abs(p.get(k, float()) - q.get(k, float())) for k in keys)


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


def run_metrics_tournament(agent_strategy, episodes_per_pair, selected_games):
    """Run *agent_strategy* against every base 2P strategy on *selected_games*.

    Returns a markdown report whose headline is per-game Nash distance for
    games with declared equilibria, followed by the alignment-side metrics.
    """
    if not selected_games:
        return "_Select at least one game._"
    if not _HAS_FULL_STRATEGIES:
        return "_STRATEGIES_2P registry not available; cannot run tournament._"
    n_eps = max(_ONE, int(episodes_per_pair))
    out = [
        f"# Tournament Results — agent = `{agent_strategy}`",
        f"*{n_eps} episode(s) per opponent. Headline: Nash distance to nearest "
        "declared equilibrium.*",
        "",
        "| Game | Nash distance | Mean self-payoff | Cooperation rate |",
        "|------|---------------|------------------|------------------|",
    ]
    opponents = [s for s in STRATEGIES_2P if s != agent_strategy]
    for gname in selected_games:
        info = _GAME_INFO.get(gname)
        if not info or info.get("num_players", _TWO) > _TWO:
            continue
        rounds = info.get("default_rounds", DEFAULT_NUM_ROUNDS)
        agg_counts = {}
        agg_total = _ZERO
        agg_p_score = float()
        agg_rounds = _ZERO
        agg_coop = _ZERO
        for opp in opponents:
            for _ep in range(n_eps):
                ps, _os, c, t, k = _simulate_episode(info, agent_strategy, opp, rounds)
                agg_p_score += ps
                agg_rounds += t
                agg_coop += k
                for a, n in c.items():
                    agg_counts[a] = agg_counts.get(a, _ZERO) + n
                    agg_total += n
        if agg_total == _ZERO:
            continue
        empirical = {a: n / agg_total for a, n in agg_counts.items()}
        eqs = info.get("nash_equilibria", ()) or ()
        nash_str = (
            f"{min(_tv(empirical, eq) for eq in eqs):.3f}" if eqs else "—"
        )
        mean_pay = agg_p_score / agg_rounds if agg_rounds else float()
        coop_rate = agg_coop / agg_rounds if agg_rounds else float()
        out.append(f"| {gname} | {nash_str} | {mean_pay:.3f} | {coop_rate:.3f} |")
    return "\n".join(out)
