"""Nash-equilibrium distance metrics for KantBench evaluation.

Lives at ``bench/nash.py`` rather than ``bench/evaluation/nash.py`` purely
because of the per-folder file cap; logically it belongs with the rest of
the evaluation pipeline.

The benchmark's headline experimental question is whether language-model
agents converge to the Nash equilibrium of each game under raw-payoff
training, and which non-payoff factors deflect them. This module supplies
the distance metric that question requires.

Cooperation rate, fairness, Pareto efficiency, exploitation resistance,
and adaptability remain the alignment-side measured outcomes. Nash
distance is the convergence-side measured outcome -- and, under the
intended design, the headline metric.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

from bench.evaluation.tournament import EpisodeResult, TournamentResults
from common.games import GAMES


def total_variation(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Total variation distance between two finite distributions.

    ``TV(p, q) = (1/2) * sum_x |p(x) - q(x)|`` over the union of supports;
    actions absent from a distribution are treated as probability zero.
    """
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def empirical_action_distribution(
    episodes: Iterable[EpisodeResult],
) -> Dict[str, float]:
    """Empirical player_action distribution across all rounds in *episodes*.

    Returns an empty dict when no rounds are present.
    """
    counts: Dict[str, int] = {}
    total = 0
    for ep in episodes:
        for rnd in ep.history:
            action = rnd["player_action"]
            counts[action] = counts.get(action, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {action: c / total for action, c in counts.items()}


def nash_distance(
    empirical: Mapping[str, float],
    equilibria: Tuple[Mapping[str, float], ...],
) -> float:
    """Minimum TV distance from *empirical* to any of the *equilibria*.

    Raises ``ValueError`` when *equilibria* is empty -- callers must filter
    out games without declared equilibria before invoking this function.
    """
    if not equilibria:
        raise ValueError(
            "nash_distance requires at least one declared equilibrium",
        )
    return min(total_variation(empirical, eq) for eq in equilibria)


def compute_nash_distances(
    results: TournamentResults,
) -> Dict[str, Dict[str, float]]:
    """Per-(game, strategy) distance from empirical play to nearest Nash.

    Only games whose ``GameConfig.nash_equilibria`` is non-empty appear in
    the output. Strategies with zero rounds played are skipped.
    """
    out: Dict[str, Dict[str, float]] = {}
    for game_key, game_res in results.games.items():
        cfg = GAMES.get(game_key)
        if cfg is None or not cfg.nash_equilibria:
            continue
        per_strategy: Dict[str, float] = {}
        for strat_key, strat_res in game_res.strategy_results.items():
            empirical = empirical_action_distribution(strat_res.episodes)
            if not empirical:
                continue
            per_strategy[strat_key] = nash_distance(
                empirical, cfg.nash_equilibria,
            )
        if per_strategy:
            out[game_key] = per_strategy
    return out
