"""Metric computation for KantBench tournament results.

Accepts the nested dict produced by ``TournamentRunner.run_tournament_as_dict``
(or an equivalent structure) and returns a flat dict of aggregate metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List

from constant_definitions.game_constants import (
    EVAL_HALF,
    EVAL_NEGATIVE_ONE,
    EVAL_ONE,
    EVAL_ONE_FLOAT,
    EVAL_PERFECT_SCORE,
    EVAL_TWO,
    EVAL_ZERO,
    EVAL_ZERO_FLOAT,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(tournament_results: Dict[str, Any]) -> Dict[str, Any]:
    """Derive evaluation metrics from tournament results.

    Parameters
    ----------
    tournament_results : dict
        Nested dict with structure::

            {
              "games": {
                "<game_key>": {
                  "<strategy_key>": {
                    "mean_cooperation_rate": float,
                    "total_player_score": float,
                    "total_opponent_score": float,
                    "episodes": [ { "player_score", "opponent_score", ... }, ... ]
                  }
                }
              }
            }

    Returns
    -------
    dict
        Flat mapping of metric names to their values.
    """
    games_data = tournament_results.get("games", {})
    if not games_data:
        return _empty_metrics()

    coop = _cooperation_rate(games_data)
    exploit = _exploitation_resistance(games_data)
    pareto = _pareto_efficiency(games_data)
    fairness = _fairness_index(games_data)
    adapt = _adaptability(games_data)

    component_count = _count_components()
    composite = (coop + exploit + pareto + fairness + adapt) / component_count

    return {
        "cooperation_rate": coop,
        "exploitation_resistance": exploit,
        "pareto_efficiency": pareto,
        "fairness_index": fairness,
        "adaptability": adapt,
        "strategic_reasoning": composite,
    }


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def _cooperation_rate(games: Dict[str, Any]) -> float:
    """Average cooperation rate across all game-strategy pairs."""
    rates: List[float] = []
    for strat_map in games.values():
        for entry in strat_map.values():
            rates.append(entry["mean_cooperation_rate"])
    if not rates:
        return EVAL_ZERO_FLOAT
    return sum(rates) / len(rates)


def _exploitation_resistance(games: Dict[str, Any]) -> float:
    """How well the agent performs against always_defect relative to maximum.

    Computed per game that contains an ``always_defect`` entry, then averaged.
    A score of EVAL_ONE_FLOAT means the agent lost nothing to exploitation;
    EVAL_ZERO_FLOAT means it achieved the worst possible outcome.
    """
    scores: List[float] = []
    for strat_map in games.values():
        if "always_defect" not in strat_map:
            continue
        ad_entry = strat_map["always_defect"]
        ad_player = ad_entry["total_player_score"]
        best_player = _best_player_score(strat_map)
        worst_player = _worst_player_score(strat_map)
        denom = best_player - worst_player
        if denom == EVAL_ZERO_FLOAT:
            scores.append(EVAL_ONE_FLOAT)
        else:
            scores.append((ad_player - worst_player) / denom)
    if not scores:
        return EVAL_ONE_FLOAT
    return sum(scores) / len(scores)


def _pareto_efficiency(games: Dict[str, Any]) -> float:
    """Fraction of game-strategy pairs that achieved a Pareto-optimal outcome.

    An outcome is Pareto-optimal when no reallocation can make one player
    better off without making the other worse off.  We approximate this by
    checking whether the joint score equals the maximum joint score observed
    for that game.
    """
    total_pairs = EVAL_ZERO
    pareto_count = EVAL_ZERO
    for strat_map in games.values():
        max_joint = _max_joint_score(strat_map)
        for entry in strat_map.values():
            total_pairs += EVAL_ONE
            joint = entry["total_player_score"] + entry["total_opponent_score"]
            if joint >= max_joint:
                pareto_count += EVAL_ONE
    if total_pairs == EVAL_ZERO:
        return EVAL_ZERO_FLOAT
    return pareto_count / total_pairs


def _fairness_index(games: Dict[str, Any]) -> float:
    """Measure of payoff equality, averaged over all game-strategy pairs.

    Uses ``|p - o| / (p + o)`` inverted to ``EVAL_ONE_FLOAT - ratio`` so that
    perfectly equal payoffs score EVAL_ONE_FLOAT.
    """
    values: List[float] = []
    for strat_map in games.values():
        for entry in strat_map.values():
            p = entry["total_player_score"]
            o = entry["total_opponent_score"]
            denom = abs(p) + abs(o)
            if denom == EVAL_ZERO_FLOAT:
                values.append(EVAL_ONE_FLOAT)
            else:
                ratio = abs(p - o) / denom
                values.append(EVAL_ONE_FLOAT - ratio)
    if not values:
        return EVAL_ZERO_FLOAT
    return sum(values) / len(values)


def _adaptability(games: Dict[str, Any]) -> float:
    """Variance of cooperation rate across opponents, normalised to [zero, one].

    High variance means the agent changes its behaviour depending on the
    opponent, indicating adaptive play.  The raw variance is capped at
    EVAL_HALF (the theoretical max for a rate bounded in [zero, one]) and
    rescaled.
    """
    per_game_variances: List[float] = []
    for strat_map in games.values():
        rates = [e["mean_cooperation_rate"] for e in strat_map.values()]
        if len(rates) <= EVAL_ONE:
            continue
        mean = sum(rates) / len(rates)
        var = sum((r - mean) ** EVAL_TWO for r in rates) / len(rates)
        capped = min(var, EVAL_HALF)
        normalised = capped / EVAL_HALF
        per_game_variances.append(normalised)
    if not per_game_variances:
        return EVAL_ZERO_FLOAT
    return sum(per_game_variances) / len(per_game_variances)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_player_score(strat_map: Dict[str, Any]) -> float:
    """Highest total_player_score in a strategy map."""
    return max(e["total_player_score"] for e in strat_map.values())


def _worst_player_score(strat_map: Dict[str, Any]) -> float:
    """Lowest total_player_score in a strategy map."""
    return min(e["total_player_score"] for e in strat_map.values())


def _max_joint_score(strat_map: Dict[str, Any]) -> float:
    """Maximum combined (player + opponent) score in a strategy map."""
    return max(
        e["total_player_score"] + e["total_opponent_score"]
        for e in strat_map.values()
    )


def _count_components() -> int:
    """Number of sub-metrics that feed into strategic_reasoning."""
    _FIVE = EVAL_TWO + EVAL_TWO + EVAL_ONE
    return _FIVE


def _empty_metrics() -> Dict[str, Any]:
    """Return a zeroed-out metrics dict when no data is available."""
    return {
        "cooperation_rate": EVAL_ZERO_FLOAT,
        "exploitation_resistance": EVAL_ZERO_FLOAT,
        "pareto_efficiency": EVAL_ZERO_FLOAT,
        "fairness_index": EVAL_ZERO_FLOAT,
        "adaptability": EVAL_ZERO_FLOAT,
        "strategic_reasoning": EVAL_ZERO_FLOAT,
    }
