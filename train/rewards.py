"""Reward functions for the training pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from constant_definitions.game_constants import (
    EVAL_HALF,
    EVAL_ONE,
    EVAL_ONE_FLOAT,
    EVAL_TWO,
    EVAL_ZERO,
    EVAL_ZERO_FLOAT,
)
from constant_definitions.train.grpo_constants import (
    GRPO_SHAPING_ALPHA_DENOMINATOR,
    GRPO_SHAPING_ALPHA_NUMERATOR,
)

_FIVE = EVAL_TWO + EVAL_TWO + EVAL_ONE

# Default weight per sub-metric (equal weighting across five metrics).
_DEFAULT_WEIGHT_NUMERATOR = EVAL_ONE
_DEFAULT_WEIGHT_DENOMINATOR = _FIVE


def _default_weights() -> Dict[str, float]:
    """Return default equal weights for the five reward components."""
    w = _DEFAULT_WEIGHT_NUMERATOR / _DEFAULT_WEIGHT_DENOMINATOR
    return {
        "cooperation_rate": w,
        "pareto_efficiency": w,
        "fairness_index": w,
        "exploitation_resistance": w,
        "adaptability": w,
    }


# ---------------------------------------------------------------------------
# Per-episode reward
# ---------------------------------------------------------------------------


def episode_reward(
    player_score: float,
    opponent_score: float,
    cooperation_rate: float,
    total_rounds: int,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a scalar reward for a single episode.

    Uses per-episode metrics that can be computed without cross-strategy data:
    cooperation_rate, pareto_efficiency proxy, and fairness_index.

    Exploitation_resistance and adaptability default to neutral since they
    require cross-strategy comparison (see ``batch_reward``).
    """
    w = weights if weights is not None else _default_weights()

    # Cooperation rate: direct
    coop = cooperation_rate

    # Pareto efficiency proxy: normalised joint score
    joint = player_score + opponent_score
    if total_rounds > EVAL_ZERO:
        pareto_proxy = joint / total_rounds
        # Clamp to [zero, one]
        pareto_proxy = max(EVAL_ZERO_FLOAT, min(EVAL_ONE_FLOAT, pareto_proxy))
    else:
        pareto_proxy = EVAL_ZERO_FLOAT

    # Fairness: EVAL_ONE_FLOAT - |p - o| / (|p| + |o|)
    denom = abs(player_score) + abs(opponent_score)
    if denom > EVAL_ZERO_FLOAT:
        fairness = EVAL_ONE_FLOAT - abs(player_score - opponent_score) / denom
    else:
        fairness = EVAL_ONE_FLOAT

    # Cross-strategy metrics default to neutral midpoint
    exploit_resist = EVAL_HALF
    adapt = EVAL_HALF

    reward = (
        w["cooperation_rate"] * coop
        + w["pareto_efficiency"] * pareto_proxy
        + w["fairness_index"] * fairness
        + w["exploitation_resistance"] * exploit_resist
        + w["adaptability"] * adapt
    )
    return reward


# ---------------------------------------------------------------------------
# Batch reward (cross-strategy)
# ---------------------------------------------------------------------------


def batch_reward(
    episode_results: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute cross-strategy reward metrics over a batch of episodes.

    Parameters
    ----------
    episode_results : list of dict
        Each dict must have keys: ``game``, ``strategy``,
        ``player_score``, ``opponent_score``, ``cooperation_rate``.

    Returns
    -------
    dict
        Mapping of metric name to value for exploitation_resistance
        and adaptability computed across strategies for each game.
    """
    w = weights if weights is not None else _default_weights()

    # Group by game
    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episode_results:
        game = ep["game"]
        if game not in by_game:
            by_game[game] = []
        by_game[game].append(ep)

    exploit_scores: List[float] = []
    adapt_scores: List[float] = []

    for _game, episodes in by_game.items():
        # Group by strategy within game
        by_strat: Dict[str, List[Dict[str, Any]]] = {}
        for ep in episodes:
            strat = ep["strategy"]
            if strat not in by_strat:
                by_strat[strat] = []
            by_strat[strat].append(ep)

        if len(by_strat) <= EVAL_ONE:
            continue

        # Exploitation resistance: performance against always_defect
        # relative to best/worst across strategies
        strat_scores = {
            s: sum(e["player_score"] for e in eps)
            for s, eps in by_strat.items()
        }
        best = max(strat_scores.values())
        worst = min(strat_scores.values())
        spread = best - worst
        if "always_defect" in strat_scores and spread > EVAL_ZERO_FLOAT:
            ad_score = strat_scores["always_defect"]
            exploit_scores.append((ad_score - worst) / spread)

        # Adaptability: variance of cooperation rates across strategies
        coop_rates = []
        for eps in by_strat.values():
            rate_sum = sum(e["cooperation_rate"] for e in eps)
            coop_rates.append(rate_sum / len(eps))

        if len(coop_rates) > EVAL_ONE:
            mean_coop = sum(coop_rates) / len(coop_rates)
            var = sum(
                (r - mean_coop) ** EVAL_TWO for r in coop_rates
            ) / len(coop_rates)
            capped = min(var, EVAL_HALF)
            adapt_scores.append(capped / EVAL_HALF)

    exploit_val = (
        sum(exploit_scores) / len(exploit_scores)
        if exploit_scores else EVAL_HALF
    )
    adapt_val = (
        sum(adapt_scores) / len(adapt_scores)
        if adapt_scores else EVAL_ZERO_FLOAT
    )

    return {
        "exploitation_resistance": exploit_val,
        "adaptability": adapt_val,
    }


# ---------------------------------------------------------------------------
# Per-step shaping
# ---------------------------------------------------------------------------


def per_step_shaping(
    player_payoff: float,
    opponent_payoff: float,
    payoff_min: float,
    payoff_max: float,
) -> float:
    """Optional per-step reward shaping based on immediate payoffs.

    Returns a small bonus proportional to normalised joint payoff,
    scaled by the shaping coefficient alpha.
    """
    alpha = GRPO_SHAPING_ALPHA_NUMERATOR / GRPO_SHAPING_ALPHA_DENOMINATOR
    payoff_range = payoff_max - payoff_min
    if payoff_range <= EVAL_ZERO_FLOAT:
        return EVAL_ZERO_FLOAT
    joint = player_payoff + opponent_payoff
    normalised = (joint - payoff_min * EVAL_TWO) / (payoff_range * EVAL_TWO)
    return alpha * normalised
