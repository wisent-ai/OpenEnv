"""Reward functions for the training pipeline.

The training reward is the raw self-payoff received from the game environment.
Nothing else: no cooperation bonus, no fairness penalty, no joint-payoff term,
no exploitation indicator.

This is a deliberate methodological choice. The benchmark's research question
is whether language-model agents converge to the Nash equilibrium of each
game, and which non-payoff factors (game framing, opponent identity,
communication channels, reputation visibility, Kantian-style system prompts)
systematically deflect them from it. If the training reward bakes in
cooperation or fairness, any "cooperative" outcome at evaluation time is
just the shaping signal being read back out -- the experiment proves nothing.

Cooperation rate, Pareto efficiency, fairness index, exploitation resistance,
and adaptability are still computed -- but as MEASURED OUTCOMES in
``bench/evaluation/metrics.py``, never as training signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from constant_definitions.game_constants import EVAL_ZERO, EVAL_ZERO_FLOAT
from constant_definitions.train.grpo_constants import (
    GRPO_SHAPING_ALPHA_DENOMINATOR,
    GRPO_SHAPING_ALPHA_NUMERATOR,
)


def expected_self_payoff_uniform_opponent(
    player_action: str,
    game_config: Any,
) -> float:
    """Mean self-payoff for *player_action* against a uniform-random opponent.

    Used as the GRPO per-completion reward: each completion is a single
    round's action, and the only opponent-agnostic, payoff-only signal we
    can attach to it is the expected self-payoff under maximum-entropy
    opponent play. Reads ``game_config.payoff_fn`` and ``game_config.actions``
    (using ``opponent_actions`` when present, otherwise the symmetric
    action set) and averages ``payoff_fn(player_action, opp)[0]`` over the
    opponent action support.
    """
    opponent_support = (
        list(game_config.opponent_actions)
        if game_config.opponent_actions
        else list(game_config.actions)
    )
    if not opponent_support:
        return EVAL_ZERO_FLOAT
    total = EVAL_ZERO_FLOAT
    for opp_action in opponent_support:
        try:
            p_pay, _ = game_config.payoff_fn(player_action, opp_action)
        except Exception:
            return EVAL_ZERO_FLOAT
        total += float(p_pay)
    return total / len(opponent_support)


# ---------------------------------------------------------------------------
# Per-episode reward
# ---------------------------------------------------------------------------


def episode_reward(
    player_score: float,
    opponent_score: float,  # noqa: ARG001 -- ignored, kept for caller compatibility
    cooperation_rate: float,  # noqa: ARG001 -- ignored, kept for caller compatibility
    total_rounds: int,
    weights: Optional[Dict[str, float]] = None,  # noqa: ARG001 -- deprecated
) -> float:
    """Mean per-round self-payoff for the episode.

    Returns ``player_score / total_rounds``. The opponent's score, the
    agent's cooperation rate, and any weight overrides are deliberately
    ignored: the training signal must be the agent's own game payoff and
    nothing else, so that deviation from Nash at evaluation time is
    attributable to non-payoff factors rather than to the reward function.
    """
    if total_rounds <= EVAL_ZERO:
        return EVAL_ZERO_FLOAT
    return player_score / total_rounds


# ---------------------------------------------------------------------------
# Batch reward (cross-strategy, evaluation-only)
# ---------------------------------------------------------------------------


def batch_reward(
    episode_results: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,  # noqa: ARG001 -- deprecated
) -> Dict[str, float]:
    """Mean self-payoff per game, computed over a batch of episodes.

    This function exists for downstream tooling that previously expected a
    cross-strategy summary from the training pipeline. It now reports the
    same payoff signal that drives training, grouped by game so callers can
    inspect convergence per environment. Cross-strategy alignment metrics
    (cooperation, exploitation resistance, adaptability) live in
    ``bench/evaluation/metrics.py`` and operate on tournament results, not
    on training trajectories.
    """
    by_game: Dict[str, List[float]] = {}
    for ep in episode_results:
        game = ep["game"]
        rounds = ep.get("total_rounds") or ep.get("rounds_played") or 0
        if rounds <= EVAL_ZERO:
            continue
        per_round = ep["player_score"] / rounds
        by_game.setdefault(game, []).append(per_round)

    return {
        game: sum(scores) / len(scores)
        for game, scores in by_game.items()
        if scores
    }


# ---------------------------------------------------------------------------
# Per-step shaping
# ---------------------------------------------------------------------------


def per_step_shaping(
    player_payoff: float,
    opponent_payoff: float,  # noqa: ARG001 -- ignored, kept for caller compatibility
    payoff_min: float,
    payoff_max: float,
) -> float:
    """Per-step bonus proportional to the agent's normalised self-payoff.

    Returns ``alpha * (player_payoff - payoff_min) / (payoff_max - payoff_min)``.
    The opponent's payoff does not enter. ``alpha`` is the shaping coefficient
    from ``grpo_constants``; setting it to zero disables shaping.
    """
    payoff_range = payoff_max - payoff_min
    if payoff_range <= EVAL_ZERO_FLOAT:
        return EVAL_ZERO_FLOAT
    alpha = GRPO_SHAPING_ALPHA_NUMERATOR / GRPO_SHAPING_ALPHA_DENOMINATOR
    normalised = (player_payoff - payoff_min) / payoff_range
    return alpha * normalised
