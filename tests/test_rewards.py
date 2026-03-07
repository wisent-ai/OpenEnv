"""Tests for train/rewards.py -- reward computation."""

from __future__ import annotations

from constant_definitions.game_constants import (
    EVAL_HALF,
    EVAL_ONE,
    EVAL_ONE_FLOAT,
    EVAL_TWO,
    EVAL_ZERO_FLOAT,
)
from train.rewards import batch_reward, episode_reward, per_step_shaping

_TEN = EVAL_TWO + EVAL_TWO + EVAL_TWO + EVAL_TWO + EVAL_TWO
_THREE = EVAL_TWO + EVAL_ONE
_THIRTY = _TEN * _THREE
_FIVE = EVAL_TWO + EVAL_TWO + EVAL_ONE
_FIFTEEN = _FIVE * _THREE


def test_episode_reward_full_cooperation():
    """Full cooperation with equal scores should yield a high reward."""
    reward = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ONE_FLOAT,
        total_rounds=_TEN,
    )
    # cooperation_rate = EVAL_ONE, fairness = EVAL_ONE (equal scores),
    # pareto and cross-strategy defaults to EVAL_HALF
    assert reward > EVAL_HALF


def test_episode_reward_zero_cooperation():
    """Zero cooperation rate should reduce the reward."""
    reward_coop = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ONE_FLOAT,
        total_rounds=_TEN,
    )
    reward_defect = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ZERO_FLOAT,
        total_rounds=_TEN,
    )
    assert reward_coop > reward_defect


def test_episode_reward_unfair_reduces_score():
    """Unequal scores should reduce the fairness component."""
    reward_fair = episode_reward(
        player_score=float(_FIFTEEN),
        opponent_score=float(_FIFTEEN),
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    reward_unfair = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    assert reward_fair > reward_unfair


def test_episode_reward_zero_rounds():
    """Zero rounds should not cause division by zero."""
    reward = episode_reward(
        player_score=EVAL_ZERO_FLOAT,
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_ZERO_FLOAT,
        total_rounds=int(),
    )
    assert isinstance(reward, float)


def test_batch_reward_returns_both_metrics():
    """batch_reward should return exploitation_resistance and adaptability."""
    episodes = [
        {
            "game": "prisoners_dilemma",
            "strategy": "always_cooperate",
            "player_score": float(_THIRTY),
            "opponent_score": float(_THIRTY),
            "cooperation_rate": EVAL_ONE_FLOAT,
        },
        {
            "game": "prisoners_dilemma",
            "strategy": "always_defect",
            "player_score": float(_TEN),
            "opponent_score": float(_THIRTY),
            "cooperation_rate": EVAL_ZERO_FLOAT,
        },
    ]
    result = batch_reward(episodes)
    assert "exploitation_resistance" in result
    assert "adaptability" in result


def test_batch_reward_empty_input():
    """Empty input should return defaults without error."""
    result = batch_reward([])
    assert "exploitation_resistance" in result
    assert "adaptability" in result


def test_per_step_shaping_range():
    """Per-step shaping should be bounded and non-negative for positive payoffs."""
    shaped = per_step_shaping(
        player_payoff=float(_THREE),
        opponent_payoff=float(_THREE),
        payoff_min=EVAL_ZERO_FLOAT,
        payoff_max=float(_FIVE),
    )
    assert shaped >= EVAL_ZERO_FLOAT


def test_per_step_shaping_zero_range():
    """Zero payoff range should return zero."""
    shaped = per_step_shaping(
        player_payoff=float(_THREE),
        opponent_payoff=float(_THREE),
        payoff_min=float(_FIVE),
        payoff_max=float(_FIVE),
    )
    assert shaped == EVAL_ZERO_FLOAT
