"""Tests for train/rewards.py -- reward computation.

The training reward is raw self-payoff. These tests assert that property:
the reward depends only on the agent's own score and the number of rounds,
and is invariant to opponent score, cooperation rate, and any weight kwargs.
"""

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


def test_episode_reward_is_mean_self_payoff():
    """Reward equals player_score / total_rounds."""
    reward = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ONE_FLOAT,
        total_rounds=_TEN,
    )
    assert reward == float(_THIRTY) / _TEN


def test_episode_reward_ignores_cooperation_rate():
    """Cooperation rate must not influence the training reward."""
    reward_cooperative = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ONE_FLOAT,
        total_rounds=_TEN,
    )
    reward_defecting = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_ZERO_FLOAT,
        total_rounds=_TEN,
    )
    assert reward_cooperative == reward_defecting


def test_episode_reward_ignores_opponent_score():
    """Reward must depend on the agent's own payoff alone, not on fairness."""
    reward_equal = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=float(_THIRTY),
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    reward_winning = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    assert reward_equal == reward_winning


def test_episode_reward_monotone_in_self_payoff():
    """Higher self-payoff yields strictly higher reward (round count fixed)."""
    low = episode_reward(
        player_score=float(_FIFTEEN),
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    high = episode_reward(
        player_score=float(_THIRTY),
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_HALF,
        total_rounds=_TEN,
    )
    assert high > low


def test_episode_reward_zero_rounds():
    """Zero rounds returns zero (no division by zero)."""
    reward = episode_reward(
        player_score=EVAL_ZERO_FLOAT,
        opponent_score=EVAL_ZERO_FLOAT,
        cooperation_rate=EVAL_ZERO_FLOAT,
        total_rounds=int(),
    )
    assert reward == EVAL_ZERO_FLOAT


def test_batch_reward_groups_by_game():
    """batch_reward returns mean per-round self-payoff per game."""
    episodes = [
        {
            "game": "prisoners_dilemma",
            "strategy": "always_cooperate",
            "player_score": float(_THIRTY),
            "opponent_score": float(_THIRTY),
            "cooperation_rate": EVAL_ONE_FLOAT,
            "rounds_played": _TEN,
        },
        {
            "game": "prisoners_dilemma",
            "strategy": "always_defect",
            "player_score": float(_TEN),
            "opponent_score": float(_THIRTY),
            "cooperation_rate": EVAL_ZERO_FLOAT,
            "rounds_played": _TEN,
        },
    ]
    result = batch_reward(episodes)
    assert "prisoners_dilemma" in result
    expected = (float(_THIRTY) / _TEN + float(_TEN) / _TEN) / 2.0
    assert result["prisoners_dilemma"] == expected


def test_batch_reward_empty_input():
    """Empty input returns an empty dict, not an error."""
    assert batch_reward([]) == {}


def test_per_step_shaping_uses_self_payoff_only():
    """Shaping bonus tracks the agent's own payoff, not the opponent's."""
    high_self = per_step_shaping(
        player_payoff=float(_FIVE),
        opponent_payoff=EVAL_ZERO_FLOAT,
        payoff_min=EVAL_ZERO_FLOAT,
        payoff_max=float(_FIVE),
    )
    low_self_high_opp = per_step_shaping(
        player_payoff=EVAL_ZERO_FLOAT,
        opponent_payoff=float(_FIVE),
        payoff_min=EVAL_ZERO_FLOAT,
        payoff_max=float(_FIVE),
    )
    assert high_self > low_self_high_opp


def test_per_step_shaping_zero_range():
    """Zero payoff range returns zero (no division by zero)."""
    shaped = per_step_shaping(
        player_payoff=float(_THREE),
        opponent_payoff=float(_THREE),
        payoff_min=float(_FIVE),
        payoff_max=float(_FIVE),
    )
    assert shaped == EVAL_ZERO_FLOAT
