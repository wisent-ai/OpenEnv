"""Tests for train/agent.py -- prompt building and action parsing."""

from __future__ import annotations

from env.models import GameObservation
from train.agent import PromptBuilder, parse_action
from constant_definitions.game_constants import EVAL_TWO

_ONE = int(bool(True))


def test_prompt_excludes_opponent_strategy():
    """Prompt should NOT include opponent strategy name."""
    obs = GameObservation(
        game_name="prisoners_dilemma",
        game_description="test",
        available_actions=["cooperate", "defect"],
        current_round=_ONE,
        total_rounds=EVAL_TWO + EVAL_TWO + EVAL_TWO,
        opponent_strategy="tit_for_tat",
    )
    prompt = PromptBuilder.build(obs)
    assert "tit_for_tat" not in prompt


# ── parse_action tests ──


def test_parse_exact_match():
    """Exact string match should work."""
    result = parse_action("cooperate", ["cooperate", "defect"])
    assert result == "cooperate"


def test_parse_case_insensitive():
    """Case-insensitive match should work."""
    result = parse_action("COOPERATE", ["cooperate", "defect"])
    assert result == "cooperate"


def test_parse_substring():
    """Substring match: response containing action name."""
    result = parse_action("I will cooperate this round", ["cooperate", "defect"])
    assert result == "cooperate"


def test_parse_raises_on_unmatched():
    """Off-vocabulary response must raise ParseActionError, not silently
    substitute a random action. Was previously a random.choice fallback
    that contaminated action distributions with coin-flip tokens."""
    import pytest
    from train.agent import ParseActionError
    with pytest.raises(ParseActionError):
        parse_action("banana", ["cooperate", "defect"])


