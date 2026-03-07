"""Tests for train/agent.py -- prompt building and action parsing."""

from __future__ import annotations

from env.models import GameAction, GameObservation, RoundResult
from train.agent import APIAgent, LLMAgent, PromptBuilder, parse_action
from constant_definitions.train.agent_constants import (
    PROMPT_SECTION_ACTIONS,
    PROMPT_SECTION_GAME,
    PROMPT_SECTION_HISTORY,
    PROMPT_SECTION_SCORES,
    SYSTEM_PROMPT,
)
from constant_definitions.game_constants import EVAL_ONE, EVAL_TWO, EVAL_ZERO

_ONE = int(bool(True))
_THREE = EVAL_TWO + EVAL_ONE


def _make_obs(
    game_name: str = "prisoners_dilemma",
    available_actions: list | None = None,
    history: list | None = None,
) -> GameObservation:
    """Helper to build a GameObservation for testing."""
    if available_actions is None:
        available_actions = ["cooperate", "defect"]
    return GameObservation(
        game_name=game_name,
        game_description="A classic social dilemma.",
        available_actions=available_actions,
        current_round=_ONE,
        total_rounds=EVAL_TWO + EVAL_TWO + EVAL_TWO,
        player_score=float(),
        opponent_score=float(),
        history=history or [],
    )


# ── PromptBuilder tests ──


def test_prompt_contains_game_section():
    """Prompt should include the GAME section."""
    obs = _make_obs()
    prompt = PromptBuilder.build(obs)
    assert PROMPT_SECTION_GAME in prompt
    assert "prisoners_dilemma" in prompt


def test_prompt_contains_actions_section():
    """Prompt should list available actions."""
    obs = _make_obs(available_actions=["cooperate", "defect"])
    prompt = PromptBuilder.build(obs)
    assert PROMPT_SECTION_ACTIONS in prompt
    assert "cooperate" in prompt
    assert "defect" in prompt


def test_prompt_contains_scores():
    """Prompt should include score information."""
    obs = _make_obs()
    prompt = PromptBuilder.build(obs)
    assert PROMPT_SECTION_SCORES in prompt


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


def test_prompt_includes_history():
    """When history is present, prompt includes HISTORY section."""
    rnd = RoundResult(
        round_number=_ONE,
        player_action="cooperate",
        opponent_action="defect",
        player_payoff=float(),
        opponent_payoff=float(EVAL_TWO + _THREE),
    )
    obs = _make_obs(history=[rnd])
    prompt = PromptBuilder.build(obs)
    assert PROMPT_SECTION_HISTORY in prompt


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


def test_parse_random_default():
    """Completely unrelated text yields one of the available actions."""
    result = parse_action("banana", ["cooperate", "defect"])
    assert result in ["cooperate", "defect"]


# ── LLMAgent tests ──


def test_llm_agent_callable():
    """LLMAgent should be callable with GameObservation -> GameAction."""
    def mock_generate(prompt: str) -> str:
        return "cooperate"

    agent = LLMAgent(generate_fn=mock_generate)
    obs = _make_obs()
    action = agent(obs)
    assert isinstance(action, GameAction)
    assert action.action == "cooperate"


def test_llm_agent_stores_last_prompt():
    """LLMAgent should store the last prompt for trajectory collection."""
    def mock_generate(prompt: str) -> str:
        return "defect"

    agent = LLMAgent(generate_fn=mock_generate)
    obs = _make_obs()
    agent(obs)
    assert len(agent.last_prompt) > EVAL_ZERO
    assert agent.last_completion == "defect"


# ── APIAgent tests ──


def test_api_agent_passes_system_prompt():
    """APIAgent should pass the system prompt to the API call function."""
    received_system = []

    def mock_api(system: str, user: str) -> str:
        received_system.append(system)
        return "cooperate"

    agent = APIAgent(api_call_fn=mock_api)
    obs = _make_obs()
    agent(obs)
    assert received_system[EVAL_ZERO] == SYSTEM_PROMPT
