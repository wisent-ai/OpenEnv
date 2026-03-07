"""LLM agent for game-theory environments."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from env.models import GameAction, GameObservation
from constant_definitions.train.agent_constants import (
    MAX_ACTION_TOKENS,
    MAX_PROMPT_HISTORY_ROUNDS,
    PARSE_FAILURE_SENTINEL,
    PROMPT_SECTION_ACTIONS,
    PROMPT_SECTION_GAME,
    PROMPT_SECTION_HISTORY,
    PROMPT_SECTION_INSTRUCTION,
    PROMPT_SECTION_SCORES,
    SYSTEM_PROMPT,
    TRAIN_TEMPERATURE_DENOMINATOR,
    TRAIN_TEMPERATURE_NUMERATOR,
)

_ZERO = int()
_ONE = int(bool(True))
_NEWLINE = "\n"
_SECTION_SEP = "\n\n"
_BRACKET_OPEN = "["
_BRACKET_CLOSE = "]"
_COLON_SPACE = ": "
_DASH_SPACE = "- "
_ROUND_PREFIX = "Round "
_YOU_PLAYED = " | You played: "
_OPP_PLAYED = " | Opponent played: "
_YOUR_PAYOFF = " | Your payoff: "
_OPP_PAYOFF = " | Opp payoff: "


class PromptBuilder:
    """Formats GameObservation into a structured text prompt.

    The prompt intentionally excludes the opponent strategy name
    to prevent the model from shortcutting via strategy recognition.
    """

    @staticmethod
    def build(obs: GameObservation) -> str:
        """Build a structured prompt from a game observation."""
        sections: List[str] = []

        # Game section
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_GAME + _BRACKET_CLOSE
            + _NEWLINE + obs.game_name
            + _NEWLINE + obs.game_description
        )

        # History section (limited to last N rounds)
        if obs.history:
            history_lines: List[str] = []
            history_slice = obs.history[-MAX_PROMPT_HISTORY_ROUNDS:]
            for rnd in history_slice:
                line = (
                    _ROUND_PREFIX + str(rnd.round_number)
                    + _YOU_PLAYED + rnd.player_action
                    + _OPP_PLAYED + rnd.opponent_action
                    + _YOUR_PAYOFF + str(rnd.player_payoff)
                    + _OPP_PAYOFF + str(rnd.opponent_payoff)
                )
                history_lines.append(line)
            sections.append(
                _BRACKET_OPEN + PROMPT_SECTION_HISTORY + _BRACKET_CLOSE
                + _NEWLINE + _NEWLINE.join(history_lines)
            )

        # Scores section
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_SCORES + _BRACKET_CLOSE
            + _NEWLINE + "Your score" + _COLON_SPACE + str(obs.player_score)
            + _NEWLINE + "Opponent score" + _COLON_SPACE + str(obs.opponent_score)
            + _NEWLINE + "Round" + _COLON_SPACE + str(obs.current_round)
            + " of " + str(obs.total_rounds)
        )

        # Available actions
        action_lines = [_DASH_SPACE + a for a in obs.available_actions]
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_ACTIONS + _BRACKET_CLOSE
            + _NEWLINE + _NEWLINE.join(action_lines)
        )

        # Instruction
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_INSTRUCTION + _BRACKET_CLOSE
            + _NEWLINE + SYSTEM_PROMPT
        )

        return _SECTION_SEP.join(sections)


def parse_action(response: str, available_actions: List[str]) -> str:
    """Parse an action from LLM response text.

    Tries: exact match -> case-insensitive -> substring -> random selection.
    """
    stripped = response.strip()

    # Exact match
    if stripped in available_actions:
        return stripped

    # Case-insensitive match
    lower = stripped.lower()
    for action in available_actions:
        if action.lower() == lower:
            return action

    # Substring match (response contains action name)
    for action in available_actions:
        if action.lower() in lower:
            return action

    # Random selection as last resort
    return random.choice(available_actions)


class LLMAgent:
    """LLM-based agent compatible with TournamentRunner agent_fn interface.

    Parameters
    ----------
    generate_fn : callable
        A function that takes a prompt string and returns a completion string.
        This abstracts over different model backends (HF, vLLM, API).
    prompt_builder : PromptBuilder, optional
        Custom prompt builder. Defaults to the standard PromptBuilder.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._last_prompt: str = ""
        self._last_completion: str = ""

    def __call__(self, obs: GameObservation) -> GameAction:
        """Select an action given a game observation."""
        prompt = self._prompt_builder.build(obs)
        self._last_prompt = prompt
        completion = self._generate_fn(prompt)
        self._last_completion = completion
        action_str = parse_action(completion, obs.available_actions)
        return GameAction(action=action_str)

    @property
    def last_prompt(self) -> str:
        """The most recently constructed prompt."""
        return self._last_prompt

    @property
    def last_completion(self) -> str:
        """The most recent raw model completion."""
        return self._last_completion


class APIAgent(LLMAgent):
    """Agent that uses an external API (OpenAI/Anthropic) for generation.

    Parameters
    ----------
    api_call_fn : callable
        Function(system_prompt, user_prompt) -> str that calls the API.
    """

    def __init__(
        self,
        api_call_fn: Callable[[str, str], str],
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        def _generate(prompt: str) -> str:
            return api_call_fn(SYSTEM_PROMPT, prompt)

        super().__init__(generate_fn=_generate, prompt_builder=prompt_builder)
