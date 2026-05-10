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

        # Free-chat: render the opponent's most-recent free-form message
        # verbatim into the prompt so the agent can react to natural-
        # language signaling. Driven by env-side metadata, so any game
        # that becomes a free-chat variant via apply_free_chat picks this
        # up automatically without per-game prompt-builder changes.
        last_opp_msg = (obs.metadata or {}).get("last_opp_message", "")
        if last_opp_msg:
            sections.append("[Opponent said last round]\n" + last_opp_msg)

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

        # Instruction. For free-chat games (signaled by metadata flag the
        # env sets, OR by presence of last_opp_message earlier in this
        # episode) ask for MESSAGE + ACTION two-line format instead of
        # the bare-action SYSTEM_PROMPT.
        is_free_chat = bool(
            (obs.metadata or {}).get("last_opp_message")
            or (obs.metadata or {}).get("free_chat")
        )
        if is_free_chat:
            instruction = (
                "Write ONE short message to your opponent on the first line, "
                "starting with 'MESSAGE:'. Then on a new line write exactly "
                "'ACTION: <action>' where <action> is one of the available "
                "actions listed above. The message is non-binding cheap talk; "
                "only the action affects payoff."
            )
        else:
            instruction = SYSTEM_PROMPT
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_INSTRUCTION + _BRACKET_CLOSE
            + _NEWLINE + instruction
        )

        return _SECTION_SEP.join(sections)


_FREE_CHAT_ACTION_RE = None


def _free_chat_action_re():
    """Lazy compile of the MESSAGE/ACTION regex used by parse_free_chat."""
    global _FREE_CHAT_ACTION_RE
    if _FREE_CHAT_ACTION_RE is None:
        import re
        _FREE_CHAT_ACTION_RE = re.compile(
            r"ACTION\s*[:\-]\s*([A-Za-z0-9_\-]+)", re.IGNORECASE,
        )
    return _FREE_CHAT_ACTION_RE


def parse_free_chat(response: str, available_actions: List[str]) -> tuple[str, str]:
    """Parse a free-chat completion of the form ``MESSAGE: <text>\\nACTION: <token>``.

    Returns ``(action, message)``. The action is matched against
    ``available_actions`` by the same exact / case-insensitive / substring
    cascade as :func:`parse_action`. The message is everything before the
    matched ACTION line; if no ACTION line is present the entire response
    is treated as the message and the action falls back to substring match
    on the raw response (so a response that's only a bare token still
    parses).
    """
    text = response or ""
    m = _free_chat_action_re().search(text)
    if m:
        message = text[: m.start()].strip()
        # Strip a leading 'MESSAGE:' prefix the model often emits.
        if message.lower().startswith("message:"):
            message = message[len("message:"):].strip()
        action = parse_action(m.group(1), available_actions)
        return action, message
    return parse_action(text, available_actions), text.strip()


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
        # Detect free-chat games via the prompt's instruction shape.
        # PromptBuilder injects a 'MESSAGE / ACTION' instruction when the
        # variant is active (it can read obs.metadata flags or the game's
        # applied_variants). The cheap test: if the prompt asks for a
        # MESSAGE line, parse the response with the free-chat path.
        if "MESSAGE:" in prompt and "ACTION:" in prompt:
            action_str, msg = parse_free_chat(completion, obs.available_actions)
            return GameAction(action=action_str, metadata={"message": msg})
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
