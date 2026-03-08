"""LLM agent for N-player game-theory environments."""

from __future__ import annotations

from typing import Callable, List, Optional

from env.nplayer.models import NPlayerAction, NPlayerObservation
from train.agent import parse_action
from constant_definitions.train.agent_constants import (
    MAX_PROMPT_HISTORY_ROUNDS,
    NPLAYER_PROMPT_SECTION_ALL_SCORES,
    NPLAYER_PROMPT_SECTION_PLAYERS,
    NPLAYER_SYSTEM_PROMPT,
    PROMPT_SECTION_ACTIONS,
    PROMPT_SECTION_GAME,
    PROMPT_SECTION_HISTORY,
    PROMPT_SECTION_INSTRUCTION,
    PROMPT_SECTION_SCORES,
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
_PIPE_SEP = " | "
_PLAYER_PREFIX = "Player "
_PLAYED = " played: "
_PAYOFF = " payoff: "
_YOUR_LABEL = "Your score"
_ROUND_LABEL = "Round"
_OF = " of "
_YOU_ARE = "You are Player "
_OUT_OF = " out of "
_PLAYERS = " players"


class NPlayerPromptBuilder:
    """Formats NPlayerObservation into a structured text prompt."""

    @staticmethod
    def build(obs: NPlayerObservation) -> str:
        """Build a structured prompt from an N-player observation."""
        sections: List[str] = []

        # Game section
        sections.append(
            _BRACKET_OPEN + PROMPT_SECTION_GAME + _BRACKET_CLOSE
            + _NEWLINE + obs.game_name
            + _NEWLINE + obs.game_description
        )

        # Players section
        sections.append(
            _BRACKET_OPEN + NPLAYER_PROMPT_SECTION_PLAYERS + _BRACKET_CLOSE
            + _NEWLINE + _YOU_ARE + str(obs.player_index)
            + _OUT_OF + str(obs.num_players) + _PLAYERS
        )

        # History section
        if obs.history:
            history_lines: List[str] = []
            history_slice = obs.history[-MAX_PROMPT_HISTORY_ROUNDS:]
            for rnd in history_slice:
                parts: List[str] = [_ROUND_PREFIX + str(rnd.round_number)]
                for pidx, (act, pay) in enumerate(
                    zip(rnd.actions, rnd.payoffs),
                ):
                    parts.append(
                        _PLAYER_PREFIX + str(pidx)
                        + _PLAYED + act
                        + _PAYOFF + str(pay)
                    )
                history_lines.append(_PIPE_SEP.join(parts))
            sections.append(
                _BRACKET_OPEN + PROMPT_SECTION_HISTORY + _BRACKET_CLOSE
                + _NEWLINE + _NEWLINE.join(history_lines)
            )

        # Scores section
        score_lines: List[str] = []
        for sidx, score in enumerate(obs.scores):
            label = _PLAYER_PREFIX + str(sidx) + _COLON_SPACE + str(score)
            score_lines.append(label)
        sections.append(
            _BRACKET_OPEN + NPLAYER_PROMPT_SECTION_ALL_SCORES + _BRACKET_CLOSE
            + _NEWLINE + _NEWLINE.join(score_lines)
            + _NEWLINE + _ROUND_LABEL + _COLON_SPACE + str(obs.current_round)
            + _OF + str(obs.total_rounds)
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
            + _NEWLINE + NPLAYER_SYSTEM_PROMPT
        )

        return _SECTION_SEP.join(sections)


class NPlayerLLMAgent:
    """LLM-based agent for N-player environments.

    Compatible with NPlayerEnvironment.opponent_fns interface:
    Callable[[NPlayerObservation], NPlayerAction].
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        prompt_builder: Optional[NPlayerPromptBuilder] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._prompt_builder = prompt_builder or NPlayerPromptBuilder()
        self._last_prompt: str = ""
        self._last_completion: str = ""

    def __call__(self, obs: NPlayerObservation) -> NPlayerAction:
        """Select an action given an N-player observation."""
        prompt = self._prompt_builder.build(obs)
        self._last_prompt = prompt
        completion = self._generate_fn(prompt)
        self._last_completion = completion
        action_str = parse_action(completion, obs.available_actions)
        return NPlayerAction(action=action_str)

    @property
    def last_prompt(self) -> str:
        """The most recently constructed prompt."""
        return self._last_prompt

    @property
    def last_completion(self) -> str:
        """The most recent raw model completion."""
        return self._last_completion
