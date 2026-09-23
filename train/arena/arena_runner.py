"""ArenaRunner — wires real AI models into the MetagameArena."""
from __future__ import annotations

from typing import Any, Callable, Optional

from env.arena.engine import MetagameArena
from env.arena.models import ArenaRoundResult
from constant_definitions.arena.arena_constants import (
    DEFAULT_TOTAL_ROUNDS,
    MODEL_TYPE_API,
    MODEL_TYPE_LOCAL,
    MODEL_TYPE_STRATEGY,
)
from constant_definitions.train.agent_constants import SYSTEM_PROMPT
from common.machine_to_stado.model_router import chat_completion

try:
    from constant_definitions.train.models.anthropic_constants import CLAUDE_HAIKU
except ImportError:
    CLAUDE_HAIKU = "claude-haiku"

try:
    from constant_definitions.train.models.openai_constants import GPT_5_4
except ImportError:
    GPT_5_4 = "gpt-latest"


def _make_anthropic_fn(model: str) -> Callable[[str], str]:
    """Create a generate function backed by the Stado model router."""
    def _generate(prompt: str) -> str:
        return chat_completion(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            max_tokens=int("255"),
        )
    return _generate


def _make_openai_fn(model: str) -> Callable[[str], str]:
    """Create a generate function backed by the Stado model router."""
    def _generate(prompt: str) -> str:
        return chat_completion(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
        )
    return _generate


class ArenaRunner:
    """Wires real model backends into a MetagameArena and runs it."""

    def __init__(self, total_rounds: int = DEFAULT_TOTAL_ROUNDS) -> None:
        self._arena = MetagameArena(total_rounds=total_rounds)
        self._configs: list[dict[str, Any]] = []

    @property
    def arena(self) -> MetagameArena:
        return self._arena

    def add_anthropic_model(
        self, model_id: str, model_name: str = "",
    ) -> None:
        """Add an Anthropic-routed model."""
        name = model_name or CLAUDE_HAIKU
        fn = _make_anthropic_fn(name)
        self._arena.add_model(model_id, fn, MODEL_TYPE_API)

    def add_openai_model(
        self, model_id: str, model_name: str = "",
    ) -> None:
        """Add an OpenAI-routed model."""
        name = model_name or GPT_5_4
        fn = _make_openai_fn(name)
        self._arena.add_model(model_id, fn, MODEL_TYPE_API)

    def add_local_model(
        self, model_id: str, generate_fn: Callable[[str], str],
    ) -> None:
        """Add a local model with a custom generate function."""
        self._arena.add_model(model_id, generate_fn, MODEL_TYPE_LOCAL)

    def add_strategy_model(
        self, model_id: str, strategy_action: str = "cooperate",
    ) -> None:
        """Add a deterministic strategy wrapped as a generate function."""
        action = strategy_action

        def _generate(prompt: str) -> str:
            return action

        self._arena.add_model(model_id, _generate, MODEL_TYPE_STRATEGY)

    def run(self) -> list[ArenaRoundResult]:
        """Execute the full arena."""
        return self._arena.run_full_arena()
