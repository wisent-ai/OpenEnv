"""Frozen opponents and opponent pool for self-play training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional

from env.models import GameAction, GameObservation
from train.agent import PromptBuilder, parse_action
from constant_definitions.train.agent_constants import (
    MAX_ACTION_TOKENS,
    SYSTEM_PROMPT,
)
from constant_definitions.var.meta.self_play_constants import (
    SELF_PLAY_POOL_MAX_SIZE,
)

_ZERO = int()


class FrozenOpponent:
    """Wraps a generation function for use as opponent_fn in KantEnvironment.

    Runs inference with no gradients. Compatible with the
    ``opponent_fn: Callable[[GameObservation], GameAction]`` interface
    that KantEnvironment.reset() accepts.

    Parameters
    ----------
    generate_fn : callable
        A function ``(prompt: str) -> str`` that produces a completion.
    prompt_builder : PromptBuilder, optional
        Custom prompt builder.  Defaults to the standard PromptBuilder.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._builder = prompt_builder or PromptBuilder()

    def __call__(self, obs: GameObservation) -> GameAction:
        """Select an action given a game observation."""
        prompt = self._builder.build(obs)
        completion = self._generate_fn(prompt)
        action_str = parse_action(completion, obs.available_actions)
        return GameAction(action=action_str)

    @classmethod
    def from_model(
        cls,
        model: object,
        tokenizer: object,
        max_tokens: int = MAX_ACTION_TOKENS,
    ) -> FrozenOpponent:
        """Create from an already loaded local model."""
        import torch

        def _generate(prompt: str) -> str:
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt")
                input_len = len(inputs["input_ids"][_ZERO])
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                )
                return tokenizer.decode(
                    outputs[_ZERO][input_len:],
                    skip_special_tokens=True,
                )

        return cls(generate_fn=_generate)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        tokenizer_path: str,
        max_tokens: int = MAX_ACTION_TOKENS,
    ) -> FrozenOpponent:
        """Load a frozen opponent from a saved checkpoint directory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        checkpoint = Path(path).expanduser()
        tokenizer_dir = Path(tokenizer_path).expanduser()
        if not checkpoint.is_absolute() or not tokenizer_dir.is_absolute():
            raise ValueError("checkpoint and tokenizer paths must be absolute")
        checkpoint = checkpoint.resolve(strict=True)
        tokenizer_dir = tokenizer_dir.resolve(strict=True)
        if not checkpoint.is_dir() or not tokenizer_dir.is_dir():
            raise ValueError("checkpoint and tokenizer paths must be directories")

        loaded_model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True
        )
        loaded_model.eval()
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        return cls.from_model(loaded_model, loaded_tokenizer, max_tokens)

    @classmethod
    def from_router(
        cls,
        model_name: str,
        max_tokens: int = MAX_ACTION_TOKENS,
    ) -> FrozenOpponent:
        """Create an API opponent through the authenticated Stado model router."""
        from common.machine_to_stado.model_router import chat_completion

        return cls(generate_fn=lambda prompt: chat_completion(
            model_name,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        ))


class OpponentPool:
    """Maintains a pool of past model checkpoints as diverse opponents.

    Samples uniformly from the pool for opponent diversity.
    Evicts the oldest entry when the pool exceeds ``max_size``.

    Parameters
    ----------
    max_size : int
        Maximum number of frozen opponents to keep in the pool.
    """

    def __init__(self, max_size: int = SELF_PLAY_POOL_MAX_SIZE) -> None:
        self._pool: List[FrozenOpponent] = []
        self._max_size = max_size

    def add(self, opponent: FrozenOpponent) -> None:
        """Add a frozen opponent to the pool, evicting oldest if full."""
        self._pool.append(opponent)
        if len(self._pool) > self._max_size:
            self._pool.pop(_ZERO)

    def sample(self) -> FrozenOpponent:
        """Return a randomly chosen opponent from the pool.

        Raises
        ------
        IndexError
            If the pool is empty.
        """
        if not self._pool:
            raise IndexError("Cannot sample from an empty opponent pool.")
        return random.choice(self._pool)

    def get_opponent_fn(self) -> Callable[[GameObservation], GameAction]:
        """Return a callable that uses a sampled opponent."""
        return self.sample()

    @property
    def size(self) -> int:
        """Current number of opponents in the pool."""
        return len(self._pool)
