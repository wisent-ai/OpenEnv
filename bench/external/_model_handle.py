"""Unified model interface for external benchmark evaluation."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional
from common.machine_to_stado.model_router import chat_completion

from bench.external.constants import EVAL_MAX_NEW_TOKENS, ZERO, ONE
from constant_definitions.train.models.model_constants import API_MODELS

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelHandle:
    """Unify machine-staged local inference and Stado-routed inference.

    ``model_name_or_path`` is either an absolute staged model directory or a
    router model name. A caller may instead supply an already loaded local model
    and tokenizer.
    """

    model_name_or_path: str
    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS

    @property
    def is_api_model(self) -> bool:
        """Return ``True`` if the model is served via an external API."""
        return self.model_name_or_path in API_MODELS

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a completion for *prompt*.

        Dispatches to staged local generation or the Stado model router.
        """
        if self.is_api_model:
            return self._generate_api(prompt)
        return self._generate_local(prompt)

    # ------------------------------------------------------------------
    # Local staged generation
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Lazy-load a machine-staged local model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return
        if "://" in self.model_name_or_path:
            raise ValueError("local model must be a machine-staged directory")
        model_path = Path(self.model_name_or_path).expanduser()
        if not model_path.is_absolute():
            raise ValueError("local model path must be absolute")
        model_path = model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise ValueError(f"local model path is not a directory: {model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = (
                "transformers is required for local model inference. "
                "Install with: pip install transformers"
            )
            raise ImportError(msg) from exc

        logger.info("Loading staged model %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            local_files_only=True,
        )

    def _generate_local(self, prompt: str) -> str:
        """Generate with a machine-staged local model."""
        self.ensure_loaded()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[ONE]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        completion_ids = outputs[ZERO][input_len:]
        return self.tokenizer.decode(
            completion_ids, skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # API generation
    # ------------------------------------------------------------------

    def _generate_api(self, prompt: str) -> str:
        """Generate through the provider-neutral Stado model router."""
        return chat_completion(
            self.model_name_or_path,
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
        )
