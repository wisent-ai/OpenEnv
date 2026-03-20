"""Unified model interface for external benchmark evaluation."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

from bench.external.constants import EVAL_MAX_NEW_TOKENS, ZERO, ONE
from constant_definitions.train.models.model_constants import API_MODELS

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelHandle:
    """Lightweight wrapper that unifies local HF and API model generation.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id / local path, or API model name.
    model : Any, optional
        Pre-loaded HuggingFace model (avoids reloading).
    tokenizer : Any, optional
        Pre-loaded HuggingFace tokenizer.
    max_new_tokens : int
        Maximum tokens to generate per call.
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

        Dispatches to local HuggingFace generation or API call depending
        on ``is_api_model``.
        """
        if self.is_api_model:
            return self._generate_api(prompt)
        return self._generate_local(prompt)

    # ------------------------------------------------------------------
    # Local HuggingFace generation
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer if not already present."""
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = (
                "transformers is required for local model inference. "
                "Install with: pip install transformers"
            )
            raise ImportError(msg) from exc

        logger.info("Loading model %s", self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
        )

    def _generate_local(self, prompt: str) -> str:
        """Generate with a local HuggingFace model."""
        self.ensure_loaded()
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
        """Generate via an external API (OpenAI or Anthropic)."""
        name = self.model_name_or_path
        if name.startswith("claude"):
            return self._generate_anthropic(prompt)
        return self._generate_openai(prompt)

    def _generate_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            msg = (
                "openai is required for API inference. "
                "Install with: pip install openai"
            )
            raise ImportError(msg) from exc

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
        )
        return response.choices[ZERO].message.content or ""

    def _generate_anthropic(self, prompt: str) -> str:
        try:
            from anthropic import AnthropicVertex
        except ImportError as exc:
            msg = (
                "anthropic[vertex] is required for API inference. "
                "Install with: pip install anthropic[vertex]"
            )
            raise ImportError(msg) from exc

        import os
        project = os.environ.get("GCP_PROJECT", "wisent-480400")
        region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
        client = AnthropicVertex(project_id=project, region=region)
        response = client.messages.create(
            model=self.model_name_or_path,
            max_tokens=self.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[ZERO].text
