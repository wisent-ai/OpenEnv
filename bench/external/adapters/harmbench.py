"""HarmBench jailbreak resistance benchmark."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.constants import (
    BENCH_HARMBENCH,
    HARMBENCH_CLASSIFIER,
    HARMBENCH_DATASET,
    HARMBENCH_MAX_CLASSIFY_TOKENS,
    ONE_FLOAT,
    ZERO,
    ZERO_FLOAT,
)

logger = logging.getLogger(__name__)


class HarmBenchAdapter(BenchmarkAdapter):
    """Measure jailbreak resistance via HarmBench behaviors."""

    @property
    def name(self) -> str:
        return BENCH_HARMBENCH

    @property
    def display_name(self) -> str:
        return "HarmBench (Jailbreak Resistance)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            msg = (
                "datasets is required for HarmBench. "
                "Install with: pip install datasets"
            )
            raise ImportError(msg) from exc

        ds = load_dataset(HARMBENCH_DATASET, split="test")
        behaviors: List[Dict[str, str]] = []
        for row in ds:
            behaviors.append({
                "behavior": row.get("behavior", ""),
                "category": row.get("category", ""),
            })

        classifier = self._load_classifier()

        raw_outputs: List[Dict[str, Any]] = []
        attack_successes = ZERO_FLOAT
        refusals = ZERO_FLOAT
        total = ZERO_FLOAT

        for entry in behaviors:
            prompt = entry["behavior"]
            response = model_handle.generate(prompt)
            is_harmful = self._classify(classifier, prompt, response)

            if is_harmful:
                attack_successes += ONE_FLOAT
            else:
                refusals += ONE_FLOAT
            total += ONE_FLOAT

            raw_outputs.append({
                "behavior": prompt,
                "response": response,
                "is_harmful": is_harmful,
            })

        asr = (
            attack_successes / total if total > ZERO_FLOAT
            else ZERO_FLOAT
        )
        refusal_rate = (
            refusals / total if total > ZERO_FLOAT
            else ZERO_FLOAT
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            scores={
                "attack_success_rate": asr,
                "refusal_rate": refusal_rate,
            },
            primary_metric="refusal_rate",
            metadata={"behavior_count": int(total)},
            raw_outputs=raw_outputs,
        )

    @staticmethod
    def _load_classifier() -> Any:
        """Load the HarmBench classifier pipeline."""
        try:
            from transformers import pipeline
        except ImportError as exc:
            msg = (
                "transformers is required for the HarmBench classifier. "
                "Install with: pip install transformers"
            )
            raise ImportError(msg) from exc

        return pipeline(
            "text-classification",
            model=HARMBENCH_CLASSIFIER,
            max_new_tokens=HARMBENCH_MAX_CLASSIFY_TOKENS,
        )

    @staticmethod
    def _classify(
        classifier: Any, prompt: str, response: str,
    ) -> bool:
        """Return ``True`` if the response is classified as harmful."""
        text = f"Prompt: {prompt}\nResponse: {response}"
        result = classifier(text)
        if isinstance(result, list) and len(result) > ZERO:
            label = result[ZERO].get("label", "")
            return label.lower() == "harmful"
        return False
