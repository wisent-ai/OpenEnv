"""HarmBench jailbreak resistance benchmark."""

from __future__ import annotations

import csv
from typing import Any, Dict, List

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.runner_to_staged._staged import staged_directory, staged_file
from bench.external.constants import (
    BENCH_HARMBENCH,
    HARMBENCH_MAX_CLASSIFY_TOKENS,
    ONE_FLOAT,
    ZERO,
    ZERO_FLOAT,
)



class HarmBenchAdapter(BenchmarkAdapter):
    """Measure jailbreak resistance via HarmBench behaviors."""

    @property
    def name(self) -> str:
        return BENCH_HARMBENCH

    @property
    def display_name(self) -> str:
        return "HarmBench (Jailbreak Resistance)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        behaviors = self._load_behaviors()

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
    def _load_behaviors() -> List[Dict[str, str]]:
        """Load HarmBench behavior rows from the staged CSV object."""
        path = staged_file("OPENENV_HARMBENCH_DATA_PATH")
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            behaviors = [{
                "behavior": row.get("Behavior", row.get("behavior", "")),
                "category": row.get(
                    "SemanticCategory", row.get("FunctionalCategory", "")
                ),
            } for row in reader]
        if not behaviors or any(not row["behavior"] for row in behaviors):
            raise RuntimeError("staged HarmBench CSV contains invalid behavior rows")
        return behaviors

    @staticmethod
    def _load_classifier() -> Any:
        """Load the machine-staged HarmBench classifier offline."""
        from transformers import pipeline

        classifier_path = staged_directory(
            "OPENENV_HARMBENCH_CLASSIFIER_PATH"
        )
        return pipeline(
            "text-classification",
            model=str(classifier_path),
            max_new_tokens=HARMBENCH_MAX_CLASSIFY_TOKENS,
            model_kwargs={"local_files_only": True},
        )

    @staticmethod
    def _classify(
        classifier: Any, prompt: str, response: str,
    ) -> bool:
        """Return ``True`` when the staged classifier marks a response harmful."""

        text = f"Prompt: {prompt}\nResponse: {response}"
        result = classifier(text)
        if isinstance(result, list) and len(result) > ZERO:
            label = result[ZERO].get("label", "")
            return label.lower() == "harmful"
        return False
