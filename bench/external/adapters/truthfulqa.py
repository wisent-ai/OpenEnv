"""TruthfulQA benchmark via lm-evaluation-harness."""

from __future__ import annotations

from typing import Any

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.constants import (
    BENCH_TRUTHFULQA,
    LM_EVAL_TRUTHFULQA_TASK,
    ZERO_FLOAT,
)


class TruthfulQAAdapter(BenchmarkAdapter):
    """Evaluate model truthfulness via TruthfulQA (MC variant)."""

    @property
    def name(self) -> str:
        return BENCH_TRUTHFULQA

    @property
    def display_name(self) -> str:
        return "TruthfulQA (MC)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            import lm_eval
        except ImportError as exc:
            msg = (
                "lm-eval is required for TruthfulQA evaluation. "
                "Install with: pip install lm-eval"
            )
            raise ImportError(msg) from exc

        model_handle.ensure_loaded()

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_handle.model_name_or_path}",
            tasks=[LM_EVAL_TRUTHFULQA_TASK],
        )

        task_results = results.get("results", {})
        tqa_data = task_results.get(LM_EVAL_TRUTHFULQA_TASK, {})
        mc_score = tqa_data.get("acc,none", ZERO_FLOAT)

        return BenchmarkResult(
            benchmark_name=self.name,
            scores={"mc_score": mc_score},
            primary_metric="mc_score",
            metadata={"task": LM_EVAL_TRUTHFULQA_TASK},
        )
