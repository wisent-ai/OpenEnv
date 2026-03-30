"""ETHICS commonsense morality benchmark via lm-evaluation-harness."""

from __future__ import annotations

from typing import Any

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.constants import (
    BENCH_ETHICS,
    LM_EVAL_ETHICS_TASK,
    ZERO_FLOAT,
)


class EthicsAdapter(BenchmarkAdapter):
    """Evaluate commonsense moral reasoning via the ETHICS dataset."""

    @property
    def name(self) -> str:
        return BENCH_ETHICS

    @property
    def display_name(self) -> str:
        return "ETHICS (Commonsense Morality)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            import lm_eval
        except ImportError as exc:
            msg = (
                "lm-eval is required for ETHICS evaluation. "
                "Install with: pip install lm-eval"
            )
            raise ImportError(msg) from exc

        model_handle.ensure_loaded()

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_handle.model_name_or_path},trust_remote_code=True",
            tasks=[LM_EVAL_ETHICS_TASK],
        )

        task_results = results.get("results", {})
        ethics_data = task_results.get(LM_EVAL_ETHICS_TASK, {})
        accuracy = ethics_data.get("acc,none", ZERO_FLOAT)

        return BenchmarkResult(
            benchmark_name=self.name,
            scores={"accuracy": accuracy},
            primary_metric="accuracy",
            metadata={"task": LM_EVAL_ETHICS_TASK},
        )
