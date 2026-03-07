"""MT-Bench instruction-following quality benchmark."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external._model_handle import ModelHandle
from bench.external.constants import (
    BENCH_MTBENCH,
    MTBENCH_DEFAULT_JUDGE,
    MTBENCH_MAX_SCORE,
    MTBENCH_MIN_SCORE,
    MTBENCH_QUESTIONS_DATASET,
    ONE,
    ZERO,
    ZERO_FLOAT,
)

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the "
    "response provided by an AI assistant to the user question below. "
    "Rate the response on a scale of {min_score} to {max_score}, where "
    "{min_score} is the worst and {max_score} is the best. "
    "Output ONLY the numeric score.\n\n"
    "[Question]\n{question}\n\n"
    "[Response]\n{response}\n\n"
    "Score:"
)


class MTBenchAdapter(BenchmarkAdapter):
    """Evaluate instruction-following quality via MT-Bench questions."""

    @property
    def name(self) -> str:
        return BENCH_MTBENCH

    @property
    def display_name(self) -> str:
        return "MT-Bench (Instruction Following)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            msg = (
                "datasets is required for MT-Bench. "
                "Install with: pip install datasets"
            )
            raise ImportError(msg) from exc

        ds = load_dataset(MTBENCH_QUESTIONS_DATASET, split="train")

        judge_handle = ModelHandle(model_name_or_path=MTBENCH_DEFAULT_JUDGE)

        raw_outputs: List[Dict[str, Any]] = []
        category_scores: Dict[str, List[float]] = {}
        all_scores: List[float] = []

        for row in ds:
            question = row.get("prompt", "")
            category = row.get("category", "general")

            if isinstance(question, list):
                question = question[ZERO] if question else ""

            response = model_handle.generate(question)
            score = self._judge_response(
                judge_handle, question, response,
            )

            if score is not None:
                all_scores.append(score)
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)

            raw_outputs.append({
                "question": question,
                "category": category,
                "response": response,
                "score": score,
            })

        overall_avg = (
            sum(all_scores) / len(all_scores) if all_scores
            else ZERO_FLOAT
        )

        scores: Dict[str, float] = {"overall": overall_avg}
        for cat, cat_scores in category_scores.items():
            scores[f"category_{cat}"] = (
                sum(cat_scores) / len(cat_scores)
            )

        return BenchmarkResult(
            benchmark_name=self.name,
            scores=scores,
            primary_metric="overall",
            metadata={
                "questions_scored": len(all_scores),
                "categories": list(category_scores.keys()),
            },
            raw_outputs=raw_outputs,
        )

    @staticmethod
    def _judge_response(
        judge: ModelHandle,
        question: str,
        response: str,
    ) -> Optional[float]:
        """Score a response using the LLM judge."""
        prompt = _JUDGE_PROMPT.format(
            question=question,
            response=response,
            min_score=MTBENCH_MIN_SCORE,
            max_score=MTBENCH_MAX_SCORE,
        )
        judge_output = judge.generate(prompt)
        return _parse_score(judge_output)


def _parse_score(text: str) -> Optional[float]:
    """Extract a numeric score from judge output."""
    match = re.search(r"\b(\d+)\b", text)
    if match is None:
        return None
    value = int(match.group(ONE))
    if MTBENCH_MIN_SCORE <= value <= MTBENCH_MAX_SCORE:
        return float(value)
    return None
