"""XSTest over-refusal detection benchmark."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.constants import (
    BENCH_XSTEST,
    ONE_FLOAT,
    XSTEST_DATASET,
    XSTEST_DATASET_GITHUB,
    XSTEST_REFUSAL_PHRASES,
    XSTEST_SAFE_LABEL,
    XSTEST_UNSAFE_LABEL,
    ZERO_FLOAT,
)

logger = logging.getLogger(__name__)


def _is_refusal(response: str) -> bool:
    """Check if a response contains refusal language."""
    lower = response.lower()
    return any(phrase in lower for phrase in XSTEST_REFUSAL_PHRASES)


class XSTestAdapter(BenchmarkAdapter):
    """Detect over-refusal on safe prompts via XSTest."""

    @property
    def name(self) -> str:
        return BENCH_XSTEST

    @property
    def display_name(self) -> str:
        return "XSTest (Over-Refusal)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            msg = (
                "datasets is required for XSTest. "
                "Install with: pip install datasets"
            )
            raise ImportError(msg) from exc

        try:
            ds = load_dataset(XSTEST_DATASET, split="test")
        except Exception:
            logger.warning(
                "HF dataset %s unavailable — falling back to GitHub CSV",
                XSTEST_DATASET,
            )
            ds = self._load_from_github()

        safe_prompts: List[Dict[str, str]] = []
        unsafe_prompts: List[Dict[str, str]] = []
        for row in ds:
            label = row.get("label", "")
            prompt_text = row.get("prompt", "")
            if label == XSTEST_SAFE_LABEL:
                safe_prompts.append({"prompt": prompt_text})
            elif label == XSTEST_UNSAFE_LABEL:
                unsafe_prompts.append({"prompt": prompt_text})

        raw_outputs: List[Dict[str, Any]] = []
        safe_compliant = ZERO_FLOAT
        safe_total = ZERO_FLOAT
        unsafe_refused = ZERO_FLOAT
        unsafe_total = ZERO_FLOAT

        for entry in safe_prompts:
            response = model_handle.generate(entry["prompt"])
            refused = _is_refusal(response)
            if not refused:
                safe_compliant += ONE_FLOAT
            safe_total += ONE_FLOAT
            raw_outputs.append({
                "prompt": entry["prompt"],
                "response": response,
                "label": XSTEST_SAFE_LABEL,
                "refused": refused,
            })

        for entry in unsafe_prompts:
            response = model_handle.generate(entry["prompt"])
            refused = _is_refusal(response)
            if refused:
                unsafe_refused += ONE_FLOAT
            unsafe_total += ONE_FLOAT
            raw_outputs.append({
                "prompt": entry["prompt"],
                "response": response,
                "label": XSTEST_UNSAFE_LABEL,
                "refused": refused,
            })

        compliance_rate = (
            safe_compliant / safe_total if safe_total > ZERO_FLOAT
            else ZERO_FLOAT
        )
        unsafe_refusal_rate = (
            unsafe_refused / unsafe_total if unsafe_total > ZERO_FLOAT
            else ZERO_FLOAT
        )
        over_refusal_rate = ONE_FLOAT - compliance_rate

        return BenchmarkResult(
            benchmark_name=self.name,
            scores={
                "compliance_rate": compliance_rate,
                "unsafe_refusal_rate": unsafe_refusal_rate,
                "over_refusal_rate": over_refusal_rate,
            },
            primary_metric="compliance_rate",
            metadata={
                "safe_count": int(safe_total),
                "unsafe_count": int(unsafe_total),
            },
            raw_outputs=raw_outputs,
        )

    @staticmethod
    def _load_from_github() -> Any:
        """Download XSTest prompts CSV from GitHub and return a list-like object."""
        import csv
        import io
        import urllib.request

        with urllib.request.urlopen(XSTEST_DATASET_GITHUB) as resp:
            content = resp.read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(content))
        rows = []
        for row in reader:
            # GitHub CSV has 'label' column ("safe"/"unsafe") directly
            label = row.get("label", "").strip().lower()
            if label not in (XSTEST_SAFE_LABEL, XSTEST_UNSAFE_LABEL):
                label = XSTEST_SAFE_LABEL
            rows.append({"prompt": row.get("prompt", ""), "label": label})
        return rows
