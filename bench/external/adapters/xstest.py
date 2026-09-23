"""XSTest over-refusal detection benchmark."""

from __future__ import annotations

import csv
from typing import Any, Dict, List

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.runner_to_staged._staged import staged_file
from bench.external.constants import (
    BENCH_XSTEST,
    ONE_FLOAT,
    XSTEST_REFUSAL_PHRASES,
    XSTEST_SAFE_LABEL,
    XSTEST_UNSAFE_LABEL,
    ZERO_FLOAT,
)



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
        ds = self._load_staged()

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
    def _load_staged() -> Any:
        """Load XSTest prompts from the machine-staged CSV object."""
        path = staged_file("OPENENV_XSTEST_DATA_PATH")
        rows = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                label = row.get("label", "").strip().lower()
                prompt = row.get("prompt", "").strip()
                if label not in (XSTEST_SAFE_LABEL, XSTEST_UNSAFE_LABEL):
                    raise RuntimeError(f"invalid XSTest label in staged CSV: {label}")
                if not prompt:
                    raise RuntimeError("staged XSTest CSV contains an empty prompt")
                rows.append({"prompt": prompt, "label": label})
        if not rows:
            raise RuntimeError("staged XSTest CSV is empty")
        return rows
