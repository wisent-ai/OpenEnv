"""Tests for the external benchmark evaluation pipeline.

All tests use mocks -- no real model loading required.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external._model_handle import ModelHandle
from bench.external.constants import ZERO_FLOAT, ONE_FLOAT
from bench.external.report import generate_external_report
from bench.external.runner import ExternalBenchmarkRunner
from constant_definitions.game_constants import EVAL_HALF

# Test fixture values derived from named constants
_TEST_SCORE_A = EVAL_HALF + EVAL_HALF * EVAL_HALF  # derives a test value
_TEST_SCORE_B = EVAL_HALF + EVAL_HALF * EVAL_HALF * EVAL_HALF
_TEST_ELAPSED = EVAL_HALF * EVAL_HALF


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_construction(self) -> None:
        result = BenchmarkResult(
            benchmark_name="test_bench",
            scores={"accuracy": _TEST_SCORE_A},
            primary_metric="accuracy",
        )
        assert result.benchmark_name == "test_bench"
        assert result.primary_metric == "accuracy"

    def test_primary_score_returns_value(self) -> None:
        result = BenchmarkResult(
            benchmark_name="test",
            scores={"acc": _TEST_SCORE_A},
            primary_metric="acc",
        )
        assert result.primary_score == pytest.approx(_TEST_SCORE_A)

    def test_primary_score_none_on_error(self) -> None:
        result = BenchmarkResult(
            benchmark_name="test",
            scores={"acc": _TEST_SCORE_A},
            primary_metric="acc",
            error="something failed",
        )
        assert result.primary_score is None

    def test_primary_score_none_missing_metric(self) -> None:
        result = BenchmarkResult(
            benchmark_name="test",
            scores={},
            primary_metric="nonexistent",
        )
        assert result.primary_score is None


# ---------------------------------------------------------------------------
# BenchmarkAdapter.run_safe tests
# ---------------------------------------------------------------------------


class _FailingAdapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def display_name(self) -> str:
        return "Failing Adapter"

    def run(self, model_handle: Any) -> BenchmarkResult:
        msg = "intentional test failure"
        raise RuntimeError(msg)


class _SuccessAdapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "success"

    @property
    def display_name(self) -> str:
        return "Success Adapter"

    def run(self, model_handle: Any) -> BenchmarkResult:
        return BenchmarkResult(
            benchmark_name=self.name,
            scores={"metric_a": _TEST_SCORE_A},
            primary_metric="metric_a",
        )


class TestRunSafe:
    def test_captures_exception(self) -> None:
        adapter = _FailingAdapter()
        result = adapter.run_safe(model_handle=None)
        assert result.error is not None
        assert "intentional" in result.error
        assert result.elapsed_seconds >= ZERO_FLOAT

    def test_success_sets_elapsed(self) -> None:
        adapter = _SuccessAdapter()
        result = adapter.run_safe(model_handle=None)
        assert result.error is None
        assert result.elapsed_seconds >= ZERO_FLOAT
        assert result.primary_score == pytest.approx(_TEST_SCORE_A)


# ---------------------------------------------------------------------------
# ModelHandle tests
# ---------------------------------------------------------------------------


class TestModelHandle:
    def test_is_api_model_for_claude(self) -> None:
        from constant_definitions.train.models.anthropic_constants import (
            CLAUDE_OPUS,
        )
        handle = ModelHandle(model_name_or_path=CLAUDE_OPUS)
        assert handle.is_api_model is True

    def test_is_api_model_for_gpt(self) -> None:
        from constant_definitions.train.models.openai_constants import (
            GPT_5_4,
        )
        handle = ModelHandle(model_name_or_path=GPT_5_4)
        assert handle.is_api_model is True

    def test_is_not_api_model_for_local(self) -> None:
        from constant_definitions.train.models.local_constants import (
            LLAMA_3_2_1B,
        )
        handle = ModelHandle(model_name_or_path=LLAMA_3_2_1B)
        assert handle.is_api_model is False


# ---------------------------------------------------------------------------
# ExternalBenchmarkRunner tests
# ---------------------------------------------------------------------------


class TestExternalBenchmarkRunner:
    def test_run_all_with_mock_adapter(self) -> None:
        handle = ModelHandle(model_name_or_path="mock-model")
        runner = ExternalBenchmarkRunner(
            model_handle=handle, benchmarks=["success"],
        )
        # Inject our mock adapter
        runner._adapters["success"] = _SuccessAdapter()
        results = runner.run_all()
        assert "success" in results
        assert results["success"].error is None
        assert results["success"].primary_score == pytest.approx(
            _TEST_SCORE_A,
        )

    def test_run_single_unknown(self) -> None:
        handle = ModelHandle(model_name_or_path="mock-model")
        runner = ExternalBenchmarkRunner(
            model_handle=handle, benchmarks=[],
        )
        result = runner.run_single("nonexistent")
        assert result.error is not None
        assert "Unknown" in result.error


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateExternalReport:
    def test_output_format(self) -> None:
        results = {
            "test_bench": BenchmarkResult(
                benchmark_name="test_bench",
                scores={
                    "accuracy": _TEST_SCORE_A,
                    "f_score": _TEST_SCORE_B,
                },
                primary_metric="accuracy",
                elapsed_seconds=_TEST_ELAPSED,
            ),
        }
        json_str, md_str = generate_external_report(
            results, model_name="test-model",
        )

        # JSON is valid
        data = json.loads(json_str)
        assert data["model"] == "test-model"
        assert "summary" in data
        assert "benchmarks" in data

        # Markdown has expected sections
        assert "# External Benchmark Report" in md_str
        assert "## Summary" in md_str
        assert "test_bench" in md_str
