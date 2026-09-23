"""Tests for the external benchmark evaluation pipeline.
"""

from __future__ import annotations


import pytest

from bench.external._base import BenchmarkResult
from bench.external._model_handle import ModelHandle
from bench.external.runner_to_staged.runner import ExternalBenchmarkRunner
from constant_definitions.game_constants import EVAL_HALF

# Test fixture values derived from named constants
_TEST_SCORE_A = EVAL_HALF + EVAL_HALF * EVAL_HALF  # derives a test value


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
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
    def test_run_single_unknown(self) -> None:
        handle = ModelHandle(model_name_or_path="mock-model")
        runner = ExternalBenchmarkRunner(
            model_handle=handle, benchmarks=[],
        )
        result = runner.run_single("nonexistent")
        assert result.error is not None
        assert "Unknown" in result.error


