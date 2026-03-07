"""Orchestrator for running external benchmark evaluations."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external._model_handle import ModelHandle
from bench.external.constants import ALL_BENCHMARKS

logger = logging.getLogger(__name__)


class ExternalBenchmarkRunner:
    """Run one or more external benchmarks against a model.

    Parameters
    ----------
    model_handle : ModelHandle
        Unified model interface for generation.
    benchmarks : sequence of str, optional
        Which benchmarks to run.  Defaults to ``ALL_BENCHMARKS``.
    """

    def __init__(
        self,
        model_handle: ModelHandle,
        benchmarks: Optional[Sequence[str]] = None,
    ) -> None:
        self._model_handle = model_handle
        self._benchmark_names = (
            list(benchmarks) if benchmarks is not None
            else list(ALL_BENCHMARKS)
        )
        self._adapters: Dict[str, BenchmarkAdapter] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run every configured benchmark and return results."""
        results: Dict[str, BenchmarkResult] = {}
        for name in self._benchmark_names:
            adapter = self._get_adapter(name)
            if adapter is None:
                continue
            logger.info("Running benchmark: %s", name)
            results[name] = adapter.run_safe(self._model_handle)
        return results

    def run_single(self, name: str) -> BenchmarkResult:
        """Run a single benchmark by name."""
        adapter = self._get_adapter(name)
        if adapter is None:
            return BenchmarkResult(
                benchmark_name=name,
                error=f"Unknown benchmark: {name}",
            )
        return adapter.run_safe(self._model_handle)

    # ------------------------------------------------------------------
    # Adapter registry
    # ------------------------------------------------------------------

    def _get_adapter(self, name: str) -> Optional[BenchmarkAdapter]:
        """Lazily instantiate and cache a benchmark adapter."""
        if name in self._adapters:
            return self._adapters[name]

        adapter = self._create_adapter(name)
        if adapter is not None:
            self._adapters[name] = adapter
        return adapter

    @staticmethod
    def _create_adapter(name: str) -> Optional[BenchmarkAdapter]:
        """Import and instantiate the adapter for *name*."""
        from bench.external.constants import (
            BENCH_ETHICS,
            BENCH_HARMBENCH,
            BENCH_MACHIAVELLI,
            BENCH_MTBENCH,
            BENCH_TRUTHFULQA,
            BENCH_XSTEST,
        )

        if name == BENCH_ETHICS:
            from bench.external.adapters.ethics import EthicsAdapter
            return EthicsAdapter()
        if name == BENCH_TRUTHFULQA:
            from bench.external.adapters.truthfulqa import (
                TruthfulQAAdapter,
            )
            return TruthfulQAAdapter()
        if name == BENCH_HARMBENCH:
            from bench.external.adapters.harmbench import (
                HarmBenchAdapter,
            )
            return HarmBenchAdapter()
        if name == BENCH_XSTEST:
            from bench.external.adapters.xstest import XSTestAdapter
            return XSTestAdapter()
        if name == BENCH_MTBENCH:
            from bench.external.adapters.tier2.mtbench import (
                MTBenchAdapter,
            )
            return MTBenchAdapter()
        if name == BENCH_MACHIAVELLI:
            from bench.external.adapters.tier2.machiavelli import (
                MachiavelliAdapter,
            )
            return MachiavelliAdapter()

        logger.warning("Unknown benchmark: %s", name)
        return None
