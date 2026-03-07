"""MACHIAVELLI benchmark stub (tier-two, not yet integrated)."""

from __future__ import annotations

import logging
from typing import Any

from bench.external._base import BenchmarkAdapter, BenchmarkResult
from bench.external.constants import BENCH_MACHIAVELLI

logger = logging.getLogger(__name__)


class MachiavelliAdapter(BenchmarkAdapter):
    """Stub adapter for the MACHIAVELLI benchmark.

    This benchmark measures Machiavellian behavior in interactive
    text-based game environments.  Full integration requires the
    ``machiavelli`` package.
    """

    @property
    def name(self) -> str:
        return BENCH_MACHIAVELLI

    @property
    def display_name(self) -> str:
        return "MACHIAVELLI (Stub)"

    def run(self, model_handle: Any) -> BenchmarkResult:
        try:
            import machiavelli  # noqa: F401
        except ImportError:
            return BenchmarkResult(
                benchmark_name=self.name,
                error=(
                    "machiavelli package not installed. "
                    "This is a tier-two benchmark stub."
                ),
            )
        return self._run_episodes(model_handle)

    def _run_episodes(self, model_handle: Any) -> BenchmarkResult:
        """Run MACHIAVELLI game episodes.

        Not yet implemented -- raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "MACHIAVELLI episode runner is not yet implemented."
        )
