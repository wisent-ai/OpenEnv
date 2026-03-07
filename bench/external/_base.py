"""Core abstractions for external benchmark adapters."""

from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from bench.external.constants import ZERO_FLOAT

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BenchmarkResult:
    """Result from running a single external benchmark.

    Parameters
    ----------
    benchmark_name : str
        Machine-readable benchmark identifier.
    scores : dict
        Metric name to float value mapping.
    primary_metric : str
        Key into *scores* for the single headline number.
    metadata : dict
        Arbitrary extra info (dataset version, sample count, etc.).
    raw_outputs : list
        Per-sample outputs for debugging / qualitative review.
    elapsed_seconds : float
        Wall-clock time for the benchmark run.
    error : str or None
        If the run failed, a description of the error.
    """

    benchmark_name: str
    scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    primary_metric: str = ""
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    raw_outputs: list = dataclasses.field(default_factory=list)
    elapsed_seconds: float = ZERO_FLOAT
    error: Optional[str] = None

    @property
    def primary_score(self) -> Optional[float]:
        """Return the primary metric value, or ``None`` on error."""
        if self.error is not None:
            return None
        return self.scores.get(self.primary_metric)


class BenchmarkAdapter(ABC):
    """Abstract base class for external benchmark integrations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable benchmark name."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable benchmark name."""

    @abstractmethod
    def run(self, model_handle: Any) -> BenchmarkResult:
        """Execute the benchmark and return results.

        Parameters
        ----------
        model_handle : ModelHandle
            Unified model interface for generation.

        Returns
        -------
        BenchmarkResult
        """

    def run_safe(self, model_handle: Any) -> BenchmarkResult:
        """Execute the benchmark, catching any exception.

        Returns a ``BenchmarkResult`` with the *error* field populated on
        failure so that the overall pipeline never crashes.
        """
        start = time.monotonic()
        try:
            result = self.run(model_handle)
            result.elapsed_seconds = time.monotonic() - start
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            logger.exception("Benchmark %s failed", self.name)
            return BenchmarkResult(
                benchmark_name=self.name,
                error=str(exc),
                elapsed_seconds=elapsed,
            )
