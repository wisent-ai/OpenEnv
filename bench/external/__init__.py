"""External benchmark evaluation pipeline for safety transfer testing."""

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkResult",
    "ExternalBenchmarkRunner",
    "ModelHandle",
    "generate_external_report",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid pulling in heavy deps at package load time."""
    if name in ("BenchmarkAdapter", "BenchmarkResult"):
        from bench.external._base import BenchmarkAdapter, BenchmarkResult
        _map = {
            "BenchmarkAdapter": BenchmarkAdapter,
            "BenchmarkResult": BenchmarkResult,
        }
        return _map[name]
    if name == "ModelHandle":
        from bench.external._model_handle import ModelHandle
        return ModelHandle
    if name == "ExternalBenchmarkRunner":
        from bench.external.runner import ExternalBenchmarkRunner
        return ExternalBenchmarkRunner
    if name == "generate_external_report":
        from bench.external.report import generate_external_report
        return generate_external_report
    msg = f"module 'bench.external' has no attribute {name!r}"
    raise AttributeError(msg)
