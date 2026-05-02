"""Markdown builders for the gradio Payoff Matrices, Reference, and Tournament tabs."""

from .builders import (
    _build_all_matrices_md,
    _build_matrix_md,
    _build_reference_md,
)
from .tournament import run_metrics_tournament

__all__ = [
    "_build_all_matrices_md",
    "_build_matrix_md",
    "_build_reference_md",
    "run_metrics_tournament",
]
