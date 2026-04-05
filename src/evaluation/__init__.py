"""Evaluation module for RAG system."""

from .metrics import (
    exact_match_score,
    f1_score,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_metrics,
    aggregate_metrics,
    FActScorer
)

__all__ = [
    "exact_match_score",
    "f1_score",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_metrics",
    "aggregate_metrics",
    "FActScorer"
]
