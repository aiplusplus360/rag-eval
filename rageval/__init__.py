"""Retrieval evaluation you can put in CI."""

__version__ = "0.1.0"

from .dataset import Case, DatasetError, load, parse_cases
from .evaluate import Comparison, Report, compare, evaluate
from .metrics import (
    aggregate,
    evaluate_one,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "Case",
    "Comparison",
    "DatasetError",
    "Report",
    "aggregate",
    "compare",
    "evaluate",
    "evaluate_one",
    "load",
    "mean",
    "ndcg_at_k",
    "parse_cases",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
