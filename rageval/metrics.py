"""Retrieval metrics, implemented once and tested against hand-computed values.

Every metric here answers a different question, and reporting only one of them
is how retrieval quality gets misrepresented:

  recall@k     did we surface the answer at all?
  precision@k  how much of what we surfaced was useful?
  MRR          how far down the list was the first hit?
  nDCG@k       does the *ordering* reward the good documents?

A retriever can improve recall while wrecking nDCG — it found the document, but
buried it at position 9. A report that shows only recall calls that a win.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def _hits(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> List[str]:
    """Top-k retrieved ids that are relevant, in retrieved order."""
    relevant_set = set(relevant)
    return [doc for doc in retrieved[:k] if doc in relevant_set]


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant documents found in the top k.

    Undefined when there are no relevant documents — that is a broken test
    case, not a perfect score, so it returns 0.0 and the dataset validator
    rejects it before it gets here.
    """
    if not relevant:
        return 0.0
    return len(set(_hits(retrieved, relevant, k))) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top k that are relevant.

    Divides by k, not by `len(retrieved)`: a retriever that returns only 1
    document when k=10 should not score 1.0 for that one document.
    """
    if k <= 0:
        return 0.0
    return len(set(_hits(retrieved, relevant, k))) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant document, or 0."""
    relevant_set = set(relevant)
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized discounted cumulative gain, binary relevance.

        DCG  = sum over positions i (1-based) of rel_i / log2(i + 1)
        IDCG = the same sum with every relevant document packed at the front
        nDCG = DCG / IDCG

    Binary relevance is the honest choice for a labelled eval set: "relevant"
    or "not" is what a human annotator can actually be consistent about.
    """
    if not relevant or k <= 0:
        return 0.0

    relevant_set = set(relevant)
    dcg = 0.0
    for position, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(position + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_one(
    retrieved: Sequence[str], relevant: Sequence[str], k: int
) -> Dict[str, float]:
    """All four metrics for a single query."""
    return {
        "recall": recall_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across queries."""
    if not rows:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return {key: mean([row[key] for row in rows]) for key in rows[0]}


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile.

    Averages hide the thing people care about most: the tail. `p95 recall` of
    0.0 means 1 in 20 queries finds nothing, which the mean of 0.85 conceals.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]
