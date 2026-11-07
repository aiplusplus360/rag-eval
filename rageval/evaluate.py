"""Run an eval, and compare two runs honestly.

The comparison is the important half. Aggregate scores hide regressions:
one query improving from 0.2 to 0.9 and another dropping from 0.9 to 0.1 nets
to zero. A mean that did not move is not the same as nothing having changed.

So `compare` reports **per-query wins, losses and ties** alongside the delta,
and flags a run as a regression whenever a *significant* number of queries got
worse — even if the average went up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .dataset import Case
from .metrics import aggregate, evaluate_one, mean, percentile

Retriever = Callable[[str, int], Sequence[str]]


@dataclass
class QueryResult:
    query: str
    relevant: List[str]
    retrieved: List[str]
    scores: Dict[str, float]

    @property
    def hit(self) -> bool:
        return self.scores["recall"] > 0


@dataclass
class Report:
    results: List[QueryResult]
    k: int
    warnings: List[str] = field(default_factory=list)

    @property
    def metrics(self) -> Dict[str, float]:
        return aggregate([r.scores for r in self.results])

    @property
    def misses(self) -> List[QueryResult]:
        """Queries where nothing relevant was retrieved at all.

        Listed explicitly because these are the only rows worth reading in a
        report — the successes need no explanation.
        """
        return [r for r in self.results if not r.hit]

    def tail(self) -> Dict[str, float]:
        return {
            "p50_recall": percentile([r.scores["recall"] for r in self.results], 50),
            "p95_recall": percentile([r.scores["recall"] for r in self.results], 95),
            "worst_mrr": min((r.scores["mrr"] for r in self.results), default=0.0),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "k": self.k,
            "cases": len(self.results),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "tail": {k: round(v, 4) for k, v in self.tail().items()},
            "misses": [
                {"query": r.query, "relevant": r.relevant, "retrieved": r.retrieved[:10]}
                for r in self.misses
            ],
            "warnings": self.warnings,
        }

    def render(self) -> str:
        metrics = self.metrics
        tail = self.tail()
        lines = [
            f"cases        {len(self.results)}   k={self.k}",
            f"recall@{self.k}   {metrics['recall']:.3f}",
            f"precision    {metrics['precision']:.3f}",
            f"mrr          {metrics['mrr']:.3f}",
            f"ndcg@{self.k}    {metrics['ndcg']:.3f}",
            f"p95 recall   {tail['p95_recall']:.3f}   (tail, not the mean)",
        ]
        if self.misses:
            lines.append("")
            lines.append(f"{len(self.misses)} miss(es):")
            for result in self.misses[:10]:
                lines.append(f"  ! {result.query[:60]}")
                lines.append(f"    wanted {result.relevant[:3]}  got {result.retrieved[:3]}")
            if len(self.misses) > 10:
                lines.append(f"  ... and {len(self.misses) - 10} more")
        return "\n".join(lines)


def evaluate(cases: Sequence[Case], retriever: Optional[Retriever], k: int = 10) -> Report:
    """Score every case. Cases may carry their own `retrieved` list."""
    if k <= 0:
        raise ValueError("k must be positive")

    results: List[QueryResult] = []
    for case in cases:
        if case.retrieved is not None:
            retrieved = list(case.retrieved)
        elif retriever is not None:
            retrieved = list(retriever(case.query, k))
        else:
            raise ValueError(
                "no retriever supplied and case has no pre-computed 'retrieved'"
            )

        # Retriever output is untrusted: dedupe while preserving order, because
        # a duplicate at position 2 inflates every rank-based metric below it.
        seen = set()
        deduped = []
        for doc in retrieved:
            if doc not in seen:
                seen.add(doc)
                deduped.append(doc)

        results.append(
            QueryResult(
                query=case.query,
                relevant=list(case.relevant),
                retrieved=deduped,
                scores=evaluate_one(deduped, case.relevant, k),
            )
        )

    return Report(results=results, k=k)


@dataclass
class Comparison:
    baseline: Report
    candidate: Report
    deltas: Dict[str, float]
    wins: int
    losses: int
    ties: int
    regressions: List[Dict[str, object]]
    improvements: List[Dict[str, object]]
    regressed: bool

    def render(self) -> str:
        sign = lambda v: f"{v:+.3f}"
        lines = [
            f"{'metric':<12} {'baseline':>9} {'candidate':>10} {'delta':>8}",
        ]
        for key in ("recall", "precision", "mrr", "ndcg"):
            base = self.baseline.metrics[key]
            cand = self.candidate.metrics[key]
            lines.append(
                f"{key:<12} {base:>9.3f} {cand:>10.3f} {sign(cand - base):>8}"
            )
        lines.append("")
        lines.append(f"wins {self.wins}   losses {self.losses}   ties {self.ties}")

        if self.regressions:
            lines.append("")
            lines.append(f"{len(self.regressions)} regression(s):")
            for row in self.regressions[:10]:
                lines.append(
                    f"  - {str(row['query'])[:55]:<55} "
                    f"{row['before']:.2f} -> {row['after']:.2f}"
                )

        lines.append("")
        # The verdict must not contradict the list above it. A run can improve
        # on average *and* contain a real regression — saying "no regression"
        # while printing one is how a report loses its reader's trust.
        if self.regressed:
            lines.append("VERDICT: regression (more queries got worse than better)")
        elif self.losses and self.wins == self.losses:
            lines.append(
                "VERDICT: mixed (aggregate moved, but some queries regressed)"
            )
        elif self.losses and self.wins > self.losses:
            lines.append("VERDICT: net improvement, with some regressions")
        else:
            lines.append("VERDICT: no significant regression")
        return "\n".join(lines)


def compare(
    baseline: Report, candidate: Report, *, tolerance: float = 1e-9
) -> Comparison:
    """Per-query comparison, not just aggregate deltas.

    A query counts as a loss when its recall dropped by more than `tolerance`,
    or — recall being equal — its MRR dropped. Recall first, because losing the
    document entirely is worse than ranking it lower.
    """
    by_query = {r.query: r for r in baseline.results}

    wins = losses = ties = 0
    regressions: List[Dict[str, object]] = []
    improvements: List[Dict[str, object]] = []

    for result in candidate.results:
        previous = by_query.get(result.query)
        if previous is None:
            continue

        before, after = previous.scores["recall"], result.scores["recall"]
        if abs(after - before) <= tolerance:
            before_mrr, after_mrr = previous.scores["mrr"], result.scores["mrr"]
            if after_mrr > before_mrr + tolerance:
                wins += 1
                improvements.append(
                    {"query": result.query, "before": before_mrr, "after": after_mrr}
                )
            elif after_mrr < before_mrr - tolerance:
                losses += 1
                regressions.append(
                    {"query": result.query, "before": before_mrr, "after": after_mrr}
                )
            else:
                ties += 1
        elif after > before:
            wins += 1
            improvements.append(
                {"query": result.query, "before": before, "after": after}
            )
        else:
            losses += 1
            regressions.append(
                {"query": result.query, "before": before, "after": after}
            )

    deltas = {
        key: candidate.metrics[key] - baseline.metrics[key]
        for key in candidate.metrics
    }

    return Comparison(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        wins=wins,
        losses=losses,
        ties=ties,
        regressions=regressions,
        improvements=improvements,
        regressed=losses > wins,
    )
