"""The eval set: load it, validate it, and refuse to score a broken one.

An eval set with a query that has no relevant documents is not a hard query, it
is a typo. Scoring it produces recall 0.0, which drags the average down and
sends someone hunting for a retrieval bug that does not exist.

So validation is strict and happens before any scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class DatasetError(ValueError):
    """The eval set is unusable. Better to say so than to score it."""


@dataclass
class Case:
    query: str
    relevant: List[str]
    # Optional pre-computed retrieval, so a dataset can be scored without a
    # live retriever (CI, a recorded run, a competitor's output).
    retrieved: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_str_list(value: Any, field_name: str, index: int) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DatasetError(f"case {index}: {field_name} must be a list of strings")
    return value


def parse_cases(text: str) -> List[Case]:
    """Parse JSONL or a JSON array of cases."""
    stripped = text.strip()
    if not stripped:
        raise DatasetError("eval set is empty")

    raw_items: List[Any] = []
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise DatasetError("top-level JSON array expected")
        raw_items = data
    else:
        for number, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {number} is not valid JSON: {exc}") from exc

    if not raw_items:
        raise DatasetError("eval set contains no cases")

    cases: List[Case] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise DatasetError(f"case {index}: expected an object")

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise DatasetError(f"case {index}: 'query' must be a non-empty string")

        relevant = _as_str_list(item.get("relevant"), "relevant", index)
        if not relevant:
            # This is the single most damaging mistake in an eval set: it
            # silently reads as a retrieval failure.
            raise DatasetError(
                f"case {index} ({query[:40]!r}): 'relevant' is empty — "
                "a case with no relevant documents cannot be scored"
            )
        if len(set(relevant)) != len(relevant):
            raise DatasetError(f"case {index}: 'relevant' contains duplicates")

        retrieved = None
        if "retrieved" in item and item["retrieved"] is not None:
            retrieved = _as_str_list(item["retrieved"], "retrieved", index)

        metadata = {k: v for k, v in item.items() if k not in {"query", "relevant", "retrieved"}}
        cases.append(Case(query=query, relevant=relevant, retrieved=retrieved, metadata=metadata))

    return cases


def load(path: str) -> List[Case]:
    if path == "-":
        import sys

        return parse_cases(sys.stdin.read())
    try:
        return parse_cases(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc


def validate_coverage(cases: Iterable[Case]) -> List[str]:
    """Non-fatal warnings: duplicate queries, ids never labelled relevant."""
    warnings: List[str] = []
    seen: Dict[str, int] = {}
    for case in cases:
        seen[case.query] = seen.get(case.query, 0) + 1
    for query, count in seen.items():
        if count > 1:
            warnings.append(f"duplicate query appears {count} times: {query[:50]!r}")
    return warnings
