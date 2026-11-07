"""CLI.

    rag-eval score cases.jsonl --k 10
    rag-eval score cases.jsonl --json
    rag-eval compare baseline.jsonl candidate.jsonl
    rag-eval demo
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .dataset import DatasetError, load, validate_coverage
from .evaluate import compare, evaluate


def cmd_score(args: argparse.Namespace) -> int:
    cases = load(args.cases)
    warnings = validate_coverage(cases)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    report = evaluate(cases, retriever=None, k=args.k)
    report.warnings = warnings

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    # A miss is a failed query. Surfacing it in the exit code lets CI gate on
    # retrieval quality the same way it gates on tests.
    if args.fail_under is not None and report.metrics["recall"] < args.fail_under:
        print(
            f"error: recall {report.metrics['recall']:.3f} is below "
            f"--fail-under {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = evaluate(load(args.baseline), retriever=None, k=args.k)
    candidate = evaluate(load(args.candidate), retriever=None, k=args.k)
    result = compare(baseline, candidate)

    if args.json:
        print(
            json.dumps(
                {
                    "baseline": baseline.metrics,
                    "candidate": candidate.metrics,
                    "deltas": result.deltas,
                    "wins": result.wins,
                    "losses": result.losses,
                    "ties": result.ties,
                    "regressed": result.regressed,
                },
                indent=2,
            )
        )
    else:
        print(result.render())

    return 1 if (args.fail_on_regression and result.regressed) else 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Score a tiny built-in set and show a before/after comparison."""
    baseline = [
        {"query": "how do I rotate an api key",
         "relevant": ["doc-rotate", "doc-keys"],
         "retrieved": ["doc-rotate", "doc-billing", "doc-keys"]},
        {"query": "what is the rate limit",
         "relevant": ["doc-limits"],
         "retrieved": ["doc-billing", "doc-faq", "doc-plans"]},
        {"query": "export my data",
         "relevant": ["doc-export"],
         "retrieved": ["doc-export", "doc-faq"]},
    ]
    candidate = [
        {"query": "how do I rotate an api key",
         "relevant": ["doc-rotate", "doc-keys"],
         "retrieved": ["doc-rotate", "doc-keys", "doc-billing"]},
        {"query": "what is the rate limit",
         "relevant": ["doc-limits"],
         "retrieved": ["doc-limits", "doc-faq"]},
        {"query": "export my data",
         "relevant": ["doc-export"],
         "retrieved": ["doc-faq", "doc-export"]},
    ]

    from .dataset import parse_cases

    base_report = evaluate(parse_cases(json.dumps(baseline)), retriever=None, k=10)
    cand_report = evaluate(parse_cases(json.dumps(candidate)), retriever=None, k=10)

    print("baseline")
    print(base_report.render())
    print("\ncandidate")
    print(cand_report.render())
    print("\n")
    print(compare(base_report, cand_report).render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Measure retrieval quality instead of guessing.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score a labelled set")
    score.add_argument("cases", help="JSONL/JSON of cases, or - for stdin")
    score.add_argument("--k", type=int, default=10)
    score.add_argument("--json", action="store_true")
    score.add_argument(
        "--fail-under", type=float, help="exit 1 if recall is below this"
    )
    score.set_defaults(func=cmd_score)

    cmp_ = sub.add_parser("compare", help="compare two scored sets")
    cmp_.add_argument("baseline")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--k", type=int, default=10)
    cmp_.add_argument("--json", action="store_true")
    cmp_.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when more queries got worse than better",
    )
    cmp_.set_defaults(func=cmd_compare)

    demo = sub.add_parser("demo", help="run a built-in demonstration")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
