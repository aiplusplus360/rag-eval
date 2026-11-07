# rag-eval

**Measure retrieval quality instead of guessing. Recall, precision, MRR and nDCG against a labelled set — with per-query regression detection, so an improving average can't hide a query that got worse.**

[![CI](https://github.com/aiplusplus360/rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/rag-eval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```console
$ rag-eval demo

metric        baseline  candidate    delta
recall           0.667      1.000   +0.333
precision        0.100      0.133   +0.033
mrr              0.667      0.833   +0.167
ndcg             0.640      0.877   +0.237

wins 1   losses 1   ties 1

1 regression(s):
  - export my data                                          1.00 -> 0.50

VERDICT: mixed (aggregate moved, but some queries regressed)
```

Recall went **up 50%**. Every single aggregate metric improved. And one query still got worse — which is precisely the thing a dashboard would have hidden.

---

## Why this exists

RAG quality is usually evaluated by looking at three or four outputs and deciding they seem fine. That works until a change quietly breaks a query nobody happened to try.

The problem with fixing it is that aggregate metrics are *worse than no metrics*, because they look authoritative. A recall@10 that rose from 0.667 to 1.000 reads as an unambiguous win — even when one query dropped from perfect to a miss. Two queries moved; the average is dominated by whichever one had more room to improve.

So this tool reports both:

- **aggregate metrics**, for the trend, and
- **per-query wins/losses/ties**, for the truth

## Install

No dependencies.

```bash
pip install git+https://github.com/aiplusplus360/rag-eval
```

## Usage

Score a labelled set. Cases may carry pre-computed retrieval (`"retrieved"`) or be scored against a live retriever in Python:

```bash
rag-eval score cases.jsonl --k 10
rag-eval score cases.jsonl --json
rag-eval score cases.jsonl --fail-under 0.85   # CI gate
```

```python
from rageval import evaluate, load

report = evaluate(load("cases.jsonl"), retriever=my_search, k=10)
print(report.render())

for miss in report.misses:
    print(miss.query, "wanted", miss.relevant, "got", miss.retrieved[:3])
```

Compare a change:

```bash
rag-eval compare baseline.jsonl candidate.jsonl --fail-on-regression
```

```python
from rageval import compare, evaluate

result = compare(evaluate(base, None, 10), evaluate(cand, None, 10))
print(result.wins, result.losses, result.ties)
print(result.regressions)   # [{query, before, after}, ...]
```

## The eval set

JSONL or a JSON array:

```json
{"query": "how do I rotate an api key", "relevant": ["doc-rotate", "doc-keys"]}
{"query": "what is the rate limit", "relevant": ["doc-limits"]}
```

Optional `"retrieved": ["doc-rotate", "doc-billing"]` to score a recorded run without a live retriever — which is what makes this usable in CI.

**A case with an empty `relevant` list is rejected, not scored.** It is the single most damaging mistake in an eval set: recall computes to `0.0`, so a typo looks exactly like a retrieval failure and sends someone hunting for a bug that does not exist.

## Metrics

| Metric | Question it answers | Formula |
|---|---|---|
| **recall@k** | Did we find the answer at all? | `\|relevant ∩ top_k\| / \|relevant\|` |
| **precision@k** | How much of what we returned was useful? | `\|relevant ∩ top_k\| / k` |
| **MRR** | How far down was the first hit? | `1 / rank_of_first_relevant` |
| **nDCG@k** | Does the ordering reward the good docs? | `DCG / IDCG`, binary relevance |

Each is tested against a **hand-computed** value, not just against itself. For example:

```
retrieved = [x, a], relevant = {a}
DCG  = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.63093
IDCG = 1/log2(2)             = 1.0
nDCG = 0.63093
```

Two deliberate choices worth defending:

**Precision divides by `k`, not by the number returned.** A retriever that returns one document, correctly, should not score `1.0` at `k=10`.

**Relevance is binary.** "Relevant or not" is what a human annotator can be consistent about. Graded relevance produces numbers that look precise and are not reproducible.

## Tail metrics

A mean hides the tail, and the tail is what users experience:

```console
$ rag-eval score cases.jsonl
cases        200   k=10
recall@10   0.950
p95 recall  0.000   (tail, not the mean)
```

`p95 recall` of 0.0 means **1 in 20 queries finds nothing** while the headline reads 95%. Both numbers are true; only one is useful.

## Design notes

**The verdict must not contradict the list.** An early version printed `VERDICT: no significant regression` directly underneath a listed regression, because it only compared `losses > wins`. The verdict now distinguishes regression / mixed / net-improvement-with-losses / clean, so it can never contradict the evidence above it.

**Aggregates and per-query results are both reported, always.** The demo is the argument: every aggregate improved on a change that also broke a query.

**Retrieved lists are deduplicated before scoring.** A duplicate at position 2 shifts everything after it and inflates every rank-based metric below. Deduplication preserves order.

**Cases with no relevant documents are rejected at load time**, with the query quoted in the error so it can be fixed.

**A missing hit is a failed query**, so `--fail-under` and `--fail-on-regression` give CI something real to gate on rather than a metric printed into a log nobody reads.

## Development

```bash
python -m unittest discover -s tests -v   # 48 tests
```

CI runs on Python 3.9, 3.11 and 3.13, and asserts the central claim end to end: an aggregate that improves while a query regresses is reported as `mixed`, not as a win.

## License

[MIT](LICENSE)
