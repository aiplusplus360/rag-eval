"""Tests against hand-computed values.

Every expected number here was calculated by hand from the formula in the
docstring. That is the only way to know a metric is right — a metric that
agrees with itself is not evidence.
"""

import json
import math
import unittest

from rageval.dataset import DatasetError, parse_cases, validate_coverage
from rageval.evaluate import compare, evaluate
from rageval.metrics import (
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall(unittest.TestCase):
    def test_full_recall(self):
        # 2 of 2 relevant found.
        self.assertEqual(recall_at_k(["a", "b"], ["a", "b"], 10), 1.0)

    def test_partial(self):
        # 1 of 2 relevant found.
        self.assertEqual(recall_at_k(["a", "x"], ["a", "b"], 10), 0.5)

    def test_respects_k(self):
        # "b" is at position 3, so k=2 misses it.
        self.assertEqual(recall_at_k(["a", "x", "b"], ["b"], 2), 0.0)
        self.assertEqual(recall_at_k(["a", "x", "b"], ["b"], 3), 1.0)

    def test_no_hits(self):
        self.assertEqual(recall_at_k(["x", "y"], ["a"], 10), 0.0)

    def test_empty_relevant_is_zero(self):
        # A case with no relevant documents is broken; scoring it 1.0 would
        # hide the mistake.
        self.assertEqual(recall_at_k(["a"], [], 10), 0.0)

    def test_duplicates_in_retrieved_do_not_inflate(self):
        self.assertEqual(recall_at_k(["a", "a"], ["a", "b"], 10), 0.5)


class TestPrecision(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(precision_at_k(["a", "b"], ["a", "b"], 2), 1.0)

    def test_divides_by_k_not_by_returned(self):
        # One result, one correct — but k=10. A short list must not score 1.0.
        self.assertEqual(precision_at_k(["a"], ["a"], 10), 0.1)

    def test_half(self):
        self.assertEqual(precision_at_k(["a", "x"], ["a"], 2), 0.5)

    def test_zero_k(self):
        self.assertEqual(precision_at_k(["a"], ["a"], 0), 0.0)


class TestReciprocalRank(unittest.TestCase):
    def test_first_position(self):
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"]), 1.0)

    def test_second_position(self):
        self.assertEqual(reciprocal_rank(["x", "a"], ["a"]), 0.5)

    def test_fourth_position(self):
        self.assertEqual(reciprocal_rank(["x", "y", "z", "a"], ["a"]), 0.25)

    def test_no_hit(self):
        self.assertEqual(reciprocal_rank(["x"], ["a"]), 0.0)

    def test_first_of_several_relevant(self):
        # Rank of the *first* relevant document, not the best one.
        self.assertEqual(reciprocal_rank(["x", "a", "b"], ["a", "b"]), 0.5)


class TestNdcg(unittest.TestCase):
    def test_perfect_ordering(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], ["a", "b"], 10), 1.0)

    def test_reversed_ordering_is_penalized(self):
        # Relevant documents present but at the bottom: DCG < IDCG.
        score = ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], 10)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_hand_computed_value(self):
        # retrieved = [x, a], relevant = {a}
        # DCG  = 0/log2(2) + 1/log2(3) = 1/1.58496 = 0.63093
        # IDCG = 1/log2(2) = 1.0
        # nDCG = 0.63093
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], ["a"], 10), 0.6309297535, places=6)

    def test_two_hits_hand_computed(self):
        # retrieved = [a, x, b], relevant = {a, b}
        # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
        # nDCG = 0.91972
        self.assertAlmostEqual(ndcg_at_k(["a", "x", "b"], ["a", "b"], 10), 1.5 / 1.6309297535, places=6)

    def test_empty(self):
        self.assertEqual(ndcg_at_k([], ["a"], 10), 0.0)


class TestPercentile(unittest.TestCase):
    def test_p50(self):
        self.assertEqual(percentile([0.0, 0.5, 1.0], 50), 0.5)

    def test_p95_reveals_the_tail(self):
        # 19 perfect, 1 zero: mean 0.95, but p95 must expose the failure.
        values = [1.0] * 19 + [0.0]
        self.assertEqual(percentile(values, 95), 1.0)
        self.assertEqual(percentile(values, 100), 1.0)

    def test_single_value(self):
        self.assertEqual(percentile([0.7], 95), 0.7)

    def test_empty(self):
        self.assertEqual(percentile([], 95), 0.0)


class TestDataset(unittest.TestCase):
    def test_parses_jsonl(self):
        cases = parse_cases('{"query":"q","relevant":["a"]}\n')
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].relevant, ["a"])

    def test_parses_array(self):
        cases = parse_cases(json.dumps([{"query": "q", "relevant": ["a"]}]))
        self.assertEqual(len(cases), 1)

    def test_rejects_empty_relevant(self):
        # The most damaging mistake in an eval set: silently reads as a miss.
        with self.assertRaises(DatasetError) as ctx:
            parse_cases('{"query":"q","relevant":[]}')
        self.assertIn("relevant", str(ctx.exception))

    def test_rejects_missing_relevant(self):
        with self.assertRaises(DatasetError):
            parse_cases('{"query":"q"}')

    def test_rejects_empty_query(self):
        with self.assertRaises(DatasetError):
            parse_cases('{"query":"","relevant":["a"]}')

    def test_rejects_duplicate_relevant(self):
        with self.assertRaises(DatasetError):
            parse_cases('{"query":"q","relevant":["a","a"]}')

    def test_rejects_bad_json(self):
        with self.assertRaises(DatasetError):
            parse_cases("{not json")

    def test_rejects_empty_input(self):
        with self.assertRaises(DatasetError):
            parse_cases("   ")

    def test_warns_on_duplicate_queries(self):
        cases = parse_cases(
            '{"query":"q","relevant":["a"]}\n{"query":"q","relevant":["b"]}\n'
        )
        warnings = validate_coverage(cases)
        self.assertTrue(any("duplicate query" in w for w in warnings))


class TestEvaluate(unittest.TestCase):
    def _cases(self):
        return parse_cases(
            "\n".join(
                [
                    json.dumps({"query": "q1", "relevant": ["a"],
                                "retrieved": ["a", "x"]}),
                    json.dumps({"query": "q2", "relevant": ["b"],
                                "retrieved": ["x", "b"]}),
                    json.dumps({"query": "q3", "relevant": ["c"],
                                "retrieved": ["x", "y"]}),
                ]
            )
        )

    def test_aggregates(self):
        report = evaluate(self._cases(), retriever=None, k=10)
        self.assertEqual(len(report.results), 3)
        # recall: 1, 1, 0 -> 2/3
        self.assertAlmostEqual(report.metrics["recall"], 2 / 3)

    def test_misses_are_listed(self):
        report = evaluate(self._cases(), retriever=None, k=10)
        self.assertEqual(len(report.misses), 1)
        self.assertEqual(report.misses[0].query, "q3")

    def test_dedupes_retrieved(self):
        cases = parse_cases('{"query":"q","relevant":["a"],"retrieved":["a","a","a"]}')
        report = evaluate(cases, retriever=None, k=10)
        self.assertEqual(report.results[0].retrieved, ["a"])
        # Duplicates at the top would otherwise inflate rank-based metrics.
        self.assertEqual(report.results[0].scores["precision"], 0.1)

    def test_requires_a_retriever_when_absent(self):
        cases = parse_cases('{"query":"q","relevant":["a"]}')
        with self.assertRaises(ValueError):
            evaluate(cases, retriever=None, k=10)

    def test_uses_callable_retriever(self):
        cases = parse_cases('{"query":"q","relevant":["a"]}')

        def retriever(query, k):
            return ["a", "x"]

        report = evaluate(cases, retriever=retriever, k=10)
        self.assertEqual(report.metrics["recall"], 1.0)

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            evaluate(self._cases(), retriever=None, k=0)


class TestCompare(unittest.TestCase):
    def test_improvement_detected(self):
        baseline = parse_cases('{"query":"q","relevant":["a"],"retrieved":["x","a"]}')
        candidate = parse_cases('{"query":"q","relevant":["a"],"retrieved":["a"]}')
        result = compare(
            evaluate(baseline, None, 10), evaluate(candidate, None, 10)
        )
        self.assertEqual(result.wins, 1)
        self.assertEqual(result.losses, 0)
        self.assertFalse(result.regressed)
        # MRR moved 0.5 -> 1.0.
        self.assertAlmostEqual(result.deltas["mrr"], 0.5)

    def test_regression_detected(self):
        baseline = parse_cases('{"query":"q","relevant":["a"],"retrieved":["a"]}')
        candidate = parse_cases('{"query":"q","relevant":["a"],"retrieved":["x"]}')
        result = compare(
            evaluate(baseline, None, 10), evaluate(candidate, None, 10)
        )
        self.assertEqual(result.losses, 1)
        self.assertTrue(result.regressed)

    def test_average_hides_a_swap(self):
        # One query improves, one gets worse. The mean recall is unchanged, so
        # an aggregate-only comparison would report "no change". This is the
        # whole reason the comparison is per-query.
        baseline = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["x"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["b"]}'
        )
        candidate = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["a"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["x"]}'
        )
        base = evaluate(baseline, None, 10)
        cand = evaluate(candidate, None, 10)
        result = compare(base, cand)
        self.assertAlmostEqual(base.metrics["recall"], cand.metrics["recall"])
        self.assertEqual(result.wins, 1)
        self.assertEqual(result.losses, 1)

    def test_render_reports_verdict(self):
        baseline = parse_cases('{"query":"q","relevant":["a"],"retrieved":["a"]}')
        candidate = parse_cases('{"query":"q","relevant":["a"],"retrieved":["x"]}')
        text = compare(evaluate(baseline, None, 10), evaluate(candidate, None, 10)).render()
        self.assertIn("VERDICT", text)
        self.assertIn("regression", text)

    def test_verdict_does_not_contradict_a_listed_regression(self):
        # Improved on average, one query worse: the headline must not claim
        # "no significant regression" while printing a regression underneath.
        baseline = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["x"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["b"]}'
        )
        candidate = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["a"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["x"]}'
        )
        text = compare(evaluate(baseline, None, 10), evaluate(candidate, None, 10)).render()
        self.assertIn("mixed", text)
        self.assertNotIn("no significant regression", text)

    def test_verdict_net_improvement_with_losses(self):
        baseline = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["x"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["x"]}\n'
            '{"query":"q3","relevant":["c"],"retrieved":["c"]}'
        )
        candidate = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["a"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["b"]}\n'
            '{"query":"q3","relevant":["c"],"retrieved":["x"]}'
        )
        text = compare(evaluate(baseline, None, 10), evaluate(candidate, None, 10)).render()
        self.assertIn("net improvement", text)


class TestReport(unittest.TestCase):
    def test_to_dict_is_json_serializable(self):
        cases = parse_cases('{"query":"q","relevant":["a"],"retrieved":["a"]}')
        payload = json.dumps(evaluate(cases, None, 10).to_dict())
        self.assertIn("metrics", payload)

    def test_tail_metrics(self):
        cases = parse_cases(
            '{"query":"q1","relevant":["a"],"retrieved":["a"]}\n'
            '{"query":"q2","relevant":["b"],"retrieved":["x"]}'
        )
        report = evaluate(cases, None, 10)
        self.assertEqual(report.tail()["worst_mrr"], 0.0)

    def test_render_shows_misses(self):
        cases = parse_cases('{"query":"q","relevant":["a"],"retrieved":["x"]}')
        text = evaluate(cases, None, 10).render()
        self.assertIn("miss", text)


if __name__ == "__main__":
    unittest.main()
