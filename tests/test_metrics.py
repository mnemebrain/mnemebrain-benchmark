"""Tests for mnemebrain_benchmark.metrics."""
from __future__ import annotations

from mnemebrain_benchmark.dataset import ClaimPair
from mnemebrain_benchmark.metrics import (
    PairResult,
    compute_metrics,
    evaluate_pair,
)


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 0.95


class TestEvaluatePair:
    def test_predicted_same(self):
        pair = ClaimPair("p1", "a", "b", "same", "fact", "easy")
        result = evaluate_pair(FakeEmbedder(), pair, threshold=0.9)
        assert result.predicted_same is True
        assert result.actual_same is True
        assert result.similarity == 0.95
        assert result.embed_time_a_ms >= 0
        assert result.embed_time_b_ms >= 0

    def test_predicted_different(self):
        class LowSim:
            def embed(self, text): return [1.0]
            def similarity(self, a, b): return 0.3

        pair = ClaimPair("p1", "a", "b", "different", "fact", "easy")
        result = evaluate_pair(LowSim(), pair, threshold=0.9)
        assert result.predicted_same is False
        assert result.actual_same is False


class TestComputeMetrics:
    def test_empty(self):
        m = compute_metrics([])
        assert m.tp == 0
        assert m.accuracy == 0.0

    def test_perfect_classification(self):
        results = [
            PairResult("p1", 0.95, True, True, 1.0, 1.0),
            PairResult("p2", 0.3, False, False, 1.0, 1.0),
        ]
        m = compute_metrics(results)
        assert m.tp == 1
        assert m.tn == 1
        assert m.fp == 0
        assert m.fn == 0
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_mixed_classification(self):
        results = [
            PairResult("p1", 0.95, True, True, 1.0, 1.0),   # TP
            PairResult("p2", 0.95, True, False, 1.0, 1.0),  # FP
            PairResult("p3", 0.3, False, False, 1.0, 1.0),   # TN
            PairResult("p4", 0.3, False, True, 1.0, 1.0),    # FN
        ]
        m = compute_metrics(results)
        assert m.tp == 1
        assert m.fp == 1
        assert m.tn == 1
        assert m.fn == 1
        assert m.accuracy == 0.5
        assert abs(m.precision - 0.5) < 0.01

    def test_similarity_distributions(self):
        results = [
            PairResult("p1", 0.95, True, True, 1.0, 1.0),
            PairResult("p2", 0.90, True, True, 1.0, 1.0),
            PairResult("p3", 0.30, False, False, 1.0, 1.0),
        ]
        m = compute_metrics(results)
        assert m.mean_sim_same > 0.9
        assert m.mean_sim_different < 0.5
        assert m.separation_gap > 0.5

    def test_latency_metrics(self):
        results = [
            PairResult("p1", 0.95, True, True, 10.0, 12.0),
            PairResult("p2", 0.30, False, False, 8.0, 9.0),
        ]
        m = compute_metrics(results)
        assert m.mean_embed_ms > 0
        assert m.p95_embed_ms > 0
        assert m.throughput_pairs_per_sec > 0

    def test_no_positive_predictions(self):
        results = [
            PairResult("p1", 0.3, False, True, 1.0, 1.0),  # FN
        ]
        m = compute_metrics(results)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
