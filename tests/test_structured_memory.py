"""Tests for mnemebrain_benchmark.adapters.structured_memory."""
from __future__ import annotations

from helpers import HighSimEmbedder, LowSimEmbedder

from mnemebrain_benchmark.adapters.structured_memory import StructuredMemoryBaseline
from mnemebrain_benchmark.interface import Capability


class TestStructuredMemoryBaseline:
    def test_name(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        assert adapter.name() == "structured_memory"

    def test_capabilities(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        expected = {
            Capability.STORE, Capability.QUERY, Capability.RETRACT,
            Capability.EXPLAIN, Capability.REVISE,
        }
        assert adapter.capabilities() == expected

    # -- store --

    def test_store_new(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        result = adapter.store("test", [{"content": "ev1"}])
        assert result.belief_id
        assert result.merged is False
        assert result.truth_state == "true"
        assert result.confidence == 0.6  # 0.5 + 0.1 * 1 evidence

    def test_store_merged(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder(), threshold=0.9)
        r1 = adapter.store("first", [{"content": "ev1"}])
        r2 = adapter.store("updated", [{"content": "ev2"}])
        assert r2.merged is True
        assert r2.belief_id == r1.belief_id
        assert r2.confidence > r1.confidence  # evidence_count increased

    def test_store_confidence_caps_at_095(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder(), threshold=0.9)
        # Store with 10 pieces of evidence to exceed 0.95 cap
        adapter.store("test", [{"c": str(i)} for i in range(10)])
        result = adapter.store("test", [{"c": "more"}])
        assert result.confidence <= 0.95

    # -- query --

    def test_query_empty(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        assert adapter.query("anything") == []

    def test_query_returns_similar(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        adapter.store("test claim", [{}])
        results = adapter.query("test claim")
        assert len(results) == 1
        assert results[0].truth_state == "true"

    def test_query_skips_deleted(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        r = adapter.store("test", [{}])
        adapter.retract(r.belief_id)
        results = adapter.query("test")
        assert len(results) == 0

    def test_query_skips_dissimilar(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        adapter.store("test", [{}])
        results = adapter.query("test")
        assert len(results) == 0

    # -- retract --

    def test_retract_found(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        r = adapter.store("test", [{}])
        result = adapter.retract(r.belief_id)
        assert result.affected_beliefs == 1

    def test_retract_not_found(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        result = adapter.retract("nonexistent")
        assert result.affected_beliefs == 0

    # -- explain --

    def test_explain_found(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        adapter.store("the sky is blue", [{"c": "ev1"}, {"c": "ev2"}])
        result = adapter.explain("the sky is blue")
        assert result.has_evidence is True
        assert result.supporting_count == 2
        assert result.truth_state == "true"

    def test_explain_not_found(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        result = adapter.explain("unknown")
        assert result.has_evidence is False
        assert result.supporting_count == 0
        assert result.truth_state is None

    def test_explain_skips_deleted(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        r = adapter.store("test", [{"c": "ev"}])
        adapter.retract(r.belief_id)
        result = adapter.explain("test")
        assert result.has_evidence is False

    # -- revise --

    def test_revise_found(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        r = adapter.store("test", [{"c": "ev1"}])
        result = adapter.revise(r.belief_id, [{"c": "ev2"}, {"c": "ev3"}])
        assert result.belief_id == r.belief_id
        assert result.truth_state == "true"
        assert result.confidence > 0.6  # was 0.6, now 0.5 + 0.1*3 = 0.8

    def test_revise_not_found(self):
        adapter = StructuredMemoryBaseline(LowSimEmbedder())
        result = adapter.revise("nonexistent", [{"c": "ev"}])
        assert result.truth_state is None
        assert result.confidence is None

    # -- reset --

    def test_reset(self):
        adapter = StructuredMemoryBaseline(HighSimEmbedder())
        adapter.store("test", [{}])
        adapter.reset()
        assert adapter.query("test") == []
