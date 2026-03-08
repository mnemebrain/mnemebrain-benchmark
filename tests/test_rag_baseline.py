"""Tests for mnemebrain_benchmark.adapters.rag_baseline."""
from __future__ import annotations

from mnemebrain_benchmark.adapters.rag_baseline import RAGBaseline
from mnemebrain_benchmark.interface import Capability

from helpers import FakeEmbedder, HighSimEmbedder, LowSimEmbedder


class TestRAGBaseline:
    def test_name(self):
        adapter = RAGBaseline(FakeEmbedder())
        assert adapter.name() == "rag_baseline"

    def test_capabilities(self):
        adapter = RAGBaseline(FakeEmbedder())
        assert adapter.capabilities() == {Capability.STORE, Capability.QUERY}

    def test_store_new(self):
        adapter = RAGBaseline(FakeEmbedder())
        result = adapter.store("test", [])
        assert result.belief_id
        assert result.merged is False

    def test_store_overwrites_on_similarity(self):
        adapter = RAGBaseline(HighSimEmbedder(), threshold=0.9)
        r1 = adapter.store("original claim", [])
        r2 = adapter.store("updated claim", [])
        assert r2.merged is True
        assert r2.belief_id == r1.belief_id
        # The claim should have been overwritten
        results = adapter.query("updated claim")
        assert len(results) == 1
        assert results[0].claim == "updated claim"

    def test_store_no_merge_when_different(self):
        adapter = RAGBaseline(LowSimEmbedder(), threshold=0.9)
        r1 = adapter.store("first", [])
        r2 = adapter.store("second", [])
        assert r2.merged is False
        assert r2.belief_id != r1.belief_id

    def test_query_empty(self):
        adapter = RAGBaseline(FakeEmbedder())
        assert adapter.query("anything") == []

    def test_query_returns_similar(self):
        adapter = RAGBaseline(HighSimEmbedder())
        adapter.store("test", [])
        results = adapter.query("test")
        assert len(results) == 1

    def test_query_skips_dissimilar(self):
        adapter = RAGBaseline(LowSimEmbedder())
        adapter.store("test", [])
        results = adapter.query("test")
        assert len(results) == 0

    def test_reset(self):
        adapter = RAGBaseline(HighSimEmbedder())
        adapter.store("test", [])
        adapter.reset()
        assert adapter.query("test") == []
