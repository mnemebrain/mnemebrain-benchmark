"""Tests for mnemebrain_benchmark.adapters.naive_baseline."""
from __future__ import annotations

from helpers import FakeEmbedder, HighSimEmbedder, LowSimEmbedder

from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline
from mnemebrain_benchmark.interface import Capability


class TestNaiveBaseline:
    def test_name(self):
        adapter = NaiveBaseline(FakeEmbedder())
        assert adapter.name() == "naive_baseline"

    def test_capabilities(self):
        adapter = NaiveBaseline(FakeEmbedder())
        assert adapter.capabilities() == {Capability.STORE, Capability.QUERY}

    def test_store_new(self):
        adapter = NaiveBaseline(FakeEmbedder())
        result = adapter.store("test claim", [])
        assert result.belief_id
        assert result.merged is False

    def test_store_merged_when_similar(self):
        adapter = NaiveBaseline(HighSimEmbedder(), threshold=0.9)
        r1 = adapter.store("first claim", [])
        r2 = adapter.store("second claim", [])
        assert r2.merged is True
        assert r2.belief_id == r1.belief_id

    def test_store_not_merged_when_different(self):
        adapter = NaiveBaseline(LowSimEmbedder(), threshold=0.9)
        r1 = adapter.store("first", [])
        r2 = adapter.store("second", [])
        assert r2.merged is False
        assert r2.belief_id != r1.belief_id

    def test_query_empty(self):
        adapter = NaiveBaseline(FakeEmbedder())
        assert adapter.query("anything") == []

    def test_query_returns_similar(self):
        adapter = NaiveBaseline(HighSimEmbedder())
        adapter.store("test claim", [])
        results = adapter.query("test claim")
        assert len(results) == 1
        assert results[0].claim == "test claim"

    def test_query_skips_dissimilar(self):
        adapter = NaiveBaseline(LowSimEmbedder())
        adapter.store("test", [])
        results = adapter.query("test")
        assert len(results) == 0

    def test_reset(self):
        adapter = NaiveBaseline(HighSimEmbedder())
        adapter.store("test", [])
        adapter.reset()
        assert adapter.query("test") == []
