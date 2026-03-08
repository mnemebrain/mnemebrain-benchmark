"""Tests for mnemebrain_benchmark.adapters.langchain_buffer."""

from __future__ import annotations

from mnemebrain_benchmark.adapters.langchain_buffer import LangChainBufferBaseline
from mnemebrain_benchmark.interface import Capability


class TestLangChainBufferBaseline:
    def setup_method(self):
        self.adapter = LangChainBufferBaseline()

    def test_name(self):
        assert self.adapter.name() == "langchain_buffer"

    def test_capabilities(self):
        caps = self.adapter.capabilities()
        assert caps == {Capability.STORE, Capability.QUERY}

    def test_store(self):
        result = self.adapter.store("The sky is blue", [{"source_ref": "test"}])
        assert result.belief_id
        assert result.merged is False
        assert result.contradiction_detected is False
        assert result.truth_state is None

    def test_query_empty(self):
        results = self.adapter.query("anything")
        assert results == []

    def test_store_and_query(self):
        self.adapter.store("The sky is blue in summer", [])
        results = self.adapter.query("blue sky summer")
        assert len(results) > 0
        assert "sky" in results[0].claim.lower()

    def test_query_no_match(self):
        self.adapter.store("The sky is blue", [])
        results = self.adapter.query("xyz123")
        assert results == []

    def test_query_long_word_match(self):
        self.adapter.store("Photosynthesis converts sunlight to energy", [])
        results = self.adapter.query("photosynthesis")
        assert len(results) > 0

    def test_reset(self):
        self.adapter.store("test claim", [])
        self.adapter.reset()
        results = self.adapter.query("test claim test claim")
        assert results == []
