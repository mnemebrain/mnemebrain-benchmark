"""Tests for mnemebrain_benchmark.adapters.openai_rag_adapter (mocked OpenAI)."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from mnemebrain_benchmark.interface import Capability


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Inject a fake openai module so we can import the adapter without the real package."""
    fake_openai = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[1.0, 0.0, 0.0])]
            self.embeddings.create.return_value = mock_resp

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai

    # Force reimport of the adapter module
    mod_name = "mnemebrain_benchmark.adapters.openai_rag_adapter"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    yield fake_openai

    del sys.modules["openai"]
    if mod_name in sys.modules:
        del sys.modules[mod_name]


def _get_adapter_class():
    from mnemebrain_benchmark.adapters.openai_rag_adapter import OpenAIRAGAdapter
    return OpenAIRAGAdapter


class TestOpenAIRAGAdapter:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        Cls = _get_adapter_class()
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Cls()

    def test_name(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        assert adapter.name() == "openai_rag"

    def test_capabilities(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        assert adapter.capabilities() == {Capability.STORE, Capability.QUERY}

    def test_store_new(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        result = adapter.store("test claim", [])
        assert result.belief_id
        assert result.merged is False

    def test_store_merges_similar(self):
        """Same embedding = similarity 1.0 >= threshold 0.8 → merge."""
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        r1 = adapter.store("first", [])
        r2 = adapter.store("second", [])
        assert r2.merged is True
        assert r2.belief_id == r1.belief_id

    def test_query_empty(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        assert adapter.query("anything") == []

    def test_query_returns_similar(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        adapter.store("test", [])
        results = adapter.query("test")
        assert len(results) == 1

    def test_reset(self):
        Cls = _get_adapter_class()
        adapter = Cls(api_key="test-key")
        adapter.store("test", [])
        adapter.reset()
        assert adapter.query("test") == []


class TestCosineSimHelper:
    def test_identical_vectors(self):
        from mnemebrain_benchmark.adapters.openai_rag_adapter import _cosine_sim
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        from mnemebrain_benchmark.adapters.openai_rag_adapter import _cosine_sim
        assert abs(_cosine_sim([1.0, 0.0], [0.0, 1.0])) < 0.001

    def test_zero_vector(self):
        from mnemebrain_benchmark.adapters.openai_rag_adapter import _cosine_sim
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0
