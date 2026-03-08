"""Tests for mnemebrain_benchmark.adapters.mem0_adapter (mocked)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# Mock the mem0 package before importing the adapter
@pytest.fixture(autouse=True)
def mock_mem0():
    """Inject a fake mem0 module so we can import the adapter without mem0ai."""
    fake_mem0 = ModuleType("mem0")
    fake_client = MagicMock()
    fake_client.project.update = MagicMock()

    class FakeMemoryClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.project = fake_client.project
            self._store = {}
            self._mock = fake_client

        def add(self, messages=None, user_id=None):
            return {"results": [{"id": "mem-1"}]}

        def search(self, query=None, filters=None, limit=5):
            return {"results": [{"id": "mem-1", "memory": "test claim", "score": 0.9}]}

        def delete(self, memory_id=None):
            pass

        def delete_all(self, user_id=None):
            pass

        def update(self, memory_id=None, data=None):
            pass

    fake_mem0.MemoryClient = FakeMemoryClient
    sys.modules["mem0"] = fake_mem0
    yield fake_mem0
    del sys.modules["mem0"]


# Import after mocking
def _get_adapter_class():
    from mnemebrain_benchmark.adapters.mem0_adapter import Mem0Adapter

    return Mem0Adapter


class TestMem0Adapter:
    def test_requires_api_key(self, mock_mem0, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        Mem0Adapter = _get_adapter_class()
        with pytest.raises(ValueError, match="MEM0_API_KEY"):
            Mem0Adapter()

    def test_name(self, mock_mem0, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        assert adapter.name() == "mem0"

    def test_capabilities(self, mock_mem0, monkeypatch):
        from mnemebrain_benchmark.interface import Capability

        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        caps = adapter.capabilities()
        assert Capability.STORE in caps
        assert Capability.QUERY in caps
        assert Capability.RETRACT in caps
        assert Capability.EXPLAIN in caps
        assert Capability.REVISE in caps

    @patch("time.sleep")
    def test_store(self, mock_sleep, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        result = adapter.store("test claim", [{"content": "evidence"}])
        assert result.belief_id
        assert result.truth_state == "true"

    @patch("time.sleep")
    def test_store_with_fallback_search(self, mock_sleep, mock_mem0):
        """When add() doesn't return an id, adapter searches for it."""
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        # Patch add to return empty results
        adapter._client.add = MagicMock(return_value={"results": []})
        result = adapter.store("test claim", [])
        assert result.belief_id

    def test_query(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        results = adapter.query("test claim")
        assert len(results) == 1
        assert results[0].claim == "test claim"

    def test_query_filters_low_score(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        adapter._client.search = MagicMock(
            return_value={"results": [{"id": "m1", "memory": "low", "score": 0.1}]}
        )
        results = adapter.query("test")
        assert len(results) == 0

    @patch("time.sleep")
    def test_retract_found(self, mock_sleep, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        store_result = adapter.store("test", [])
        retract_result = adapter.retract(store_result.belief_id)
        assert retract_result.affected_beliefs == 1

    def test_retract_not_found(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        result = adapter.retract("nonexistent")
        assert result.affected_beliefs == 0

    def test_explain_found(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        result = adapter.explain("test claim")
        assert result.has_evidence is True
        assert result.supporting_count == 1

    def test_explain_not_found(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        adapter._client.search = MagicMock(return_value={"results": []})
        result = adapter.explain("unknown")
        assert result.has_evidence is False

    @patch("time.sleep")
    def test_revise(self, mock_sleep, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        store_result = adapter.store("test", [])
        revise_result = adapter.revise(store_result.belief_id, [{"content": "new"}])
        assert revise_result.truth_state == "true"

    @patch("time.sleep")
    def test_revise_update_fails_falls_back_to_add(self, mock_sleep, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        store_result = adapter.store("test", [])
        adapter._client.update = MagicMock(side_effect=Exception("fail"))
        adapter._client.add = MagicMock(return_value={"results": []})
        result = adapter.revise(store_result.belief_id, [{"content": "new"}])
        assert result.truth_state == "true"
        adapter._client.add.assert_called_once()

    def test_reset(self, mock_mem0):
        Mem0Adapter = _get_adapter_class()
        adapter = Mem0Adapter(api_key="test-key")
        old_user = adapter._user_id
        adapter.reset()
        assert adapter._user_id != old_user
        assert adapter._store_to_memory == {}
