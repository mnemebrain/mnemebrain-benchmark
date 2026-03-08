"""Tests for mnemebrain_benchmark.adapters.mnemebrain_adapter (mocked SDK)."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from mnemebrain_benchmark.interface import Capability


@pytest.fixture(autouse=True)
def mock_mnemebrain_sdk():
    """Inject a fake mnemebrain module so we can import the adapter without the real SDK."""
    fake_mod = ModuleType("mnemebrain")

    class FakeEvidenceInput:
        def __init__(self, source_ref="", content="", polarity="supports", weight=0.7, reliability=0.8):
            self.source_ref = source_ref
            self.content = content
            self.polarity = polarity
            self.weight = weight
            self.reliability = reliability

    class FakeClient:
        def __init__(self, base_url="", timeout=30.0):
            self.base_url = base_url

        def believe(self, claim="", evidence=None, belief_type="inference"):
            r = MagicMock()
            r.id = "b-1#1"
            r.conflict = False
            r.truth_state = "true"
            r.confidence = 0.9
            return r

        def search(self, query=""):
            r = MagicMock()
            item = MagicMock()
            item.belief_id = "b-1#1"
            item.claim = query
            item.confidence = 0.9
            item.truth_state = "true"
            r.results = [item]
            return r

        def retract(self, evidence_id=""):
            r = MagicMock()
            r.id = "b-1#1"
            r.truth_state = "false"
            r.confidence = 0.0
            return [r]

        def explain(self, claim=""):
            r = MagicMock()
            r.claim = claim
            r.supporting = [MagicMock()]
            r.attacking = []
            r.expired = []
            r.truth_state = "true"
            r.confidence = 0.9
            return r

        def revise(self, belief_id="", evidence=None):
            r = MagicMock()
            r.id = belief_id
            r.truth_state = "true"
            r.confidence = 0.95
            return r

        def set_time_offset(self, days=0):
            pass

        def consolidate(self):
            r = MagicMock()
            r.semantic_beliefs_created = 2
            r.episodics_pruned = 1
            r.clusters_found = 1
            return r

        def get_memory_tier(self, belief_id=""):
            r = MagicMock()
            r.belief_id = belief_id
            r.memory_tier = "semantic"
            r.consolidated_from_count = 3
            return r

        def query_multihop(self, query=""):
            r = MagicMock()
            item = MagicMock()
            item.belief_id = "b-1#1"
            item.claim = "multihop"
            item.confidence = 0.8
            item.truth_state = "true"
            r.results = [item]
            return r

        def benchmark_sandbox_fork(self, label=""):
            r = MagicMock()
            r.sandbox_id = "sb-1"
            r.resolved_truth_state = "true"
            r.canonical_unchanged = True
            return r

        def benchmark_sandbox_assume(self, sid, bid, ts):
            r = MagicMock()
            r.sandbox_id = sid
            r.resolved_truth_state = ts
            r.canonical_unchanged = True
            return r

        def benchmark_sandbox_resolve(self, sid, bid):
            r = MagicMock()
            r.sandbox_id = sid
            r.resolved_truth_state = "false"
            r.canonical_unchanged = True
            return r

        def benchmark_sandbox_discard(self, sid):
            pass

        def benchmark_attack(self, attacker_id, target_id, attack_type, weight):
            r = MagicMock()
            r.edge_id = "e-1"
            r.attacker_id = attacker_id
            r.target_id = target_id
            return r

        def reset(self):
            pass

    fake_mod.MnemeBrainClient = FakeClient
    fake_mod.EvidenceInput = FakeEvidenceInput
    sys.modules["mnemebrain"] = fake_mod
    yield
    del sys.modules["mnemebrain"]


def _get_adapter():
    from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter
    return MnemeBrainAdapter(base_url="http://test:8000")


class TestMnemeBrainAdapter:
    def test_name(self):
        adapter = _get_adapter()
        assert adapter.name() == "mnemebrain"

    def test_capabilities_full(self):
        adapter = _get_adapter()
        caps = adapter.capabilities()
        assert Capability.STORE in caps
        assert Capability.CONSOLIDATION in caps
        assert Capability.HIPPORAG in caps
        assert Capability.PATTERN_SEPARATION in caps
        assert len(caps) == len(Capability)

    def test_store(self):
        adapter = _get_adapter()
        result = adapter.store("test", [{"source_ref": "src", "content": "ev", "polarity": "supports"}])
        assert result.belief_id == "b-1#1"
        assert result.truth_state == "true"

    def test_store_empty_evidence(self):
        adapter = _get_adapter()
        result = adapter.store("test", [])
        assert result.belief_id == "b-1#1"

    def test_query(self):
        adapter = _get_adapter()
        results = adapter.query("test")
        assert len(results) == 1
        assert results[0].claim == "test"

    def test_retract(self):
        adapter = _get_adapter()
        result = adapter.retract("ev-1")
        assert result.affected_beliefs == 1

    def test_explain(self):
        adapter = _get_adapter()
        result = adapter.explain("test")
        assert result.has_evidence is True
        assert result.supporting_count == 1
        assert result.attacking_count == 0

    def test_explain_none(self):
        adapter = _get_adapter()
        adapter._client.explain = MagicMock(return_value=None)
        result = adapter.explain("unknown")
        assert result.has_evidence is False
        assert result.truth_state == "neither"

    def test_revise(self):
        adapter = _get_adapter()
        result = adapter.revise("b-1#1", [{"content": "new evidence"}])
        assert result.belief_id == "b-1#1"
        assert result.truth_state == "true"

    def test_set_time_offset_days(self):
        adapter = _get_adapter()
        adapter.set_time_offset_days(30)  # should not raise

    def test_consolidate(self):
        adapter = _get_adapter()
        result = adapter.consolidate()
        assert result.semantic_beliefs_created == 2
        assert result.episodics_pruned == 1

    def test_get_memory_tier(self):
        adapter = _get_adapter()
        result = adapter.get_memory_tier("b-1#1")
        assert result.memory_tier == "semantic"
        assert result.consolidated_from_count == 3

    def test_query_multihop(self):
        adapter = _get_adapter()
        results = adapter.query_multihop("multi query")
        assert len(results) == 1
        assert results[0].claim == "multihop"

    def test_sandbox_fork(self):
        adapter = _get_adapter()
        result = adapter.sandbox_fork("test_scenario")
        assert result.sandbox_id == "sb-1"

    def test_sandbox_assume(self):
        adapter = _get_adapter()
        result = adapter.sandbox_assume("sb-1", "b-1#1", "false")
        assert result.resolved_truth_state == "false"

    def test_sandbox_resolve(self):
        adapter = _get_adapter()
        result = adapter.sandbox_resolve("sb-1", "b-1#1")
        assert result.resolved_truth_state == "false"

    def test_sandbox_discard(self):
        adapter = _get_adapter()
        adapter.sandbox_discard("sb-1")  # should not raise

    def test_add_attack(self):
        adapter = _get_adapter()
        result = adapter.add_attack("a-1", "t-1", "undermining", 0.5)
        assert result.edge_id == "e-1"
        assert result.attacker_id == "a-1"

    def test_reset(self):
        adapter = _get_adapter()
        adapter.reset()  # should not raise
