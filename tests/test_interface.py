"""Tests for mnemebrain_benchmark.interface -- dataclasses, enums, and ABC."""
from __future__ import annotations

import pytest

from mnemebrain_benchmark.interface import (
    AttackResult,
    Capability,
    ConsolidateResult,
    ExplainResult,
    MemorySystem,
    MemoryTierResult,
    QueryResult,
    RetractResult,
    ReviseResult,
    SandboxResult,
    StoreResult,
)


# -- Capability enum --

class TestCapability:
    def test_all_values(self):
        expected = {
            "store", "query", "retract", "explain", "contradiction",
            "decay", "revise", "sandbox", "attack", "consolidation",
            "hipporag", "pattern_separation",
        }
        assert {c.value for c in Capability} == expected

    def test_string_comparison(self):
        assert Capability.STORE == "store"
        assert Capability.QUERY == "query"

    def test_membership(self):
        caps = {Capability.STORE, Capability.QUERY}
        assert Capability.STORE in caps
        assert Capability.RETRACT not in caps


# -- Dataclasses --

class TestStoreResult:
    def test_defaults(self):
        r = StoreResult(belief_id="b1", merged=False, contradiction_detected=False, truth_state="true", confidence=0.9)
        assert r.was_separated is False
        assert r.memory_tier == "episodic"

    def test_all_fields(self):
        r = StoreResult(
            belief_id="b1", merged=True, contradiction_detected=True,
            truth_state="hedged", confidence=0.5, was_separated=True, memory_tier="semantic",
        )
        assert r.belief_id == "b1"
        assert r.merged is True
        assert r.contradiction_detected is True
        assert r.was_separated is True
        assert r.memory_tier == "semantic"


class TestQueryResult:
    def test_fields(self):
        r = QueryResult(belief_id="b1", claim="test", confidence=0.8, truth_state="true")
        assert r.claim == "test"
        assert r.confidence == 0.8


class TestRetractResult:
    def test_fields(self):
        r = RetractResult(affected_beliefs=2, truth_states_changed=1)
        assert r.affected_beliefs == 2


class TestExplainResult:
    def test_defaults(self):
        r = ExplainResult(claim="x", has_evidence=True, supporting_count=1, attacking_count=0, truth_state="true", confidence=0.9)
        assert r.expired_count == 0


class TestReviseResult:
    def test_fields(self):
        r = ReviseResult(belief_id="b1", truth_state="true", confidence=0.9, superseded_count=1)
        assert r.superseded_count == 1


class TestSandboxResult:
    def test_fields(self):
        r = SandboxResult(sandbox_id="sb1", resolved_truth_state="true", canonical_unchanged=True)
        assert r.sandbox_id == "sb1"


class TestAttackResult:
    def test_fields(self):
        r = AttackResult(edge_id="e1", attacker_id="a1", target_id="t1")
        assert r.edge_id == "e1"


class TestConsolidateResult:
    def test_fields(self):
        r = ConsolidateResult(semantic_beliefs_created=3, episodics_pruned=2, clusters_found=1)
        assert r.clusters_found == 1


class TestMemoryTierResult:
    def test_fields(self):
        r = MemoryTierResult(belief_id="b1", memory_tier="semantic", consolidated_from_count=5)
        assert r.memory_tier == "semantic"


# -- MemorySystem ABC --

class TestMemorySystemABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MemorySystem()  # type: ignore[abstract]

    def test_optional_methods_raise(self):
        """All optional methods should raise NotImplementedError by default."""

        class MinimalSystem(MemorySystem):
            def name(self): return "minimal"
            def capabilities(self): return set()
            def store(self, claim, evidence): ...
            def query(self, claim): return []
            def reset(self): ...

        sys = MinimalSystem()
        with pytest.raises(NotImplementedError):
            sys.retract("x")
        with pytest.raises(NotImplementedError):
            sys.explain("x")
        with pytest.raises(NotImplementedError):
            sys.set_time_offset_days(1)
        with pytest.raises(NotImplementedError):
            sys.consolidate()
        with pytest.raises(NotImplementedError):
            sys.get_memory_tier("x")
        with pytest.raises(NotImplementedError):
            sys.query_multihop("x")
        with pytest.raises(NotImplementedError):
            sys.revise("x", [])
        with pytest.raises(NotImplementedError):
            sys.sandbox_fork()
        with pytest.raises(NotImplementedError):
            sys.sandbox_assume("s", "b", "true")
        with pytest.raises(NotImplementedError):
            sys.sandbox_resolve("s", "b")
        with pytest.raises(NotImplementedError):
            sys.sandbox_discard("s")
        with pytest.raises(NotImplementedError):
            sys.add_attack("a", "t", "undermining", 0.5)
