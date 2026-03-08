"""Tests for mnemebrain_benchmark.scenarios.schema."""

from __future__ import annotations

from mnemebrain_benchmark.scenarios.schema import (
    VALID_ACTION_TYPES,
    VALID_CATEGORIES,
    Action,
    Expectation,
    Scenario,
)


class TestValidSets:
    def test_all_action_types(self):
        expected = {
            "store",
            "retract",
            "query",
            "explain",
            "wait_days",
            "revise",
            "sandbox_fork",
            "sandbox_assume",
            "sandbox_resolve",
            "sandbox_discard",
            "add_attack",
            "consolidate",
            "query_multihop",
            "get_memory_tier",
        }
        assert VALID_ACTION_TYPES == expected

    def test_all_categories(self):
        expected = {
            "contradiction",
            "retraction",
            "decay",
            "dedup",
            "extraction",
            "lifecycle",
            "belief_revision",
            "evidence_tracking",
            "temporal",
            "counterfactual",
            "consolidation",
            "multihop_retrieval",
            "pattern_separation",
        }
        assert VALID_CATEGORIES == expected


class TestAction:
    def test_defaults(self):
        a = Action(label="s1", type="store")
        assert a.claim is None
        assert a.evidence is None
        assert a.wait_days is None
        assert a.sandbox_label is None

    def test_all_fields(self):
        a = Action(
            label="s1",
            type="store",
            claim="x",
            evidence=[{"a": 1}],
            belief_type="fact",
            target_label="t",
            wait_days=5,
            belief_label="bl",
            sandbox_label="sb",
            scenario_label="sc",
            truth_state_override="false",
            belief_ref_label="br",
        )
        assert a.belief_ref_label == "br"


class TestExpectation:
    def test_defaults(self):
        e = Expectation(action_label="s1")
        assert e.beliefs_stored is None
        assert e.was_separated is None
        assert e.multihop_returns_claim is None

    def test_phase5_fields(self):
        e = Expectation(
            action_label="c1",
            semantic_beliefs_created_gte=1,
            semantic_beliefs_created=2,
            episodics_pruned_gte=1,
            memory_tier="semantic",
            consolidated_from_count_gte=3,
            was_separated=True,
            multihop_returns_claim=True,
            multihop_returns_nothing=False,
        )
        assert e.memory_tier == "semantic"


class TestScenario:
    def test_construction(self):
        s = Scenario(
            name="test",
            description="desc",
            category="contradiction",
            requires=["store", "query"],
            actions=[Action(label="s1", type="store")],
            expectations=[Expectation(action_label="s1")],
        )
        assert s.name == "test"
        assert len(s.actions) == 1
