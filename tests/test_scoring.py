"""Tests for mnemebrain_benchmark.scoring -- evaluate_expectations and aggregate_by_category."""

from __future__ import annotations

from mnemebrain_benchmark.interface import (
    ConsolidateResult,
    ExplainResult,
    MemoryTierResult,
    QueryResult,
    RetractResult,
    ReviseResult,
    SandboxResult,
    StoreResult,
)
from mnemebrain_benchmark.scenarios.schema import Expectation
from mnemebrain_benchmark.scoring import (
    CheckResult,
    ScenarioScore,
    aggregate_by_category,
    evaluate_expectations,
)

# -- ScenarioScore --


class TestScenarioScore:
    def test_score_with_checks(self):
        checks = [
            CheckResult("a", True, 1, 1),
            CheckResult("b", False, 1, 0),
            CheckResult("c", True, 1, 1),
        ]
        s = ScenarioScore("s1", "cat", checks, skipped=False)
        assert abs(s.score() - 2.0 / 3.0) < 0.001

    def test_score_all_pass(self):
        checks = [CheckResult("a", True, 1, 1)]
        s = ScenarioScore("s1", "cat", checks, skipped=False)
        assert s.score() == 1.0

    def test_score_skipped(self):
        s = ScenarioScore("s1", "cat", [], skipped=True)
        assert s.score() is None

    def test_score_empty_checks(self):
        s = ScenarioScore("s1", "cat", [], skipped=False)
        assert s.score() is None


# -- evaluate_expectations: StoreResult --


class TestEvaluateExpectationsStore:
    def test_beliefs_stored(self):
        result = StoreResult("b1", False, False, "true", 0.9)
        exp = Expectation(action_label="s1", beliefs_stored=1)
        checks = evaluate_expectations([exp], {"s1": result})
        assert len(checks) == 1
        assert checks[0].passed is True

    def test_contradiction_detected(self):
        result = StoreResult("b1", False, True, "hedged", 0.5)
        exp = Expectation(action_label="s1", contradiction_detected=True)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True

    def test_contradiction_not_detected(self):
        result = StoreResult("b1", False, False, "true", 0.9)
        exp = Expectation(action_label="s1", contradiction_detected=True)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is False

    def test_truth_state_from_store(self):
        result = StoreResult("b1", False, False, "true", 0.9)
        exp = Expectation(action_label="s1", truth_state="true")
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True

    def test_confidence_above(self):
        result = StoreResult("b1", False, False, "true", 0.9)
        exp = Expectation(action_label="s1", confidence_above=0.5)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True

    def test_confidence_below(self):
        result = StoreResult("b1", False, False, "true", 0.3)
        exp = Expectation(action_label="s1", confidence_below=0.5)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True

    def test_merged(self):
        result = StoreResult("b1", True, False, "true", 0.9)
        exp = Expectation(action_label="s1", merged=True)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True

    def test_was_separated(self):
        result = StoreResult("b1", False, False, "true", 0.9, was_separated=True)
        exp = Expectation(action_label="s1", was_separated=True)
        checks = evaluate_expectations([exp], {"s1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: QueryResult (list) --


class TestEvaluateExpectationsQuery:
    def test_query_returns_claim(self):
        result = [QueryResult("b1", "test", 0.9, "true")]
        exp = Expectation(action_label="q1", query_returns_claim=True)
        checks = evaluate_expectations([exp], {"q1": result})
        assert checks[0].passed is True

    def test_query_returns_nothing(self):
        exp = Expectation(action_label="q1", query_returns_nothing=True)
        checks = evaluate_expectations([exp], {"q1": []})
        assert checks[0].passed is True

    def test_truth_state_from_query(self):
        result = [QueryResult("b1", "test", 0.9, "true")]
        exp = Expectation(action_label="q1", truth_state="true")
        checks = evaluate_expectations([exp], {"q1": result})
        assert checks[0].passed is True

    def test_confidence_from_query(self):
        result = [QueryResult("b1", "test", 0.9, "true")]
        exp = Expectation(action_label="q1", confidence_above=0.5)
        checks = evaluate_expectations([exp], {"q1": result})
        assert checks[0].passed is True

    def test_confidence_below_from_query(self):
        result = [QueryResult("b1", "test", 0.3, "true")]
        exp = Expectation(action_label="q1", confidence_below=0.5)
        checks = evaluate_expectations([exp], {"q1": result})
        assert checks[0].passed is True

    def test_confidence_below_from_query_fails(self):
        result = [QueryResult("b1", "test", 0.9, "true")]
        exp = Expectation(action_label="q1", confidence_below=0.5)
        checks = evaluate_expectations([exp], {"q1": result})
        assert checks[0].passed is False

    def test_multihop_returns_claim(self):
        result = [QueryResult("b1", "test", 0.9, "true")]
        exp = Expectation(action_label="mh1", multihop_returns_claim=True)
        checks = evaluate_expectations([exp], {"mh1": result})
        assert checks[0].passed is True

    def test_multihop_returns_nothing(self):
        exp = Expectation(action_label="mh1", multihop_returns_nothing=True)
        checks = evaluate_expectations([exp], {"mh1": []})
        assert checks[0].passed is True


# -- evaluate_expectations: ExplainResult --


class TestEvaluateExpectationsExplain:
    def test_explanation_has_evidence(self):
        result = ExplainResult("c", True, 2, 0, "true", 0.9)
        exp = Expectation(action_label="e1", explanation_has_evidence=True)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_supporting_count_gte(self):
        result = ExplainResult("c", True, 3, 0, "true", 0.9)
        exp = Expectation(action_label="e1", explanation_supporting_count_gte=2)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_attacking_count_gte(self):
        result = ExplainResult("c", True, 0, 2, "hedged", 0.5)
        exp = Expectation(action_label="e1", explanation_attacking_count_gte=1)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_expired_count_gte(self):
        result = ExplainResult("c", True, 0, 0, "true", 0.9, expired_count=3)
        exp = Expectation(action_label="e1", explanation_expired_count_gte=2)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_truth_state_from_explain(self):
        result = ExplainResult("c", True, 1, 0, "true", 0.9)
        exp = Expectation(action_label="e1", truth_state="true")
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_confidence_from_explain(self):
        result = ExplainResult("c", True, 1, 0, "true", 0.9)
        exp = Expectation(action_label="e1", confidence_above=0.5)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True

    def test_confidence_below_from_explain(self):
        result = ExplainResult("c", True, 1, 0, "true", 0.3)
        exp = Expectation(action_label="e1", confidence_below=0.5)
        checks = evaluate_expectations([exp], {"e1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: RetractResult --


class TestEvaluateExpectationsRetract:
    def test_affected_beliefs(self):
        result = RetractResult(affected_beliefs=2, truth_states_changed=1)
        exp = Expectation(action_label="r1", affected_beliefs=1)
        checks = evaluate_expectations([exp], {"r1": result})
        assert checks[0].passed is True

    def test_affected_beliefs_insufficient(self):
        result = RetractResult(affected_beliefs=0, truth_states_changed=0)
        exp = Expectation(action_label="r1", affected_beliefs=1)
        checks = evaluate_expectations([exp], {"r1": result})
        assert checks[0].passed is False


# -- evaluate_expectations: ReviseResult --


class TestEvaluateExpectationsRevise:
    def test_truth_state_from_revise(self):
        result = ReviseResult("b1", "true", 0.9, 1)
        exp = Expectation(action_label="rv1", truth_state="true")
        checks = evaluate_expectations([exp], {"rv1": result})
        assert checks[0].passed is True

    def test_revision_superseded_count(self):
        result = ReviseResult("b1", "true", 0.9, 2)
        exp = Expectation(action_label="rv1", revision_superseded_count_gte=1)
        checks = evaluate_expectations([exp], {"rv1": result})
        assert checks[0].passed is True

    def test_confidence_from_revise(self):
        result = ReviseResult("b1", "true", 0.9, 0)
        exp = Expectation(action_label="rv1", confidence_above=0.5)
        checks = evaluate_expectations([exp], {"rv1": result})
        assert checks[0].passed is True

    def test_confidence_below_from_revise(self):
        result = ReviseResult("b1", "true", 0.3, 0)
        exp = Expectation(action_label="rv1", confidence_below=0.5)
        checks = evaluate_expectations([exp], {"rv1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: SandboxResult --


class TestEvaluateExpectationsSandbox:
    def test_sandbox_resolved_state(self):
        result = SandboxResult("sb1", "false", True)
        exp = Expectation(action_label="sf1", sandbox_resolved_state="false")
        checks = evaluate_expectations([exp], {"sf1": result})
        assert checks[0].passed is True

    def test_sandbox_canonical_unchanged(self):
        result = SandboxResult("sb1", "false", True)
        exp = Expectation(action_label="sf1", sandbox_canonical_unchanged=True)
        checks = evaluate_expectations([exp], {"sf1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: ConsolidateResult --


class TestEvaluateExpectationsConsolidate:
    def test_semantic_beliefs_created_gte(self):
        result = ConsolidateResult(semantic_beliefs_created=5, episodics_pruned=3, clusters_found=2)
        exp = Expectation(action_label="c1", semantic_beliefs_created_gte=3)
        checks = evaluate_expectations([exp], {"c1": result})
        assert checks[0].passed is True

    def test_semantic_beliefs_created_exact(self):
        result = ConsolidateResult(semantic_beliefs_created=5, episodics_pruned=3, clusters_found=2)
        exp = Expectation(action_label="c1", semantic_beliefs_created=5)
        checks = evaluate_expectations([exp], {"c1": result})
        assert checks[0].passed is True

    def test_episodics_pruned_gte(self):
        result = ConsolidateResult(semantic_beliefs_created=5, episodics_pruned=3, clusters_found=2)
        exp = Expectation(action_label="c1", episodics_pruned_gte=2)
        checks = evaluate_expectations([exp], {"c1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: MemoryTierResult --


class TestEvaluateExpectationsMemoryTier:
    def test_memory_tier(self):
        result = MemoryTierResult("b1", "semantic", 3)
        exp = Expectation(action_label="mt1", memory_tier="semantic")
        checks = evaluate_expectations([exp], {"mt1": result})
        assert checks[0].passed is True

    def test_consolidated_from_count_gte(self):
        result = MemoryTierResult("b1", "semantic", 5)
        exp = Expectation(action_label="mt1", consolidated_from_count_gte=3)
        checks = evaluate_expectations([exp], {"mt1": result})
        assert checks[0].passed is True


# -- evaluate_expectations: missing action --


class TestEvaluateExpectationsMissing:
    def test_missing_action_label(self):
        exp = Expectation(action_label="nonexistent", beliefs_stored=1)
        checks = evaluate_expectations([exp], {})
        assert len(checks) == 1
        assert checks[0].passed is False  # actual=0, expected=1

    def test_confidence_above_none(self):
        """If result is None for action, confidence checks should fail."""
        exp = Expectation(action_label="x", confidence_above=0.5)
        checks = evaluate_expectations([exp], {})
        assert len(checks) == 1
        assert checks[0].passed is False


# -- aggregate_by_category --


class TestAggregateByCategory:
    def test_basic_aggregation(self):
        scores = [
            ScenarioScore("s1", "cat_a", [CheckResult("c", True, 1, 1)], False),
            ScenarioScore("s2", "cat_a", [CheckResult("c", False, 1, 0)], False),
            ScenarioScore("s3", "cat_b", [CheckResult("c", True, 1, 1)], False),
        ]
        cats = aggregate_by_category(scores)
        assert set(cats.keys()) == {"cat_a", "cat_b"}
        assert abs(cats["cat_a"].score - 0.5) < 0.01
        assert cats["cat_b"].score == 1.0

    def test_all_skipped(self):
        scores = [
            ScenarioScore("s1", "cat_a", [], True),
            ScenarioScore("s2", "cat_a", [], True),
        ]
        cats = aggregate_by_category(scores)
        assert cats["cat_a"].skipped is True
        assert cats["cat_a"].score is None

    def test_mixed_skipped(self):
        scores = [
            ScenarioScore("s1", "cat_a", [CheckResult("c", True, 1, 1)], False),
            ScenarioScore("s2", "cat_a", [], True),
        ]
        cats = aggregate_by_category(scores)
        assert cats["cat_a"].skipped is False
        assert cats["cat_a"].score == 1.0

    def test_empty_input(self):
        cats = aggregate_by_category([])
        assert cats == {}
