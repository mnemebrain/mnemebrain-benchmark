"""Tests for mnemebrain_benchmark.task_evals.base."""
from __future__ import annotations

from mnemebrain_benchmark.interface import QueryResult
from mnemebrain_benchmark.task_evals.base import (
    TaskAction,
    TaskQuestion,
    TaskResult,
    TaskScenario,
    score_question,
)


class TestTaskAction:
    def test_defaults(self):
        a = TaskAction(type="store")
        assert a.claim is None
        assert a.evidence is None
        assert a.target_index is None
        assert a.wait_days is None

    def test_all_fields(self):
        a = TaskAction(type="store", claim="x", evidence=[{"a": 1}], target_index=0, wait_days=5)
        assert a.type == "store"
        assert a.wait_days == 5


class TestTaskQuestion:
    def test_defaults(self):
        q = TaskQuestion(query="test", expected_keywords=["kw"])
        assert q.rejected_keywords == []

    def test_all_fields(self):
        q = TaskQuestion(query="test", expected_keywords=["a"], rejected_keywords=["b"])
        assert q.rejected_keywords == ["b"]


class TestTaskResult:
    def test_fields(self):
        q = TaskQuestion(query="test", expected_keywords=["a"])
        r = TaskResult(question=q, correct=True, returned_claims=["claim a"])
        assert r.correct is True


class TestTaskScenario:
    def test_construction(self):
        s = TaskScenario(
            name="s1", description="d", category="cat",
            actions=[TaskAction(type="store")],
            questions=[TaskQuestion(query="q", expected_keywords=["k"])],
        )
        assert s.name == "s1"


class TestScoreQuestion:
    def test_correct_match(self):
        q = TaskQuestion(query="test", expected_keywords=["blue"])
        results = [
            QueryResult("b1", "The sky is blue", 0.9, "true"),
        ]
        r = score_question(q, results)
        assert r.correct is True
        assert r.returned_claims == ["The sky is blue"]

    def test_no_results(self):
        q = TaskQuestion(query="test", expected_keywords=["blue"])
        r = score_question(q, [])
        assert r.correct is False
        assert r.returned_claims == []

    def test_filters_false_truth_state(self):
        q = TaskQuestion(query="test", expected_keywords=["blue"])
        results = [
            QueryResult("b1", "The sky is blue", 0.9, "false"),
        ]
        r = score_question(q, results)
        assert r.correct is False
        assert r.returned_claims == []

    def test_keeps_non_false_states(self):
        q = TaskQuestion(query="test", expected_keywords=["blue"])
        results = [
            QueryResult("b1", "The sky is blue", 0.9, "hedged"),
            QueryResult("b2", "Water is blue", 0.8, "true"),
        ]
        r = score_question(q, results)
        assert r.correct is True
        assert len(r.returned_claims) == 2

    def test_rejected_keyword_in_top_claim(self):
        q = TaskQuestion(query="test", expected_keywords=["sky"], rejected_keywords=["red"])
        results = [
            QueryResult("b1", "The sky is red", 0.9, "true"),
        ]
        r = score_question(q, results)
        assert r.correct is False

    def test_rejected_keyword_not_in_top_claim(self):
        q = TaskQuestion(query="test", expected_keywords=["sky"], rejected_keywords=["red"])
        results = [
            QueryResult("b1", "The sky is blue", 0.9, "true"),
            QueryResult("b2", "Something red", 0.5, "true"),
        ]
        r = score_question(q, results)
        # rejected only checks top claim, and "red" is not in "The sky is blue"
        assert r.correct is True

    def test_expected_keyword_case_insensitive(self):
        q = TaskQuestion(query="test", expected_keywords=["BLUE"])
        results = [QueryResult("b1", "The sky is blue", 0.9, "true")]
        r = score_question(q, results)
        assert r.correct is True

    def test_missing_expected_keyword(self):
        q = TaskQuestion(query="test", expected_keywords=["green"])
        results = [QueryResult("b1", "The sky is blue", 0.9, "true")]
        r = score_question(q, results)
        assert r.correct is False
