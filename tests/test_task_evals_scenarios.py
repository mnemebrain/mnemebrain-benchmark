"""Tests for task evaluation scenario loading."""
from __future__ import annotations

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)
from mnemebrain_benchmark.task_evals.base import TaskScenario
from mnemebrain_benchmark.task_evals.long_horizon_qa import load_qa_scenarios
from mnemebrain_benchmark.task_evals.preference_tracking import load_preference_scenarios


class TestPreferenceScenarios:
    def test_loads_scenarios(self):
        scenarios = load_preference_scenarios()
        assert len(scenarios) >= 8
        for s in scenarios:
            assert isinstance(s, TaskScenario)
            assert len(s.actions) >= 2
            assert len(s.questions) >= 3

    def test_categories(self):
        scenarios = load_preference_scenarios()
        categories = {s.category for s in scenarios}
        assert "dietary" in categories
        assert "location" in categories

    def test_has_contradictions(self):
        scenarios = load_preference_scenarios()
        has_multi_store = any(
            sum(1 for a in s.actions if a.type == "store") >= 2
            for s in scenarios
        )
        assert has_multi_store

    def test_questions_have_rejected_keywords(self):
        scenarios = load_preference_scenarios()
        has_rejected = any(
            any(q.rejected_keywords for q in s.questions)
            for s in scenarios
        )
        assert has_rejected


class TestQAScenarios:
    def test_loads_scenarios(self):
        scenarios = load_qa_scenarios()
        assert len(scenarios) >= 8
        for s in scenarios:
            assert isinstance(s, TaskScenario)
            assert len(s.actions) >= 3
            assert len(s.questions) >= 3

    def test_categories(self):
        scenarios = load_qa_scenarios()
        categories = {s.category for s in scenarios}
        assert "project_management" in categories

    def test_has_retractions(self):
        scenarios = load_qa_scenarios()
        has_retract = any(
            any(a.type == "retract" for a in s.actions)
            for s in scenarios
        )
        assert has_retract


class KeywordMemory(MemorySystem):
    def __init__(self):
        self._claims: list[str] = []

    def name(self) -> str:
        return "keyword_memory"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        self._claims.append(claim)
        return StoreResult(
            belief_id=str(len(self._claims) - 1),
            merged=False, contradiction_detected=False,
            truth_state=None, confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        words = set(claim.lower().split())
        results = []
        for i, c in enumerate(self._claims):
            if any(w in c.lower() for w in words if len(w) > 3):
                results.append(QueryResult(
                    belief_id=str(i), claim=c, confidence=None, truth_state=None,
                ))
        return results

    def reset(self) -> None:
        self._claims.clear()


class TestIntegration:
    def test_preference_scenarios_run(self):
        from mnemebrain_benchmark.task_evals.runner import TaskEvalRunner
        scenarios = load_preference_scenarios()
        runner = TaskEvalRunner()
        system = KeywordMemory()
        report = runner.run_all([system], scenarios)
        scores = report.scores["keyword_memory"]
        assert len(scores) == len(scenarios)
        total_q = sum(s.total for s in scores)
        assert total_q >= 30

    def test_qa_scenarios_run(self):
        from mnemebrain_benchmark.task_evals.runner import TaskEvalRunner
        scenarios = load_qa_scenarios()
        runner = TaskEvalRunner()
        system = KeywordMemory()
        report = runner.run_all([system], scenarios)
        scores = report.scores["keyword_memory"]
        assert len(scores) == len(scenarios)
        total_q = sum(s.total for s in scores)
        assert total_q >= 20

    def test_table_output(self):
        from mnemebrain_benchmark.task_evals.runner import TaskEvalRunner, format_task_eval_table
        scenarios = load_preference_scenarios()[:2]
        runner = TaskEvalRunner()
        system = KeywordMemory()
        report = runner.run_all([system], scenarios)
        report.eval_name = "preference_tracking"
        table = format_task_eval_table(report)
        assert "keyword_memory" in table
        assert "PREFERENCE_TRACKING" in table
