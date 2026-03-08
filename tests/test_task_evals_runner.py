"""Tests for task evaluation runner."""
from __future__ import annotations

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)
from mnemebrain_benchmark.task_evals.base import (
    TaskAction,
    TaskQuestion,
    TaskScenario,
)
from mnemebrain_benchmark.task_evals.runner import (
    TaskEvalRunner,
    ScenarioTaskScore,
    TaskEvalReport,
    format_task_eval_table,
)


class FakeMemory(MemorySystem):
    def __init__(self):
        self._claims: list[str] = []

    def name(self) -> str:
        return "fake"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        self._claims.append(claim)
        return StoreResult(
            belief_id=str(len(self._claims) - 1), merged=False,
            contradiction_detected=False, truth_state="true", confidence=0.8,
        )

    def query(self, claim: str) -> list[QueryResult]:
        results = []
        for i, c in enumerate(self._claims):
            if any(w in c.lower() for w in claim.lower().split() if len(w) > 3):
                results.append(QueryResult(
                    belief_id=str(i), claim=c, confidence=0.8, truth_state="true",
                ))
        return results

    def reset(self) -> None:
        self._claims.clear()


class TestTaskEvalRunner:
    def test_run_scenario(self):
        system = FakeMemory()
        scenario = TaskScenario(
            name="test_scenario", description="test", category="test",
            actions=[
                TaskAction(type="store", claim="User likes pizza",
                           evidence=[{"content": "said so", "polarity": "supports"}]),
            ],
            questions=[
                TaskQuestion(query="pizza", expected_keywords=["pizza"], rejected_keywords=[]),
            ],
        )
        runner = TaskEvalRunner()
        score = runner.run_scenario(system, scenario)
        assert score.scenario_name == "test_scenario"
        assert score.total == 1
        assert score.correct == 1

    def test_run_all(self):
        system = FakeMemory()
        scenarios = [
            TaskScenario(
                name="s1", description="d", category="c",
                actions=[
                    TaskAction(type="store", claim="User likes Thai",
                               evidence=[{"content": "c", "polarity": "supports"}]),
                ],
                questions=[
                    TaskQuestion(query="Thai", expected_keywords=["Thai"], rejected_keywords=[]),
                ],
            ),
        ]
        runner = TaskEvalRunner()
        report = runner.run_all([system], scenarios)
        assert "fake" in report.scores
        assert report.scores["fake"][0].correct == 1

    def test_accuracy_property(self):
        score = ScenarioTaskScore("s", "c", 3, 5, [])
        assert abs(score.accuracy - 0.6) < 0.001

    def test_accuracy_zero_total(self):
        score = ScenarioTaskScore("s", "c", 0, 0, [])
        assert score.accuracy == 0.0


class TestFormatTable:
    def test_format(self):
        report = TaskEvalReport(
            eval_name="test",
            scores={"adapter_a": [ScenarioTaskScore("s1", "c1", 3, 5, [])]},
        )
        table = format_task_eval_table(report)
        assert "adapter_a" in table
        assert "60.0%" in table
