"""Runner for task-level evaluations."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.task_evals.base import (
    TaskResult,
    TaskScenario,
    score_question,
)


@dataclass
class ScenarioTaskScore:
    scenario_name: str
    category: str
    correct: int
    total: int
    details: list[TaskResult]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class TaskEvalReport:
    eval_name: str
    scores: dict[str, list[ScenarioTaskScore]]


class TaskEvalRunner:
    def run_scenario(self, system: MemorySystem, scenario: TaskScenario) -> ScenarioTaskScore:
        system.reset()
        store_results: list[object] = []

        for action in scenario.actions:
            if action.type == "store":
                result = system.store(
                    claim=action.claim or "",
                    evidence=action.evidence or [],
                )
                store_results.append(result)
            elif action.type == "retract":
                if action.target_index is not None and action.target_index < len(store_results):
                    target = store_results[action.target_index]
                    if hasattr(target, "belief_id"):
                        system.retract(target.belief_id)
            elif action.type == "revise":
                if action.target_index is not None and action.target_index < len(store_results):
                    target = store_results[action.target_index]
                    if hasattr(target, "belief_id"):
                        result = system.revise(
                            belief_id=target.belief_id,
                            evidence=action.evidence or [],
                        )
                        store_results.append(result)
            elif action.type == "wait_days":
                if action.wait_days:
                    try:
                        system.set_time_offset_days(action.wait_days)
                    except NotImplementedError:
                        pass

        details: list[TaskResult] = []
        for question in scenario.questions:
            query_results = system.query(question.query)
            task_result = score_question(question, query_results)
            details.append(task_result)

        correct = sum(1 for d in details if d.correct)
        return ScenarioTaskScore(
            scenario_name=scenario.name,
            category=scenario.category,
            correct=correct,
            total=len(details),
            details=details,
        )

    def run_all(
        self,
        systems: list[MemorySystem],
        scenarios: list[TaskScenario],
    ) -> TaskEvalReport:
        scores: dict[str, list[ScenarioTaskScore]] = {}
        for system in systems:
            system_scores: list[ScenarioTaskScore] = []
            for scenario in scenarios:
                try:
                    system_scores.append(self.run_scenario(system, scenario))
                except Exception as exc:
                    print(
                        f"  warning: {system.name()}/{scenario.name}: {exc}",
                        file=sys.stderr,
                    )
                    system_scores.append(
                        ScenarioTaskScore(
                            scenario_name=scenario.name,
                            category=scenario.category,
                            correct=0,
                            total=len(scenario.questions),
                            details=[],
                        )
                    )
            scores[system.name()] = system_scores
        return TaskEvalReport(eval_name="task_eval", scores=scores)


def format_task_eval_table(report: TaskEvalReport) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"TASK EVALUATION: {report.eval_name.upper()}")

    name_width = max(20, *(len(n) for n in report.scores))
    lines.append("=" * (name_width + 30))
    lines.append(f"{'Adapter':<{name_width}} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("-" * (name_width + 30))

    for adapter_name, scenario_scores in report.scores.items():
        total_correct = sum(s.correct for s in scenario_scores)
        total_questions = sum(s.total for s in scenario_scores)
        accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        lines.append(
            f"{adapter_name:<{name_width}} {total_correct:>8}"
            f" {total_questions:>8} {accuracy * 100:>9.1f}%"
        )

    lines.append("=" * (name_width + 30))
    lines.append("")
    return "\n".join(lines)
