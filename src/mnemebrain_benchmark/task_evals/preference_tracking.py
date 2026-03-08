"""Preference tracking task evaluation scenarios."""
from __future__ import annotations

import json
from pathlib import Path

from mnemebrain_benchmark.task_evals.base import (
    TaskAction,
    TaskQuestion,
    TaskScenario,
)

_DATA_PATH = Path(__file__).parent / "data" / "preference_scenarios.json"


def load_preference_scenarios(path: Path | None = None) -> list[TaskScenario]:
    """Load preference tracking scenarios from JSON."""
    data_path = path or _DATA_PATH
    with open(data_path) as f:
        raw = json.load(f)

    scenarios: list[TaskScenario] = []
    for entry in raw:
        actions = [
            TaskAction(
                type=a["type"],
                claim=a.get("claim"),
                evidence=a.get("evidence"),
                target_index=a.get("target_index"),
                wait_days=a.get("wait_days"),
            )
            for a in entry["actions"]
        ]
        questions = [
            TaskQuestion(
                query=q["query"],
                expected_keywords=q["expected_keywords"],
                rejected_keywords=q.get("rejected_keywords", []),
            )
            for q in entry["questions"]
        ]
        scenarios.append(TaskScenario(
            name=entry["name"],
            description=entry["description"],
            category=entry["category"],
            actions=actions,
            questions=questions,
        ))
    return scenarios
