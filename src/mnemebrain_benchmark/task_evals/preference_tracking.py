"""Preference tracking task evaluation scenarios."""

from __future__ import annotations

from pathlib import Path

from mnemebrain_benchmark.task_evals.base import TaskScenario, _load_task_scenarios

_DATA_PATH = Path(__file__).parent / "data" / "preference_scenarios.json"


def load_preference_scenarios(path: Path | None = None) -> list[TaskScenario]:
    """Load preference tracking scenarios from JSON."""
    return _load_task_scenarios(path or _DATA_PATH)
