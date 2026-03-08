"""Long-horizon QA task evaluation scenarios."""

from __future__ import annotations

from pathlib import Path

from mnemebrain_benchmark.task_evals.base import TaskScenario, _load_task_scenarios

_DATA_PATH = Path(__file__).parent / "data" / "qa_scenarios.json"


def load_qa_scenarios(path: Path | None = None) -> list[TaskScenario]:
    """Load long-horizon QA scenarios from JSON."""
    return _load_task_scenarios(path or _DATA_PATH)
