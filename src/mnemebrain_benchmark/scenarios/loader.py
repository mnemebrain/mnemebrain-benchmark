"""Load and validate scenarios from JSON."""
from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

from mnemebrain_benchmark.scenarios.schema import (
    Action,
    Expectation,
    Scenario,
    VALID_ACTION_TYPES,
)


def validate_scenario(scenario: Scenario) -> None:
    """Validate a scenario's internal consistency."""
    labels: set[str] = set()
    for action in scenario.actions:
        if action.type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action type '{action.type}' in scenario '{scenario.name}'. "
                f"Valid types: {VALID_ACTION_TYPES}"
            )
        if action.label in labels:
            raise ValueError(
                f"Duplicate action label '{action.label}' in scenario '{scenario.name}'"
            )
        labels.add(action.label)

    for exp in scenario.expectations:
        if exp.action_label not in labels:
            raise ValueError(
                f"Expectation references unknown action '{exp.action_label}' "
                f"in scenario '{scenario.name}'"
            )


def load_scenarios(path: Path | str | None = None) -> list[Scenario]:
    """Load and validate scenarios from a JSON file.

    If path is None, loads the bundled scenarios.json from package data.
    """
    if path is not None:
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
    else:
        ref = importlib.resources.files("mnemebrain_benchmark.scenarios") / "data" / "scenarios.json"
        raw = json.loads(ref.read_text(encoding="utf-8"))

    scenarios: list[Scenario] = []
    for entry in raw:
        actions = [Action(**{k: v for k, v in a.items()}) for a in entry.get("actions", [])]
        expectations = [Expectation(**{k: v for k, v in e.items()}) for e in entry.get("expectations", [])]
        scenario = Scenario(
            name=entry["name"],
            description=entry["description"],
            category=entry["category"],
            requires=entry.get("requires", []),
            actions=actions,
            expectations=expectations,
        )
        validate_scenario(scenario)
        scenarios.append(scenario)

    return scenarios


def load_bmb_scenarios(path: Path | str | None = None) -> list[Scenario]:
    """Load BMB scenarios. If path is None, loads from package data."""
    if path is not None:
        return load_scenarios(path)

    ref = importlib.resources.files("mnemebrain_benchmark.scenarios") / "data" / "bmb_scenarios.json"
    raw = json.loads(ref.read_text(encoding="utf-8"))

    scenarios: list[Scenario] = []
    for entry in raw:
        actions = [Action(**{k: v for k, v in a.items()}) for a in entry.get("actions", [])]
        expectations = [Expectation(**{k: v for k, v in e.items()}) for e in entry.get("expectations", [])]
        scenario = Scenario(
            name=entry["name"],
            description=entry["description"],
            category=entry["category"],
            requires=entry.get("requires", []),
            actions=actions,
            expectations=expectations,
        )
        validate_scenario(scenario)
        scenarios.append(scenario)

    return scenarios
