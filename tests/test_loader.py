"""Tests for mnemebrain_benchmark.scenarios.loader."""

from __future__ import annotations

import json

import pytest

from mnemebrain_benchmark.scenarios.loader import (
    load_bmb_scenarios,
    load_scenarios,
    validate_scenario,
)
from mnemebrain_benchmark.scenarios.schema import Action, Expectation, Scenario


class TestValidateScenario:
    def test_valid_scenario(self):
        s = Scenario(
            name="test",
            description="d",
            category="contradiction",
            requires=[],
            actions=[Action(label="s1", type="store")],
            expectations=[Expectation(action_label="s1")],
        )
        validate_scenario(s)  # should not raise

    def test_invalid_action_type(self):
        s = Scenario(
            name="test",
            description="d",
            category="contradiction",
            requires=[],
            actions=[Action(label="s1", type="invalid_type")],
            expectations=[],
        )
        with pytest.raises(ValueError, match="Invalid action type"):
            validate_scenario(s)

    def test_duplicate_label(self):
        s = Scenario(
            name="test",
            description="d",
            category="contradiction",
            requires=[],
            actions=[
                Action(label="s1", type="store"),
                Action(label="s1", type="query"),
            ],
            expectations=[],
        )
        with pytest.raises(ValueError, match="Duplicate action label"):
            validate_scenario(s)

    def test_unknown_expectation_label(self):
        s = Scenario(
            name="test",
            description="d",
            category="contradiction",
            requires=[],
            actions=[Action(label="s1", type="store")],
            expectations=[Expectation(action_label="nonexistent")],
        )
        with pytest.raises(ValueError, match="unknown action"):
            validate_scenario(s)


class TestLoadScenarios:
    def test_load_from_file(self, tmp_path):
        data = [
            {
                "name": "test_scenario",
                "description": "A test",
                "category": "contradiction",
                "requires": ["store"],
                "actions": [{"label": "s1", "type": "store", "claim": "x"}],
                "expectations": [{"action_label": "s1", "beliefs_stored": 1}],
            }
        ]
        path = tmp_path / "scenarios.json"
        path.write_text(json.dumps(data))
        scenarios = load_scenarios(path)
        assert len(scenarios) == 1
        assert scenarios[0].name == "test_scenario"
        assert scenarios[0].actions[0].claim == "x"

    def test_load_bundled(self):
        """Bundled scenarios.json should load successfully."""
        scenarios = load_scenarios()
        assert len(scenarios) > 0
        for s in scenarios:
            assert s.name
            assert s.category

    def test_invalid_scenario_in_file(self, tmp_path):
        data = [
            {
                "name": "bad",
                "description": "bad",
                "category": "contradiction",
                "actions": [{"label": "s1", "type": "BAD_TYPE"}],
                "expectations": [],
            }
        ]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Invalid action type"):
            load_scenarios(path)


class TestLoadBmbScenarios:
    def test_load_bundled(self):
        scenarios = load_bmb_scenarios()
        assert len(scenarios) > 0

    def test_load_from_file(self, tmp_path):
        data = [
            {
                "name": "bmb_test",
                "description": "A BMB test",
                "category": "contradiction",
                "requires": [],
                "actions": [{"label": "s1", "type": "store", "claim": "y"}],
                "expectations": [{"action_label": "s1"}],
            }
        ]
        path = tmp_path / "bmb.json"
        path.write_text(json.dumps(data))
        scenarios = load_bmb_scenarios(path)
        assert len(scenarios) == 1
        assert scenarios[0].name == "bmb_test"
