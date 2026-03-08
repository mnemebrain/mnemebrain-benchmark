"""Tests for mnemebrain_benchmark.system_report."""

from __future__ import annotations

import json

from mnemebrain_benchmark.scoring import CheckResult, ScenarioScore
from mnemebrain_benchmark.system_report import export_json, format_scorecard


def _make_results():
    return {
        "system_a": [
            ScenarioScore("s1", "cat_a", [CheckResult("c1", True, 1, 1)], False),
            ScenarioScore("s2", "cat_b", [CheckResult("c2", False, 1, 0)], False),
        ],
        "system_b": [
            ScenarioScore("s1", "cat_a", [CheckResult("c1", True, 1, 1)], False),
            ScenarioScore("s3", "cat_a", [], True),
        ],
    }


class TestFormatScorecard:
    def test_basic_format(self):
        results = _make_results()
        output = format_scorecard(results)
        assert "MNEMEBRAIN SYSTEM BENCHMARK" in output
        assert "system_a" in output
        assert "system_b" in output
        assert "cat_a" in output
        assert "Overall" in output

    def test_includes_percentages(self):
        results = {"sys": [ScenarioScore("s1", "cat", [CheckResult("c", True, 1, 1)], False)]}
        output = format_scorecard(results)
        assert "100.0%" in output

    def test_skipped_shows_na(self):
        results = {"sys": [ScenarioScore("s1", "cat", [], True)]}
        output = format_scorecard(results)
        assert "N/A" in output

    def test_empty_results_returns_message(self):
        """Empty results returns a graceful no-results message."""
        output = format_scorecard({})
        assert output == "No results"


class TestExportJson:
    def test_export(self, tmp_path):
        results = _make_results()
        path = str(tmp_path / "report.json")
        export_json(results, path)

        with open(path) as f:
            data = json.load(f)

        assert "system_a" in data
        assert "system_b" in data
        assert len(data["system_a"]) == 2
        assert data["system_a"][0]["scenario"] == "s1"
        assert "checks" in data["system_a"][0]

    def test_export_skipped_scenario(self, tmp_path):
        results = {"sys": [ScenarioScore("s1", "cat", [], True)]}
        path = str(tmp_path / "report.json")
        export_json(results, path)

        with open(path) as f:
            data = json.load(f)
        assert data["sys"][0]["skipped"] is True
        assert data["sys"][0]["score"] is None
