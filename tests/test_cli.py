"""Tests for CLI modules (bmb_cli, system_cli, __main__)."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from mnemebrain_benchmark.scoring import CheckResult, ScenarioScore


# -- bmb_cli --

class TestBmbCli:
    def test_main_parses_args(self):
        from mnemebrain_benchmark.bmb_cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_all_adapters_list(self):
        from mnemebrain_benchmark.bmb_cli import ALL_ADAPTERS
        assert "mnemebrain" in ALL_ADAPTERS
        assert "langchain_buffer" in ALL_ADAPTERS
        assert len(ALL_ADAPTERS) == 7

    def test_bmb_categories_list(self):
        from mnemebrain_benchmark.bmb_cli import BMB_CATEGORIES
        assert "contradiction" in BMB_CATEGORIES
        assert "consolidation" in BMB_CATEGORIES
        assert len(BMB_CATEGORIES) == 8

    @patch("mnemebrain_benchmark.bmb_cli._build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    def test_run_bmb_with_langchain(self, mock_load, mock_adapters, tmp_path):
        from mnemebrain_benchmark.adapters.langchain_buffer import LangChainBufferBaseline
        from mnemebrain_benchmark.bmb_cli import run_bmb
        from mnemebrain_benchmark.scenarios.schema import Action, Expectation, Scenario

        mock_adapters.return_value = [LangChainBufferBaseline()]
        mock_load.return_value = [
            Scenario(
                name="test_scen", description="d", category="contradiction",
                requires=["store", "query"],
                actions=[
                    Action(label="s1", type="store", claim="test", evidence=[{}]),
                    Action(label="q1", type="query", claim="test"),
                ],
                expectations=[Expectation(action_label="s1", beliefs_stored=1)],
            )
        ]

        output = str(tmp_path / "report.json")
        results = run_bmb(adapter_filter="langchain_buffer", output=output)
        assert "langchain_buffer" in results
        assert (tmp_path / "report.json").exists()

    def test_print_bmb_chart(self, capsys):
        from mnemebrain_benchmark.bmb_cli import _print_bmb_chart

        results = {
            "sys": [ScenarioScore("s1", "cat", [CheckResult("c", True, 1, 1)], False)]
        }
        _print_bmb_chart(results)
        captured = capsys.readouterr()
        assert "BELIEF MAINTENANCE BENCHMARK" in captured.out

    def test_print_bmb_chart_skipped(self, capsys):
        from mnemebrain_benchmark.bmb_cli import _print_bmb_chart
        results = {"sys": [ScenarioScore("s1", "cat", [], True)]}
        _print_bmb_chart(results)
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_build_adapters_langchain(self):
        from mnemebrain_benchmark.bmb_cli import _build_adapters
        adapters = _build_adapters("langchain_buffer")
        assert len(adapters) == 1
        assert adapters[0].name() == "langchain_buffer"


# -- system_cli --

class TestSystemCli:
    def test_main_parses_args(self):
        from mnemebrain_benchmark.system_cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# -- runner.py --

class TestRunnerModule:
    def test_create_provider_unknown(self):
        from mnemebrain_benchmark.runner import _create_provider
        with pytest.raises(ValueError, match="Unknown provider type"):
            _create_provider("unknown_type", "model")

    def test_run_benchmark_no_matching_providers(self, capsys):
        from mnemebrain_benchmark.runner import run_benchmark
        result = run_benchmark(provider_filter="nonexistent")
        assert result == {"providers": {}}

    def test_main_help(self):
        from mnemebrain_benchmark.runner import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
