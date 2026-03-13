"""Tests for bmb_cli external benchmark integration and external_evals CLI."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


class _FakeAdapter(MemorySystem):
    """Minimal adapter for testing external benchmark flows."""

    def __init__(self, adapter_name: str = "fake") -> None:
        self._name = adapter_name

    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        return StoreResult("b1", False, False, "true", 0.9)

    def query(self, claim: str) -> list[QueryResult]:
        return [QueryResult("b1", claim, 0.9, "true")]

    def reset(self) -> None:
        pass


def _make_lme_dataset(tmp_path):
    """Create a minimal LongMemEval dataset."""
    data = [
        {
            "id": "lme_001",
            "category": "knowledge_update",
            "sessions": [
                {"role": "user", "content": "Alice works at Google."},
                {"role": "assistant", "content": "Got it."},
            ],
            "questions": [
                {"question": "Where does Alice work?", "answer": "Google"}
            ],
        }
    ]
    fp = tmp_path / "lme.json"
    fp.write_text(json.dumps(data))
    return str(fp)


def _make_hotpotqa_dataset(tmp_path):
    """Create a minimal HotpotQA dataset."""
    data = [
        {
            "_id": "hpqa_001",
            "question": "Were Scott and Ed of the same nationality?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {"title": ["Scott", "Ed"], "sent_id": [0, 1]},
            "context": {
                "title": ["Scott", "Ed"],
                "sentences": [
                    ["Scott is American."],
                    ["Ed is also American."],
                ],
            },
        }
    ]
    fp = tmp_path / "hpqa.json"
    fp.write_text(json.dumps(data))
    return str(fp)


# ===========================================================================
# bmb_cli: _run_longmemeval_for_adapter
# ===========================================================================


class TestRunLongmemevalForAdapter:
    def test_basic_run(self, tmp_path):
        from mnemebrain_benchmark.bmb_cli import _run_longmemeval_for_adapter

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()
        report = _run_longmemeval_for_adapter(
            adapter, data_path=data_path, limit=None, verbose=False,
        )
        assert report.benchmark_name == "longmemeval"
        assert report.system_name == "fake"
        assert report.total_questions == 1

    def test_with_limit(self, tmp_path):
        from mnemebrain_benchmark.bmb_cli import _run_longmemeval_for_adapter

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()
        report = _run_longmemeval_for_adapter(
            adapter, data_path=data_path, limit=1, verbose=False,
        )
        assert report.total_questions == 1

    def test_verbose(self, tmp_path, capsys):
        from mnemebrain_benchmark.bmb_cli import _run_longmemeval_for_adapter

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()
        _run_longmemeval_for_adapter(
            adapter, data_path=data_path, limit=None, verbose=True,
        )
        captured = capsys.readouterr()
        assert "lme_001" in captured.err


# ===========================================================================
# bmb_cli: _run_hotpotqa_for_adapter
# ===========================================================================


class TestRunHotpotqaForAdapter:
    def test_basic_run(self, tmp_path):
        from mnemebrain_benchmark.bmb_cli import _run_hotpotqa_for_adapter

        data_path = _make_hotpotqa_dataset(tmp_path)
        adapter = _FakeAdapter()
        report = _run_hotpotqa_for_adapter(
            adapter, data_path=data_path, limit=None, verbose=False,
        )
        assert report.benchmark_name == "hotpotqa"
        assert report.system_name == "fake"
        assert report.total_questions >= 1

    def test_with_limit(self, tmp_path):
        from mnemebrain_benchmark.bmb_cli import _run_hotpotqa_for_adapter

        data_path = _make_hotpotqa_dataset(tmp_path)
        adapter = _FakeAdapter()
        report = _run_hotpotqa_for_adapter(
            adapter, data_path=data_path, limit=1, verbose=False,
        )
        assert report.total_questions >= 1

    def test_verbose(self, tmp_path, capsys):
        from mnemebrain_benchmark.bmb_cli import _run_hotpotqa_for_adapter

        data_path = _make_hotpotqa_dataset(tmp_path)
        adapter = _FakeAdapter()
        _run_hotpotqa_for_adapter(
            adapter, data_path=data_path, limit=None, verbose=True,
        )
        captured = capsys.readouterr()
        assert "hpqa_001" in captured.err


# ===========================================================================
# bmb_cli: _run_external_benchmarks
# ===========================================================================


class TestRunExternalBenchmarks:
    def test_all_benchmarks(self):
        from mnemebrain_benchmark.bmb_cli import _run_external_benchmarks

        adapter = _FakeAdapter()
        with patch("mnemebrain_benchmark.bmb_cli._run_longmemeval_for_adapter") as mock_lme, \
             patch("mnemebrain_benchmark.bmb_cli._run_hotpotqa_for_adapter") as mock_hpqa:
            mock_report = MagicMock()
            mock_lme.return_value = mock_report
            mock_hpqa.return_value = mock_report

            reports = _run_external_benchmarks(
                [adapter],
                data_path="/fake/path",
                external_benchmark="all",
            )
            assert len(reports) == 2
            mock_lme.assert_called_once()
            mock_hpqa.assert_called_once()

    def test_longmemeval_only(self):
        from mnemebrain_benchmark.bmb_cli import _run_external_benchmarks

        adapter = _FakeAdapter()
        with patch("mnemebrain_benchmark.bmb_cli._run_longmemeval_for_adapter") as mock_lme, \
             patch("mnemebrain_benchmark.bmb_cli._run_hotpotqa_for_adapter") as mock_hpqa:
            mock_lme.return_value = MagicMock()

            reports = _run_external_benchmarks(
                [adapter],
                data_path="/fake/path",
                external_benchmark="longmemeval",
            )
            assert len(reports) == 1
            mock_lme.assert_called_once()
            mock_hpqa.assert_not_called()

    def test_hotpotqa_only(self):
        from mnemebrain_benchmark.bmb_cli import _run_external_benchmarks

        adapter = _FakeAdapter()
        with patch("mnemebrain_benchmark.bmb_cli._run_longmemeval_for_adapter") as mock_lme, \
             patch("mnemebrain_benchmark.bmb_cli._run_hotpotqa_for_adapter") as mock_hpqa:
            mock_hpqa.return_value = MagicMock()

            reports = _run_external_benchmarks(
                [adapter],
                data_path="/fake/path",
                external_benchmark="hotpotqa",
            )
            assert len(reports) == 1
            mock_lme.assert_not_called()
            mock_hpqa.assert_called_once()

    def test_passes_limit_and_verbose(self):
        from mnemebrain_benchmark.bmb_cli import _run_external_benchmarks

        adapter = _FakeAdapter()
        with patch("mnemebrain_benchmark.bmb_cli._run_longmemeval_for_adapter") as mock_lme, \
             patch("mnemebrain_benchmark.bmb_cli._run_hotpotqa_for_adapter") as mock_hpqa:
            mock_lme.return_value = MagicMock()
            mock_hpqa.return_value = MagicMock()

            _run_external_benchmarks(
                [adapter],
                data_path="/fake/path",
                external_benchmark="all",
                external_limit=5,
                verbose=True,
            )
            _, kwargs = mock_lme.call_args
            assert kwargs["limit"] == 5
            assert kwargs["verbose"] is True


# ===========================================================================
# bmb_cli: run_bmb with external flags
# ===========================================================================


class TestRunBmbExternal:
    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli._run_external_benchmarks")
    def test_external_only(self, mock_ext, mock_adapters, tmp_path):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_adapters.return_value = [_FakeAdapter()]
        mock_report = MagicMock()
        mock_report.summary.return_value = "REPORT"
        mock_ext.return_value = [mock_report]

        output = str(tmp_path / "report.json")
        results = run_bmb(
            external_only=True,
            data_path="/fake/data.json",
            output=output,
        )
        assert results == {}
        mock_ext.assert_called_once()

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    def test_external_only_no_data_path(self, mock_adapters):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_adapters.return_value = [_FakeAdapter()]
        with pytest.raises(SystemExit):
            run_bmb(external_only=True, data_path=None)

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    def test_include_external_no_data_path(self, mock_adapters):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_adapters.return_value = [_FakeAdapter()]
        with pytest.raises(SystemExit):
            run_bmb(include_external=True, data_path=None)

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    @patch("mnemebrain_benchmark.bmb_cli._run_external_benchmarks")
    def test_include_external_runs_both(self, mock_ext, mock_load, mock_adapters, tmp_path):
        from mnemebrain_benchmark.bmb_cli import run_bmb
        from mnemebrain_benchmark.scenarios.schema import Action, Expectation, Scenario

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [
            Scenario(
                name="s1", description="d", category="contradiction",
                requires=["store", "query"],
                actions=[
                    Action(label="s1", type="store", claim="test", evidence=[{}]),
                    Action(label="q1", type="query", claim="test"),
                ],
                expectations=[Expectation(action_label="s1", beliefs_stored=1)],
            )
        ]
        mock_report = MagicMock()
        mock_report.summary.return_value = "REPORT"
        mock_ext.return_value = [mock_report]

        output = str(tmp_path / "report.json")
        results = run_bmb(
            include_external=True,
            data_path="/fake/data.json",
            output=output,
        )
        assert "fake" in results
        mock_ext.assert_called_once()


# ===========================================================================
# external_evals.__main__
# ===========================================================================


class TestExternalEvalsMain:
    def test_help(self):
        from mnemebrain_benchmark.external_evals.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_subcommand(self):
        from mnemebrain_benchmark.external_evals.__main__ import main

        with pytest.raises(SystemExit):
            main([])

    @patch("mnemebrain_benchmark.external_evals.__main__.build_adapters")
    def test_build_systems_no_adapters(self, mock_adapters):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _build_systems

        mock_adapters.return_value = []
        args = argparse.Namespace(system=None, embedder=None, embedder_model=None)
        with pytest.raises(SystemExit):
            _build_systems(args)

    @patch("mnemebrain_benchmark.external_evals.__main__.build_adapters")
    def test_build_systems_returns_adapters(self, mock_adapters):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _build_systems

        adapter = _FakeAdapter()
        mock_adapters.return_value = [adapter]
        args = argparse.Namespace(system="langchain_buffer", embedder=None, embedder_model=None)
        result = _build_systems(args)
        assert result == [adapter]

    def test_output_path_for_single(self):
        from mnemebrain_benchmark.external_evals.__main__ import _output_path_for

        assert _output_path_for("report.json", "fake", 1) == "report.json"

    def test_output_path_for_multiple(self):
        from mnemebrain_benchmark.external_evals.__main__ import _output_path_for

        assert _output_path_for("report.json", "fake", 2) == "report_fake.json"

    def test_output_path_for_no_extension(self):
        from mnemebrain_benchmark.external_evals.__main__ import _output_path_for

        assert _output_path_for("report", "fake", 2) == "report_fake.json"

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    def test_run_longmemeval_single_system(self, mock_run, mock_systems, capsys):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_longmemeval
        from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

        adapter = _FakeAdapter()
        mock_systems.return_value = [adapter]
        mock_run.return_value = BenchmarkReport(
            benchmark_name="longmemeval", system_name="fake",
        )

        args = argparse.Namespace(
            data_path="/fake/data.json",
            subset=None,
            limit=None,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=None,
        )
        _run_longmemeval(args)
        mock_run.assert_called_once()
        captured = capsys.readouterr()
        assert "LONGMEMEVAL" in captured.out

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    def test_run_longmemeval_with_output_json(self, mock_run, mock_systems, tmp_path):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_longmemeval
        from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

        adapter = _FakeAdapter()
        mock_systems.return_value = [adapter]
        mock_run.return_value = BenchmarkReport(
            benchmark_name="longmemeval", system_name="fake",
        )

        output = str(tmp_path / "lme.json")
        args = argparse.Namespace(
            data_path="/fake/data.json",
            subset=None,
            limit=None,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=output,
        )
        _run_longmemeval(args)
        assert (tmp_path / "lme.json").exists()

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    def test_run_longmemeval_multiple_systems_output(self, mock_run, mock_systems, tmp_path):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_longmemeval
        from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

        adapters = [_FakeAdapter("sys_a"), _FakeAdapter("sys_b")]
        mock_systems.return_value = adapters
        mock_run.side_effect = [
            BenchmarkReport(benchmark_name="longmemeval", system_name="sys_a"),
            BenchmarkReport(benchmark_name="longmemeval", system_name="sys_b"),
        ]

        output = str(tmp_path / "lme.json")
        args = argparse.Namespace(
            data_path="/fake/data.json",
            subset=None,
            limit=None,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=output,
        )
        _run_longmemeval(args)
        assert (tmp_path / "lme_sys_a.json").exists()
        assert (tmp_path / "lme_sys_b.json").exists()

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    def test_run_hotpotqa_single_system(self, mock_systems, tmp_path, capsys):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_hotpotqa

        adapter = _FakeAdapter()
        mock_systems.return_value = [adapter]
        data_path = _make_hotpotqa_dataset(tmp_path)

        args = argparse.Namespace(
            data_path=data_path,
            limit=1,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=None,
        )
        _run_hotpotqa(args)
        captured = capsys.readouterr()
        assert "HOTPOTQA" in captured.out

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    def test_run_hotpotqa_verbose(self, mock_systems, tmp_path, capsys):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_hotpotqa

        adapter = _FakeAdapter()
        mock_systems.return_value = [adapter]
        data_path = _make_hotpotqa_dataset(tmp_path)

        args = argparse.Namespace(
            data_path=data_path,
            limit=1,
            llm_extract=False,
            llm_answer=False,
            verbose=True,
            output_json=None,
        )
        _run_hotpotqa(args)
        captured = capsys.readouterr()
        assert "hpqa_001" in captured.err

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    def test_run_hotpotqa_with_output(self, mock_systems, tmp_path):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_hotpotqa

        adapter = _FakeAdapter()
        mock_systems.return_value = [adapter]
        data_path = _make_hotpotqa_dataset(tmp_path)

        output = str(tmp_path / "hpqa.json")
        args = argparse.Namespace(
            data_path=data_path,
            limit=1,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=output,
        )
        _run_hotpotqa(args)
        assert (tmp_path / "hpqa.json").exists()

    @patch("mnemebrain_benchmark.external_evals.__main__._build_systems")
    def test_run_hotpotqa_multiple_systems_output(self, mock_systems, tmp_path):
        import argparse

        from mnemebrain_benchmark.external_evals.__main__ import _run_hotpotqa

        adapters = [_FakeAdapter("sys_a"), _FakeAdapter("sys_b")]
        mock_systems.return_value = adapters
        data_path = _make_hotpotqa_dataset(tmp_path)

        output = str(tmp_path / "hpqa.json")
        args = argparse.Namespace(
            data_path=data_path,
            limit=1,
            llm_extract=False,
            llm_answer=False,
            verbose=False,
            output_json=output,
        )
        _run_hotpotqa(args)
        assert (tmp_path / "hpqa_sys_a.json").exists()
        assert (tmp_path / "hpqa_sys_b.json").exists()

    @patch("mnemebrain_benchmark.external_evals.__main__._run_longmemeval")
    def test_main_longmemeval_dispatch(self, mock_run):
        from mnemebrain_benchmark.external_evals.__main__ import main

        main(["longmemeval", "--data-path", "/fake/path", "--system", "langchain_buffer"])
        mock_run.assert_called_once()

    @patch("mnemebrain_benchmark.external_evals.__main__._run_hotpotqa")
    def test_main_hotpotqa_dispatch(self, mock_run):
        from mnemebrain_benchmark.external_evals.__main__ import main

        main(["hotpotqa", "--data-path", "/fake/path", "--system", "langchain_buffer"])
        mock_run.assert_called_once()

    def test_write_json_report(self, tmp_path):
        from mnemebrain_benchmark.external_evals.__main__ import _write_json_report
        from mnemebrain_benchmark.external_evals.scorer import (
            BenchmarkReport,
            QuestionScore,
            SubsetScore,
        )

        report = BenchmarkReport(
            benchmark_name="test",
            system_name="fake",
            subsets={
                "sub1": SubsetScore(
                    subset="sub1",
                    question_scores=[
                        QuestionScore("q1", "Q?", "yes", "yes", f1=1.0, em=1.0),
                    ],
                ),
            },
        )
        output = str(tmp_path / "report.json")
        _write_json_report(report, output)
        assert (tmp_path / "report.json").exists()

        data = json.loads((tmp_path / "report.json").read_text())
        assert data["benchmark"] == "test"
        assert data["system"] == "fake"
        assert data["subsets"]["sub1"]["count"] == 1


# ===========================================================================
# external_evals/longmemeval/run.py
# ===========================================================================


class TestLongmemevalRun:
    def test_run_longmemeval_with_system(self, tmp_path):
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        report = run_longmemeval(data_path=data_path, system=adapter)
        assert report.benchmark_name == "longmemeval"
        assert report.system_name == "fake"
        assert report.total_questions == 1

    def test_run_longmemeval_with_limit(self, tmp_path):
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        report = run_longmemeval(data_path=data_path, system=adapter, limit=1)
        assert report.total_questions == 1

    def test_run_longmemeval_verbose(self, tmp_path, capsys):
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        run_longmemeval(data_path=data_path, system=adapter, verbose=True)
        captured = capsys.readouterr()
        assert "lme_001" in captured.err
        assert "Loaded" in captured.err

    def test_run_longmemeval_subset_filter(self, tmp_path):
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        report = run_longmemeval(
            data_path=data_path, system=adapter, subset="knowledge_update",
        )
        assert report.total_questions == 1

    def test_run_longmemeval_subset_no_match(self, tmp_path):
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        report = run_longmemeval(
            data_path=data_path, system=adapter, subset="nonexistent",
        )
        assert report.total_questions == 0

    def test_create_system_lite(self):
        from mnemebrain_benchmark.external_evals.longmemeval.run import _create_system

        with patch("mnemebrain_benchmark.adapter_factory.build_adapters") as mock:
            adapter = _FakeAdapter("mnemebrain_lite")
            mock.return_value = [adapter]
            system = _create_system("lite")
            assert system.name() == "mnemebrain_lite"
            mock.assert_called_with(adapter_filter="mnemebrain_lite")

    def test_create_system_full(self):
        from mnemebrain_benchmark.external_evals.longmemeval.run import _create_system

        with patch("mnemebrain_benchmark.adapter_factory.build_adapters") as mock:
            adapter = _FakeAdapter("mnemebrain")
            mock.return_value = [adapter]
            system = _create_system("full")
            assert system.name() == "mnemebrain"
            mock.assert_called_with(adapter_filter="mnemebrain")

    def test_create_system_by_adapter_name(self):
        from mnemebrain_benchmark.external_evals.longmemeval.run import _create_system

        with patch("mnemebrain_benchmark.adapter_factory.build_adapters") as mock:
            adapter = _FakeAdapter("langchain_buffer")
            mock.return_value = [adapter]
            system = _create_system("langchain_buffer")
            assert system.name() == "langchain_buffer"

    def test_create_system_not_found(self):
        from mnemebrain_benchmark.external_evals.longmemeval.run import _create_system

        with patch("mnemebrain_benchmark.adapter_factory.build_adapters") as mock:
            mock.return_value = []
            with pytest.raises(ValueError, match="Could not build adapter"):
                _create_system("nonexistent")

    def test_run_longmemeval_fallback_to_create_system(self, tmp_path):
        """When system=None, run_longmemeval uses _create_system."""
        from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

        data_path = _make_lme_dataset(tmp_path)
        adapter = _FakeAdapter()

        with patch("mnemebrain_benchmark.external_evals.longmemeval.run._create_system") as mock:
            mock.return_value = adapter
            report = run_longmemeval(data_path=data_path, system_type="lite")
            mock.assert_called_once_with("lite", embedding_provider=None)
            assert report.system_name == "fake"


class TestLongmemevalRunMain:
    def test_help(self):
        from mnemebrain_benchmark.external_evals.longmemeval.run import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @patch("mnemebrain_benchmark.adapter_factory.build_adapters")
    def test_main_no_adapters(self, mock_adapters):
        from mnemebrain_benchmark.external_evals.longmemeval.run import main

        mock_adapters.return_value = []
        with pytest.raises(SystemExit):
            main(["--data-path", "/fake/path"])

    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    @patch("mnemebrain_benchmark.adapter_factory.build_adapters")
    def test_main_runs(self, mock_adapters, mock_run, capsys):
        from mnemebrain_benchmark.external_evals.longmemeval.run import main
        from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

        adapter = _FakeAdapter()
        mock_adapters.return_value = [adapter]
        mock_run.return_value = BenchmarkReport(
            benchmark_name="longmemeval", system_name="fake",
        )

        main(["--data-path", "/fake/path"])
        mock_run.assert_called_once()
        captured = capsys.readouterr()
        assert "LONGMEMEVAL" in captured.out

    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    @patch("mnemebrain_benchmark.adapter_factory.build_adapters")
    def test_main_with_output_json(self, mock_adapters, mock_run, tmp_path):
        from mnemebrain_benchmark.external_evals.longmemeval.run import main
        from mnemebrain_benchmark.external_evals.scorer import (
            BenchmarkReport,
            QuestionScore,
            SubsetScore,
        )

        adapter = _FakeAdapter()
        mock_adapters.return_value = [adapter]
        mock_run.return_value = BenchmarkReport(
            benchmark_name="longmemeval",
            system_name="fake",
            subsets={
                "sub1": SubsetScore(
                    subset="sub1",
                    question_scores=[
                        QuestionScore("q1", "Q?", "yes", "yes", f1=1.0, em=1.0),
                    ],
                ),
            },
        )

        output = str(tmp_path / "result.json")
        main(["--data-path", "/fake/path", "--output-json", output])
        assert (tmp_path / "result.json").exists()

    @patch("mnemebrain_benchmark.external_evals.longmemeval.run.run_longmemeval")
    @patch("mnemebrain_benchmark.adapter_factory.build_adapters")
    def test_main_with_system_filter(self, mock_adapters, mock_run):
        from mnemebrain_benchmark.external_evals.longmemeval.run import main
        from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

        adapter = _FakeAdapter()
        mock_adapters.return_value = [adapter]
        mock_run.return_value = BenchmarkReport(
            benchmark_name="longmemeval", system_name="fake",
        )

        main(["--data-path", "/fake/path", "--system", "langchain_buffer"])
        mock_adapters.assert_called_with("langchain_buffer", None, None)


# ===========================================================================
# answer_generator: _extract_answer_span
# ===========================================================================


class TestExtractAnswerSpan:
    def test_strips_question_overlap(self):
        from mnemebrain_benchmark.external_evals.answer_generator import _extract_answer_span

        result = _extract_answer_span(
            "What degree did I graduate with?",
            "I graduated with a degree in Business Administration, which helped me",
        )
        assert "Business Administration" in result
        # Should be shorter than the full claim.
        assert len(result) < len("I graduated with a degree in Business Administration, which helped me")

    def test_numeric_span_for_how_many(self):
        from mnemebrain_benchmark.external_evals.answer_generator import _extract_answer_span

        result = _extract_answer_span(
            "How many years did I work there?",
            "I worked there for 12 years before moving on",
        )
        assert "12" in result

    def test_location_for_where(self):
        from mnemebrain_benchmark.external_evals.answer_generator import _extract_answer_span

        result = _extract_answer_span(
            "Where did I grow up?",
            "I grew up in San Francisco, California",
        )
        assert "San Francisco" in result

    def test_fallback_returns_claim(self):
        from mnemebrain_benchmark.external_evals.answer_generator import _extract_answer_span

        claim = "something"
        result = _extract_answer_span("question?", claim)
        assert result  # Non-empty.

    def test_empty_claim(self):
        from mnemebrain_benchmark.external_evals.answer_generator import _extract_answer_span

        assert _extract_answer_span("What?", "") == ""

    def test_no_llm_path_uses_extraction(self):
        from mnemebrain_benchmark.external_evals.answer_generator import answer_from_beliefs

        results = [QueryResult("b1", "I live in Boston, Massachusetts", 0.9, "true")]
        answer = answer_from_beliefs("Where do you live?", results)
        # Should not return the full claim verbatim; should extract a span.
        assert len(answer) <= len("I live in Boston, Massachusetts")


# ===========================================================================
# longmemeval adapter: assistant turn ingestion
# ===========================================================================


class TestAssistantTurnIngestion:
    def test_factual_assistant_turns_ingested(self, tmp_path):
        """Assistant turns with proper nouns/numbers should be ingested."""
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import (
            LongMemEvalAdapter,
            _is_factual_sentence,
        )

        assert _is_factual_sentence("Alice works at Google since 2019")
        assert _is_factual_sentence("The meeting is in January")
        assert not _is_factual_sentence("got it")
        assert not _is_factual_sentence("sure thing")

    def test_adapter_ingests_assistant_factual_content(self, tmp_path):
        """Adapter should ingest factual assistant content with lower weight."""
        adapter = _FakeAdapter()
        from mnemebrain_benchmark.external_evals.base import Scenario

        scenario = Scenario(
            scenario_id="test_001",
            subset="knowledge_update",
            history=[
                {"role": "user", "content": "I work at Google."},
                {"role": "assistant", "content": "You work at Google since 2020."},
                {"role": "assistant", "content": "Got it, thanks!"},
            ],
            questions=[],
        )

        from mnemebrain_benchmark.external_evals.longmemeval.adapter import LongMemEvalAdapter

        lme = LongMemEvalAdapter()

        store_calls = []
        original_store = adapter.store

        def tracking_store(claim, evidence):
            store_calls.append({"claim": claim, "evidence": evidence})
            return original_store(claim, evidence)

        adapter.store = tracking_store
        lme.ingest(adapter, scenario)

        # User turn should produce a store call, and the factual assistant turn too.
        # The non-factual "Got it, thanks!" should be filtered out.
        assert len(store_calls) >= 2

    def test_non_factual_assistant_turns_skipped(self):
        """Non-factual assistant turns should be filtered out."""
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import _is_factual_sentence

        assert not _is_factual_sentence("sure, no problem")
        assert not _is_factual_sentence("okay")
        assert not _is_factual_sentence("let me know if you need anything else")


# ===========================================================================
# longmemeval adapter: query reformulation
# ===========================================================================


class TestQueryReformulation:
    def test_what_pattern(self):
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import _reformulate_question

        result = _reformulate_question("What degree did I graduate with?")
        assert result is not None
        assert "I" in result or "i" in result.lower()

    def test_where_pattern(self):
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import _reformulate_question

        result = _reformulate_question("Where did I grow up?")
        assert result is not None

    def test_no_match_returns_none(self):
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import _reformulate_question

        result = _reformulate_question("Tell me about yourself")
        assert result is None

    def test_how_many_pattern(self):
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import _reformulate_question

        result = _reformulate_question("How many years did I work there?")
        assert result is not None

    def test_answer_uses_reformulation(self):
        """answer() should query with both original and reformulated question."""
        from mnemebrain_benchmark.external_evals.longmemeval.adapter import LongMemEvalAdapter

        adapter = _FakeAdapter()

        query_calls = []
        original_query = adapter.query

        def tracking_query(claim):
            query_calls.append(claim)
            return original_query(claim)

        adapter.query = tracking_query

        lme = LongMemEvalAdapter()
        lme.answer(adapter, {"question": "What degree did I graduate with?"})

        # Should have queried at least twice (original + reformulation).
        assert len(query_calls) >= 2


# ===========================================================================
# claim_extractor: _is_keepable and lowered _MIN_CLAIM_LENGTH
# ===========================================================================


class TestClaimExtractorImprovements:
    def test_min_claim_length_lowered(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import _MIN_CLAIM_LENGTH

        assert _MIN_CLAIM_LENGTH <= 6

    def test_is_keepable_with_digits(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import _is_keepable

        assert _is_keepable("2019")
        assert _is_keepable("MIT 5")

    def test_is_keepable_with_proper_noun(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import _is_keepable

        assert _is_keepable("Boston")
        assert _is_keepable("Dr Smith")

    def test_is_keepable_rejects_lowercase(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import _is_keepable

        assert not _is_keepable("ok")
        assert not _is_keepable("yes")

    def test_short_factual_claims_kept(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import extract_claims_sentence

        claims = extract_claims_sentence("MIT. yes. 2019.")
        texts = [c.text for c in claims]
        # Sentence splitter keeps trailing punctuation; check containment.
        assert any("MIT" in t for t in texts)
        assert any("2019" in t for t in texts)
        # "yes" is too short and not keepable — should be filtered.
        assert not any(t.strip(".") == "yes" for t in texts)

    def test_normal_sentences_still_extracted(self):
        from mnemebrain_benchmark.external_evals.claim_extractor import extract_claims_sentence

        claims = extract_claims_sentence("Alice works at Google. Bob lives in NYC.")
        assert len(claims) == 2
