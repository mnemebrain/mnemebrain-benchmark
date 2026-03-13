"""Extended CLI coverage tests for bmb_cli, system_cli, and task_evals.__main__."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)
from mnemebrain_benchmark.scenarios.schema import Action, Expectation, Scenario

# ---------------------------------------------------------------------------
# Shared fake adapter for tests
# ---------------------------------------------------------------------------


class _FakeAdapter(MemorySystem):
    """Minimal adapter that always works without external deps."""

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


def _make_scenario(name: str = "s1", category: str = "contradiction") -> Scenario:
    return Scenario(
        name=name,
        description="d",
        category=category,
        requires=["store", "query"],
        actions=[
            Action(label="s1", type="store", claim="test", evidence=[{}]),
            Action(label="q1", type="query", claim="test"),
        ],
        expectations=[Expectation(action_label="s1", beliefs_stored=1)],
    )


# ===========================================================================
# bmb_cli
# ===========================================================================


class TestAdapterFactory:
    """Test build_adapters from the shared adapter_factory module."""

    def test_build_adapters_no_filter_skips_unavailable(self):
        """With no filter, unavailable adapters are silently skipped."""
        from mnemebrain_benchmark.adapter_factory import build_adapters

        # At minimum langchain_buffer should always be available
        adapters = build_adapters(adapter_filter=None)
        names = [a.name() for a in adapters]
        assert "langchain_buffer" in names

    def test_build_adapters_mnemebrain_import_error(self):
        import sys as _sys

        from mnemebrain_benchmark.adapter_factory import build_adapters

        adapter_key = "mnemebrain_benchmark.adapters.mnemebrain_adapter"
        saved_adapter = _sys.modules.pop(adapter_key, None)
        try:
            with patch.dict("sys.modules", {"mnemebrain": None}):
                adapters = build_adapters(adapter_filter=None)
                names = [a.name() for a in adapters]
                assert "mnemebrain" not in names
        finally:
            if saved_adapter is not None:
                _sys.modules[adapter_key] = saved_adapter

    def test_build_adapters_langchain_only(self):
        from mnemebrain_benchmark.adapter_factory import build_adapters

        adapters = build_adapters("langchain_buffer")
        assert len(adapters) == 1
        assert adapters[0].name() == "langchain_buffer"

    def test_build_adapters_mem0_import_error_with_filter(self):
        """When filtering to mem0 and it's not available, should sys.exit."""
        from mnemebrain_benchmark import adapter_factory

        original = adapter_factory.build_adapters

        def patched_build(adapter_filter=None, embedder_name=None, embedder_model=None):
            if adapter_filter == "mem0":
                import builtins

                real_import = builtins.__import__

                def fake_import(name, *args, **kwargs):
                    if name == "mnemebrain_benchmark.adapters.mem0_adapter":
                        raise ImportError("no mem0")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    return original(adapter_filter)
            return original(adapter_filter)

        with patch.object(adapter_factory, "build_adapters", patched_build):
            with pytest.raises(SystemExit):
                patched_build("mem0")

    def test_build_adapters_openai_rag_import_error_with_filter(self):
        """When filtering to openai_rag and it's not available, should sys.exit."""
        from mnemebrain_benchmark import adapter_factory

        original = adapter_factory.build_adapters

        def patched_build(adapter_filter=None, embedder_name=None, embedder_model=None):
            if adapter_filter == "openai_rag":
                import builtins

                real_import = builtins.__import__

                def fake_import(name, *args, **kwargs):
                    if name == "mnemebrain_benchmark.adapters.openai_rag_adapter":
                        raise ImportError("no openai")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    return original(adapter_filter)
            return original(adapter_filter)

        with patch.object(adapter_factory, "build_adapters", patched_build):
            with pytest.raises(SystemExit):
                patched_build("openai_rag")


class TestBmbCliRunBmb:
    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    def test_run_bmb_category_filter(self, mock_load, mock_adapters, tmp_path):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [
            _make_scenario("s1", "contradiction"),
            _make_scenario("s2", "belief_revision"),
        ]
        output = str(tmp_path / "r.json")
        results = run_bmb(category="contradiction", output=output)
        assert "fake" in results

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    def test_run_bmb_scenario_filter(self, mock_load, mock_adapters, tmp_path):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [
            _make_scenario("target_scen", "contradiction"),
            _make_scenario("other_scen", "contradiction"),
        ]
        output = str(tmp_path / "r.json")
        results = run_bmb(scenario_name="target_scen", output=output)
        assert "fake" in results

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    def test_run_bmb_no_matching_scenarios(self, mock_load, mock_adapters):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_load.return_value = [_make_scenario("s1", "contradiction")]
        mock_adapters.return_value = [_FakeAdapter()]
        with pytest.raises(SystemExit):
            run_bmb(category="nonexistent_category")

    @patch("mnemebrain_benchmark.bmb_cli.build_adapters")
    @patch("mnemebrain_benchmark.bmb_cli.load_bmb_scenarios")
    def test_run_bmb_no_adapters(self, mock_load, mock_adapters):
        from mnemebrain_benchmark.bmb_cli import run_bmb

        mock_load.return_value = [_make_scenario()]
        mock_adapters.return_value = []
        with pytest.raises(SystemExit):
            run_bmb()

    @patch("mnemebrain_benchmark.bmb_cli.run_bmb")
    def test_main_passes_args(self, mock_run):
        from mnemebrain_benchmark.bmb_cli import main

        main(["--category", "contradiction", "--output", "out.json"])
        mock_run.assert_called_once_with(
            adapter_filter=None,
            category="contradiction",
            scenario_name=None,
            output="out.json",
            embedder_name=None,
            embedder_model=None,
            include_external=False,
            external_only=False,
            data_path=None,
            external_benchmark="all",
            external_limit=None,
        )

    @patch("mnemebrain_benchmark.bmb_cli.run_bmb")
    def test_main_with_adapter_and_scenario(self, mock_run):
        from mnemebrain_benchmark.bmb_cli import main

        main(["--adapter", "langchain_buffer", "--scenario", "my_scen"])
        mock_run.assert_called_once_with(
            adapter_filter="langchain_buffer",
            category=None,
            scenario_name="my_scen",
            output="bmb_report.json",
            embedder_name=None,
            embedder_model=None,
            include_external=False,
            external_only=False,
            data_path=None,
            external_benchmark="all",
            external_limit=None,
        )

    @patch("mnemebrain_benchmark.bmb_cli.run_bmb")
    def test_main_with_external_flags(self, mock_run):
        from mnemebrain_benchmark.bmb_cli import main

        main(["--external-only", "--data-path", "/tmp/data.json",
              "--external-benchmark", "longmemeval", "--external-limit", "10"])
        mock_run.assert_called_once_with(
            adapter_filter=None,
            category=None,
            scenario_name=None,
            output="bmb_report.json",
            embedder_name=None,
            embedder_model=None,
            include_external=False,
            external_only=True,
            data_path="/tmp/data.json",
            external_benchmark="longmemeval",
            external_limit=10,
        )


# ===========================================================================
# system_cli
# ===========================================================================


class TestSystemCliBuildAdapters:
    """System CLI now delegates to adapter_factory.build_adapters.
    These tests verify the adapter_factory integration via the CLI path."""

    def test_build_adapters_no_filter(self):
        """Without filter, builds whatever is importable."""
        from mnemebrain_benchmark.adapter_factory import build_adapters

        adapters = build_adapters(adapter_filter=None)
        assert isinstance(adapters, list)

    def test_build_adapters_mnemebrain_import_error_with_filter(self):
        """Filtering to mnemebrain when SDK is missing exits."""
        import builtins

        from mnemebrain_benchmark.adapter_factory import build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.mnemebrain_adapter":
                raise ImportError("no sdk")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                build_adapters("mnemebrain")

    def test_build_adapters_mnemebrain_lite_import_error_with_filter(self):
        """Filtering to mnemebrain_lite when not available exits."""
        import builtins

        from mnemebrain_benchmark.adapter_factory import build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.mnemebrain_lite_adapter":
                raise ImportError("no lite")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                build_adapters("mnemebrain_lite")

    def test_build_adapters_naive_import_error_with_filter(self):
        """Filtering to naive_baseline when ST is missing exits."""
        import builtins

        from mnemebrain_benchmark.adapter_factory import build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.naive_baseline":
                raise ImportError("no ST")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                build_adapters("naive_baseline")


class TestSystemCliMain:
    @patch("mnemebrain_benchmark.system_cli.build_adapters")
    @patch("mnemebrain_benchmark.system_cli.load_scenarios")
    def test_main_full_flow(self, mock_load, mock_adapters, tmp_path, capsys):
        from mnemebrain_benchmark.system_cli import main

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [_make_scenario()]
        output = str(tmp_path / "report.json")
        main(["--output", output])
        captured = capsys.readouterr()
        assert "Running" in captured.out
        assert (tmp_path / "report.json").exists()

    @patch("mnemebrain_benchmark.system_cli.build_adapters")
    @patch("mnemebrain_benchmark.system_cli.load_scenarios")
    def test_main_category_filter(self, mock_load, mock_adapters, tmp_path, capsys):
        from mnemebrain_benchmark.system_cli import main

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [
            _make_scenario("s1", "contradiction"),
            _make_scenario("s2", "retraction"),
        ]
        output = str(tmp_path / "report.json")
        main(["--category", "contradiction", "--output", output])
        captured = capsys.readouterr()
        assert "Running 1 scenarios" in captured.out

    @patch("mnemebrain_benchmark.system_cli.build_adapters")
    @patch("mnemebrain_benchmark.system_cli.load_scenarios")
    def test_main_scenario_filter(self, mock_load, mock_adapters, tmp_path, capsys):
        from mnemebrain_benchmark.system_cli import main

        mock_adapters.return_value = [_FakeAdapter()]
        mock_load.return_value = [_make_scenario("target", "contradiction")]
        output = str(tmp_path / "report.json")
        main(["--scenario", "target", "--output", output])
        captured = capsys.readouterr()
        assert "Running" in captured.out

    @patch("mnemebrain_benchmark.system_cli.build_adapters")
    @patch("mnemebrain_benchmark.system_cli.load_scenarios")
    def test_main_no_matching_scenarios(self, mock_load, mock_adapters):
        from mnemebrain_benchmark.system_cli import main

        mock_load.return_value = []
        with pytest.raises(SystemExit):
            main([])

    @patch("mnemebrain_benchmark.system_cli.build_adapters")
    @patch("mnemebrain_benchmark.system_cli.load_scenarios")
    def test_main_no_adapters(self, mock_load, mock_adapters):
        from mnemebrain_benchmark.system_cli import main

        mock_load.return_value = [_make_scenario()]
        mock_adapters.return_value = []
        with pytest.raises(SystemExit):
            main([])


# ===========================================================================
# task_evals.__main__
# ===========================================================================


class TestTaskEvalsMainBuildAdapters:
    def test_build_adapters_no_filter_exits_without_st(self):
        """Without sentence-transformers, _build_adapters exits even with no filter."""
        from mnemebrain_benchmark.task_evals.__main__ import _build_adapters

        # task_evals _build_adapters calls sys.exit(1) if ST is missing
        # since all its adapters need embeddings. This is expected behavior.
        with pytest.raises(SystemExit):
            _build_adapters(adapter_filter=None)

    def test_build_adapters_mnemebrain_import_error_with_filter(self):
        import builtins

        from mnemebrain_benchmark.task_evals.__main__ import _build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.mnemebrain_adapter":
                raise ImportError("no sdk")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                _build_adapters("mnemebrain")

    def test_build_adapters_mnemebrain_lite_import_error_with_filter(self):
        import builtins

        from mnemebrain_benchmark.task_evals.__main__ import _build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.mnemebrain_lite_adapter":
                raise ImportError("no lite")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                _build_adapters("mnemebrain_lite")

    def test_build_adapters_naive_import_error_with_filter(self):
        import builtins

        from mnemebrain_benchmark.task_evals.__main__ import _build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.naive_baseline":
                raise ImportError("no naive")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                _build_adapters("naive_baseline")

    def test_build_adapters_rag_import_error_with_filter(self):
        import builtins

        from mnemebrain_benchmark.task_evals.__main__ import _build_adapters

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mnemebrain_benchmark.adapters.rag_baseline":
                raise ImportError("no rag")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit):
                _build_adapters("rag_baseline")


class TestTaskEvalsMain:
    def test_help(self):
        from mnemebrain_benchmark.task_evals.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @patch("mnemebrain_benchmark.task_evals.__main__._build_adapters")
    def test_main_no_adapters(self, mock_adapters):
        from mnemebrain_benchmark.task_evals.__main__ import main

        mock_adapters.return_value = []
        with pytest.raises(SystemExit):
            main([])

    @patch("mnemebrain_benchmark.task_evals.__main__._build_adapters")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_preference_scenarios")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_qa_scenarios")
    def test_main_preference_only(self, mock_qa, mock_pref, mock_adapters, capsys):
        from mnemebrain_benchmark.task_evals.__main__ import main

        adapter = _FakeAdapter("naive_baseline")
        mock_adapters.return_value = [adapter]

        from mnemebrain_benchmark.task_evals.preference_tracking import (
            load_preference_scenarios,
        )

        mock_pref.return_value = load_preference_scenarios()

        main(["--eval", "preference"])
        captured = capsys.readouterr()
        assert "PREFERENCE_TRACKING" in captured.out
        mock_qa.assert_not_called()

    @patch("mnemebrain_benchmark.task_evals.__main__._build_adapters")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_preference_scenarios")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_qa_scenarios")
    def test_main_qa_only(self, mock_qa, mock_pref, mock_adapters, capsys):
        from mnemebrain_benchmark.task_evals.__main__ import main

        adapter = _FakeAdapter("naive_baseline")
        mock_adapters.return_value = [adapter]

        from mnemebrain_benchmark.task_evals.long_horizon_qa import load_qa_scenarios

        mock_qa.return_value = load_qa_scenarios()

        main(["--eval", "qa"])
        captured = capsys.readouterr()
        assert "LONG_HORIZON_QA" in captured.out
        mock_pref.assert_not_called()

    @patch("mnemebrain_benchmark.task_evals.__main__._build_adapters")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_preference_scenarios")
    @patch("mnemebrain_benchmark.task_evals.__main__.load_qa_scenarios")
    def test_main_both_evals(self, mock_qa, mock_pref, mock_adapters, capsys):
        from mnemebrain_benchmark.task_evals.__main__ import main

        adapter = _FakeAdapter("naive_baseline")
        mock_adapters.return_value = [adapter]

        from mnemebrain_benchmark.task_evals.long_horizon_qa import load_qa_scenarios
        from mnemebrain_benchmark.task_evals.preference_tracking import (
            load_preference_scenarios,
        )

        mock_pref.return_value = load_preference_scenarios()
        mock_qa.return_value = load_qa_scenarios()

        main([])
        captured = capsys.readouterr()
        assert "PREFERENCE_TRACKING" in captured.out
        assert "LONG_HORIZON_QA" in captured.out
