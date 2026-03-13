"""Tests for adapter_factory — shared adapter building logic."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from mnemebrain_benchmark.adapter_factory import ALL_ADAPTERS, build_adapters


class TestAllAdapters:
    def test_contains_expected_adapters(self):
        assert len(ALL_ADAPTERS) == 8
        for name in [
            "mnemebrain", "mnemebrain_lite", "naive_baseline",
            "langchain_buffer", "rag_baseline", "structured_memory",
            "mem0", "openai_rag",
        ]:
            assert name in ALL_ADAPTERS


class TestBuildAdaptersImportErrors:
    """Test each adapter's ImportError path when explicitly filtered."""

    def _make_fake_import(self, blocked_module: str):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == blocked_module:
                raise ImportError(f"fake: {blocked_module}")
            return real_import(name, *args, **kwargs)

        return fake_import

    def test_rag_baseline_import_error_with_filter(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.rag_baseline"
        )):
            with pytest.raises(SystemExit):
                build_adapters("rag_baseline")

    def test_structured_memory_import_error_with_filter(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.structured_memory"
        )):
            with pytest.raises(SystemExit):
                build_adapters("structured_memory")

    def test_mem0_import_error_with_filter(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.mem0_adapter"
        )):
            with pytest.raises(SystemExit):
                build_adapters("mem0")

    def test_openai_rag_import_error_with_filter(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.openai_rag_adapter"
        )):
            with pytest.raises(SystemExit):
                build_adapters("openai_rag")

    def test_rag_baseline_import_error_no_filter_skipped(self):
        """Without filter, ImportError is silently skipped."""
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.rag_baseline"
        )):
            adapters = build_adapters(adapter_filter=None)
            names = [a.name() for a in adapters]
            assert "rag_baseline" not in names

    def test_structured_memory_import_error_no_filter_skipped(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.structured_memory"
        )):
            adapters = build_adapters(adapter_filter=None)
            names = [a.name() for a in adapters]
            assert "structured_memory" not in names

    def test_mem0_import_error_no_filter_skipped(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.mem0_adapter"
        )):
            adapters = build_adapters(adapter_filter=None)
            names = [a.name() for a in adapters]
            assert "mem0" not in names

    def test_openai_rag_import_error_no_filter_skipped(self):
        with patch("builtins.__import__", side_effect=self._make_fake_import(
            "mnemebrain_benchmark.adapters.openai_rag_adapter"
        )):
            adapters = build_adapters(adapter_filter=None)
            names = [a.name() for a in adapters]
            assert "openai_rag" not in names


class TestBuildAdaptersEmbedder:
    def test_langchain_no_embedder_needed(self):
        adapters = build_adapters("langchain_buffer")
        assert len(adapters) == 1
        assert adapters[0].name() == "langchain_buffer"

    def test_passes_embedder_args(self):
        """Embedder args are forwarded to build_embedder."""
        with patch("mnemebrain_benchmark.adapter_factory.build_embedder") as mock_embedder:
            mock_embedder.return_value = type("E", (), {"embed": lambda s, t: [0.0], "similarity": lambda s, a, b: 0.0})()
            # naive_baseline needs an embedder
            try:
                build_adapters("naive_baseline", embedder_name="openai", embedder_model="text-embedding-3-large")
            except Exception:
                pass
            if mock_embedder.called:
                mock_embedder.assert_called_with("openai", "text-embedding-3-large")
