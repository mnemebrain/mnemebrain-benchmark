"""Tests for mnemebrain_benchmark.runner -- embedding benchmark orchestrator."""

from __future__ import annotations

import json

from mnemebrain_benchmark.runner import (
    PROVIDER_CONFIGS,
    _serialize_report,
    save_report,
)


class TestProviderConfigs:
    def test_has_configs(self):
        assert len(PROVIDER_CONFIGS) > 0

    def test_configs_are_tuples(self):
        for config in PROVIDER_CONFIGS:
            assert len(config) == 2
            provider_type, model_name = config
            assert provider_type in ("sentence_transformers", "openai")
            assert isinstance(model_name, str)


class TestSerializeReport:
    def test_float_rounding(self):
        report = {"score": 0.123456789}
        result = _serialize_report(report)
        assert result["score"] == 0.123457

    def test_nested_structures(self):
        report = {"data": [{"val": 1.23456789}]}
        result = _serialize_report(report)
        assert result["data"][0]["val"] == 1.234568

    def test_non_float_passthrough(self):
        report = {"key": "value", "num": 42}
        result = _serialize_report(report)
        assert result == {"key": "value", "num": 42}


class TestSaveReport:
    def test_save_and_load(self, tmp_path):
        report = {"threshold": 0.92, "providers": {"test": {"f1": 0.95}}}
        path = tmp_path / "report.json"
        save_report(report, path)
        with open(path) as f:
            data = json.load(f)
        assert data["threshold"] == 0.92

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "report.json"
        save_report({"data": 1}, path)
        assert path.exists()
