"""Tests for mnemebrain_benchmark.dataset."""
from __future__ import annotations

import json

import pytest

from mnemebrain_benchmark.dataset import (
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    VALID_LABELS,
    BenchmarkDataset,
    ClaimPair,
)


def _make_pair(**overrides):
    defaults = {
        "id": "p1",
        "claim_a": "The sky is blue",
        "claim_b": "The sky is blue",
        "label": "same",
        "category": "fact",
        "difficulty": "easy",
    }
    defaults.update(overrides)
    return defaults


class TestClaimPair:
    def test_frozen(self):
        p = ClaimPair(id="1", claim_a="a", claim_b="b", label="same", category="fact", difficulty="easy")
        with pytest.raises(AttributeError):
            p.id = "2"  # type: ignore[misc]


class TestValidSets:
    def test_labels(self):
        assert VALID_LABELS == {"same", "different"}

    def test_categories(self):
        assert VALID_CATEGORIES == {"fact", "preference", "inference", "prediction"}

    def test_difficulties(self):
        assert VALID_DIFFICULTIES == {"easy", "medium", "hard"}


class TestBenchmarkDataset:
    def test_load_from_file(self, tmp_path):
        data = [_make_pair(id="p1"), _make_pair(id="p2", label="different")]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        assert len(ds) == 2
        assert ds.pairs[0].id == "p1"

    def test_load_bundled(self):
        ds = BenchmarkDataset.load()
        assert len(ds) > 0

    def test_duplicate_id(self, tmp_path):
        data = [_make_pair(id="dup"), _make_pair(id="dup")]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Duplicate id"):
            BenchmarkDataset.load(path)

    def test_missing_field(self, tmp_path):
        data = [{"id": "p1", "claim_a": "a"}]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing required field"):
            BenchmarkDataset.load(path)

    def test_invalid_label(self, tmp_path):
        data = [_make_pair(label="invalid")]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="invalid label"):
            BenchmarkDataset.load(path)

    def test_invalid_category(self, tmp_path):
        data = [_make_pair(category="invalid")]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="invalid category"):
            BenchmarkDataset.load(path)

    def test_invalid_difficulty(self, tmp_path):
        data = [_make_pair(difficulty="invalid")]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="invalid difficulty"):
            BenchmarkDataset.load(path)

    def test_filter_category(self, tmp_path):
        data = [
            _make_pair(id="p1", category="fact"),
            _make_pair(id="p2", category="preference"),
        ]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        filtered = ds.filter(category="fact")
        assert len(filtered) == 1
        assert filtered.pairs[0].id == "p1"

    def test_filter_difficulty(self, tmp_path):
        data = [
            _make_pair(id="p1", difficulty="easy"),
            _make_pair(id="p2", difficulty="hard"),
        ]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        filtered = ds.filter(difficulty="hard")
        assert len(filtered) == 1

    def test_filter_both(self, tmp_path):
        data = [
            _make_pair(id="p1", category="fact", difficulty="easy"),
            _make_pair(id="p2", category="fact", difficulty="hard"),
            _make_pair(id="p3", category="preference", difficulty="easy"),
        ]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        filtered = ds.filter(category="fact", difficulty="easy")
        assert len(filtered) == 1

    def test_repr(self, tmp_path):
        data = [_make_pair()]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        assert "1 pairs" in repr(ds)

    def test_pairs_returns_copy(self, tmp_path):
        data = [_make_pair()]
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(data))
        ds = BenchmarkDataset.load(path)
        pairs1 = ds.pairs
        pairs2 = ds.pairs
        assert pairs1 == pairs2
        assert pairs1 is not pairs2
