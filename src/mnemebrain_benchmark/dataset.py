"""Dataset loading and validation for the embedding benchmark."""
from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = {"same", "different"}
VALID_CATEGORIES = {"fact", "preference", "inference", "prediction"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class ClaimPair:
    """A single labeled claim pair for benchmarking."""

    id: str
    claim_a: str
    claim_b: str
    label: str  # "same" or "different"
    category: str  # fact, preference, inference, prediction
    difficulty: str  # easy, medium, hard


class BenchmarkDataset:
    """Gold-standard dataset of labeled claim pairs."""

    def __init__(self, pairs: list[ClaimPair]) -> None:
        self._pairs = pairs

    @classmethod
    def load(cls, path: Path | str | None = None) -> BenchmarkDataset:
        """Load and validate dataset from JSON file."""
        if path is not None:
            path = Path(path)
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        else:
            ref = importlib.resources.files("mnemebrain_benchmark") / "data" / "claim_pairs.json"
            raw = json.loads(ref.read_text(encoding="utf-8"))

        pairs: list[ClaimPair] = []
        seen_ids: set[str] = set()
        for i, entry in enumerate(raw):
            for field in ("id", "claim_a", "claim_b", "label", "category", "difficulty"):
                if field not in entry:
                    raise ValueError(f"Entry {i} missing required field '{field}'")

            if entry["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {entry['id']}")
            seen_ids.add(entry["id"])

            if entry["label"] not in VALID_LABELS:
                raise ValueError(
                    f"Entry {entry['id']}: invalid label '{entry['label']}', "
                    f"expected one of {VALID_LABELS}"
                )
            if entry["category"] not in VALID_CATEGORIES:
                raise ValueError(
                    f"Entry {entry['id']}: invalid category '{entry['category']}', "
                    f"expected one of {VALID_CATEGORIES}"
                )
            if entry["difficulty"] not in VALID_DIFFICULTIES:
                raise ValueError(
                    f"Entry {entry['id']}: invalid difficulty '{entry['difficulty']}', "
                    f"expected one of {VALID_DIFFICULTIES}"
                )

            fields = vars(ClaimPair)["__dataclass_fields__"]
            pairs.append(ClaimPair(**{k: entry[k] for k in fields}))

        return cls(pairs)

    @property
    def pairs(self) -> list[ClaimPair]:
        return list(self._pairs)

    def filter(
        self,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> BenchmarkDataset:
        """Return a new dataset filtered by category and/or difficulty."""
        filtered = self._pairs
        if category:
            filtered = [p for p in filtered if p.category == category]
        if difficulty:
            filtered = [p for p in filtered if p.difficulty == difficulty]
        return BenchmarkDataset(filtered)

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return f"BenchmarkDataset({len(self._pairs)} pairs)"
