"""Abstract base for external benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Scenario:
    """A single benchmark scenario (conversation history + questions).

    Attributes:
        scenario_id: Unique identifier from the benchmark dataset.
        subset: Category/subset name (e.g. "knowledge_update", "temporal").
        history: List of conversation turns or passages to ingest.
            Each entry is a dict with at least {"role": str, "content": str}
            and optionally {"timestamp": str, "session": str}.
        questions: List of questions to answer after ingestion.
            Each entry is {"question": str, "gold_answer": str, ...}.
        metadata: Any additional benchmark-specific metadata.
    """

    scenario_id: str
    subset: str
    history: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ExternalBenchmarkAdapter(ABC):
    """ABC for adapting external benchmarks to MnemeBrain's belief API.

    Subclasses implement dataset loading, ingestion, answering, and scoring
    for a specific benchmark (LongMemEval, HotpotQA, etc.).
    """

    @abstractmethod
    def name(self) -> str:
        """Return the benchmark name (e.g. 'longmemeval')."""
        ...

    @abstractmethod
    def load_dataset(self, path: str, subset: str | None = None) -> list[Scenario]:
        """Load benchmark scenarios from disk or HuggingFace.

        Args:
            path: Path to dataset directory or HuggingFace dataset ID.
            subset: Optional subset filter (e.g. "knowledge_update").

        Returns:
            List of Scenario objects ready for ingestion.
        """
        ...

    @abstractmethod
    def ingest(self, system: object, scenario: Scenario) -> None:
        """Ingest a scenario's history into the memory system.

        Args:
            system: A MemorySystem-compatible object (lite or full).
            scenario: The scenario whose history to ingest.
        """
        ...

    @abstractmethod
    def answer(self, system: object, question: dict) -> str:
        """Generate a natural-language answer from the memory system.

        Args:
            system: A MemorySystem-compatible object.
            question: A question dict with at least {"question": str}.

        Returns:
            Predicted answer string.
        """
        ...

    @abstractmethod
    def score(self, predicted: str, gold: str) -> float:
        """Score a single predicted answer against gold.

        Args:
            predicted: The system's answer.
            gold: The gold/reference answer.

        Returns:
            Score in [0, 1].
        """
        ...
