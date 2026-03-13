"""Scoring utilities for external benchmarks.

Implements token-level F1, Exact Match, and per-subset aggregation
following standard QA methodology.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    """Normalize text for scoring: lowercase, strip punctuation and articles."""
    text = text.lower()
    # Remove articles.
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace.
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return _normalize(text).split()


def token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 score between predicted and gold answers.

    Standard QA F1: harmonic mean of token-level precision and recall
    after normalization (lowercase, strip articles/punctuation).

    Args:
        predicted: Predicted answer string.
        gold: Gold/reference answer string.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def exact_match(predicted: str, gold: str) -> float:
    """Compute exact match score after normalization.

    Args:
        predicted: Predicted answer string.
        gold: Gold/reference answer string.

    Returns:
        1.0 if normalized strings match, 0.0 otherwise.
    """
    return 1.0 if _normalize(predicted) == _normalize(gold) else 0.0


@dataclass
class QuestionScore:
    """Score for a single question."""

    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    f1: float
    em: float


@dataclass
class SubsetScore:
    """Aggregated scores for a benchmark subset."""

    subset: str
    question_scores: list[QuestionScore] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.question_scores)

    @property
    def avg_f1(self) -> float:
        if not self.question_scores:
            return 0.0
        return sum(q.f1 for q in self.question_scores) / len(self.question_scores)

    @property
    def avg_em(self) -> float:
        if not self.question_scores:
            return 0.0
        return sum(q.em for q in self.question_scores) / len(self.question_scores)


@dataclass
class BenchmarkReport:
    """Full report for a benchmark run."""

    benchmark_name: str
    system_name: str
    subsets: dict[str, SubsetScore] = field(default_factory=dict)

    @property
    def overall_f1(self) -> float:
        all_scores = [q for s in self.subsets.values() for q in s.question_scores]
        if not all_scores:
            return 0.0
        return sum(q.f1 for q in all_scores) / len(all_scores)

    @property
    def overall_em(self) -> float:
        all_scores = [q for s in self.subsets.values() for q in s.question_scores]
        if not all_scores:
            return 0.0
        return sum(q.em for q in all_scores) / len(all_scores)

    @property
    def total_questions(self) -> int:
        return sum(s.count for s in self.subsets.values())

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"\n{self.benchmark_name.upper()} BENCHMARK REPORT",
            f"System: {self.system_name}",
            "=" * 60,
            f"{'Subset':<25} {'Count':>6} {'F1':>8} {'EM':>8}",
            "-" * 60,
        ]
        for name, subset in sorted(self.subsets.items()):
            lines.append(
                f"{name:<25} {subset.count:>6} "
                f"{subset.avg_f1 * 100:>7.1f}% {subset.avg_em * 100:>7.1f}%"
            )
        lines.append("-" * 60)
        lines.append(
            f"{'OVERALL':<25} {self.total_questions:>6} "
            f"{self.overall_f1 * 100:>7.1f}% {self.overall_em * 100:>7.1f}%"
        )
        lines.append("=" * 60)
        return "\n".join(lines)
