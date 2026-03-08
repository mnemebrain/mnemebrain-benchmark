"""Base types and scoring for task-level evaluations."""
from __future__ import annotations

from dataclasses import dataclass, field

from mnemebrain_benchmark.interface import QueryResult


@dataclass
class TaskAction:
    type: str  # store, retract, revise, wait_days
    claim: str | None = None
    evidence: list[dict] | None = None
    target_index: int | None = None  # index into prior store results
    wait_days: int | None = None


@dataclass
class TaskQuestion:
    query: str
    expected_keywords: list[str]
    rejected_keywords: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    question: TaskQuestion
    correct: bool
    returned_claims: list[str]


@dataclass
class TaskScenario:
    name: str
    description: str
    category: str
    actions: list[TaskAction]
    questions: list[TaskQuestion]


def score_question(question: TaskQuestion, results: list[QueryResult]) -> TaskResult:
    """Score a single question against query results.

    Filters out results with truth_state='false', then checks if
    the remaining results contain expected keywords and avoid rejected ones.
    """
    active = [r for r in results if r.truth_state != "false"]
    returned_claims = [r.claim for r in active]

    if not returned_claims:
        return TaskResult(question=question, correct=False, returned_claims=[])

    combined = " ".join(returned_claims).lower()
    has_expected = any(kw.lower() in combined for kw in question.expected_keywords)

    top_claim = returned_claims[0].lower()
    has_rejected = any(kw.lower() in top_claim for kw in question.rejected_keywords)

    correct = has_expected and not has_rejected

    return TaskResult(question=question, correct=correct, returned_claims=returned_claims)
