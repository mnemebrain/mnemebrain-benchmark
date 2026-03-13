"""LongMemEval-specific benchmark adapter.

Handles ingestion of multi-session conversations with temporal markers
and session boundaries, mapping them to MnemeBrain's belief API.
"""

from __future__ import annotations

import re

from mnemebrain_benchmark.external_evals.answer_generator import answer_from_beliefs
from mnemebrain_benchmark.external_evals.base import ExternalBenchmarkAdapter, Scenario
from mnemebrain_benchmark.external_evals.claim_extractor import extract_claims
from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval
from mnemebrain_benchmark.external_evals.scorer import token_f1

# Patterns for converting questions to declarative form for better embedding match.
_REFORMULATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^what\s+(.+?)\s+(?:did|do|does|has|have)\s+(\w+)\s+(.+?)[\?.]?$", re.I),
     r"\2 \3 \1"),
    (re.compile(r"^where\s+(?:did|do|does|has|have)\s+(\w+)\s+(.+?)[\?.]?$", re.I),
     r"\1 \2"),
    (re.compile(r"^where\s+(?:is|are|was|were)\s+(.+?)[\?.]?$", re.I),
     r"\1 is"),
    (re.compile(r"^how\s+(?:many|much|long|far|old)\s+(.+?)\s+(?:did|do|does|has|have)\s+(\w+)\s+(.+?)[\?.]?$", re.I),
     r"\2 \3 \1"),
    (re.compile(r"^when\s+(?:did|do|does|has|have)\s+(\w+)\s+(.+?)[\?.]?$", re.I),
     r"\1 \2"),
    (re.compile(r"^who\s+(?:is|are|was|were)\s+(.+?)[\?.]?$", re.I),
     r"\1 is"),
    (re.compile(r"^who\s+(?:did|do|does|has|have)\s+(\w+)\s+(.+?)[\?.]?$", re.I),
     r"\1 \2"),
]


def _reformulate_question(question: str) -> str | None:
    """Convert a question to declarative form for better embedding similarity.

    Returns the declarative form, or None if no pattern matches.
    """
    q = question.strip()
    for pattern, replacement in _REFORMULATION_PATTERNS:
        m = pattern.match(q)
        if m:
            return pattern.sub(replacement, q).strip()
    return None


def _is_factual_sentence(sentence: str) -> bool:
    """Check if an assistant sentence contains factual content worth storing.

    Returns True if the sentence contains proper nouns, numbers, dates,
    or factual verbs — indicating retrievable facts rather than filler.
    """
    # Contains a number (year, quantity, date component).
    if re.search(r"\d{2,}", sentence):
        return True
    # Contains a capitalized word (not sentence-start) — likely proper noun.
    words = sentence.split()
    if len(words) > 1 and any(w[0].isupper() and w.isalpha() for w in words[1:]):
        return True
    # Contains date-like patterns.
    if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", sentence, re.I):
        return True
    return False


class LongMemEvalAdapter(ExternalBenchmarkAdapter):
    """Adapter for LongMemEval benchmark.

    Ingests multi-session conversational histories into MnemeBrain,
    then answers questions using belief retrieval.
    """

    def __init__(
        self,
        *,
        use_llm_extract: bool = False,
        llm_fn: object | None = None,
        use_llm_answer: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            use_llm_extract: Use LLM for claim extraction (vs sentence splitting).
            llm_fn: LLM callable for extraction and/or answer generation.
            use_llm_answer: Use LLM for answer synthesis from beliefs.
        """
        self._use_llm_extract = use_llm_extract
        self._llm_fn = llm_fn
        self._use_llm_answer = use_llm_answer

    def name(self) -> str:
        return "longmemeval"

    def load_dataset(self, path: str, subset: str | None = None) -> list[Scenario]:
        return load_longmemeval(path, subset=subset)

    def ingest(self, system: object, scenario: Scenario) -> None:
        """Ingest a LongMemEval scenario's conversation history.

        Strategy:
        - Process turns chronologically, preserving session boundaries.
        - Extract claims from user messages with standard weight.
        - Extract factual claims from assistant messages with lower weight.
        - Use session and timestamp as source_ref for provenance.
        - For "knowledge_update" scenarios, newer claims about the same topic
          naturally supersede older ones through MnemeBrain's revise mechanism.
        """
        mem = system  # type: MemorySystem

        prev_claims_by_topic: dict[str, str] = {}  # rough topic -> belief_id

        for turn in scenario.history:
            role = turn.get("role", "user")

            content = turn.get("content", "")
            if not content.strip():
                continue

            # For assistant turns, only keep factual sentences.
            if role == "assistant":
                factual_lines = [
                    s.strip() for s in re.split(r"[.!?]+", content)
                    if s.strip() and _is_factual_sentence(s.strip())
                ]
                if not factual_lines:
                    continue
                content = ". ".join(factual_lines) + "."

            # Set evidence weights: user content is more reliable.
            weight = 0.7 if role == "user" else 0.5
            reliability = 0.8 if role == "user" else 0.6

            session = turn.get("session", "")
            timestamp = turn.get("timestamp", "")
            source_ref = f"session:{session}" if session else "conversation"
            if timestamp:
                source_ref = f"{source_ref}@{timestamp}"

            claims = extract_claims(
                content,
                use_llm=self._use_llm_extract,
                llm_fn=self._llm_fn,
                source_ref=source_ref,
                timestamp=timestamp,
                session=session,
            )

            for claim in claims:
                evidence = [{
                    "source_ref": claim.source_ref,
                    "content": claim.text,
                    "polarity": "supports",
                    "weight": weight,
                    "reliability": reliability,
                }]

                result = mem.store(claim=claim.text, evidence=evidence)

                topic_key = " ".join(claim.text.split()[:4]).lower()
                if topic_key in prev_claims_by_topic:
                    old_bid = prev_claims_by_topic[topic_key]
                    try:
                        mem.revise(
                            belief_id=old_bid,
                            evidence=[{
                                "source_ref": claim.source_ref,
                                "content": claim.text,
                                "polarity": "supports",
                                "weight": weight + 0.1,
                                "reliability": reliability + 0.1,
                            }],
                        )
                    except (NotImplementedError, Exception):
                        pass

                prev_claims_by_topic[topic_key] = result.belief_id

    def answer(self, system: object, question: dict) -> str:
        """Answer a LongMemEval question using belief retrieval.

        Queries with both the original question and a declarative
        reformulation (if available), merging and deduplicating results
        to improve recall against stored claims.
        """
        mem = system  # type: MemorySystem
        q_text = question.get("question", "")

        results = mem.query(q_text)

        # Query with declarative reformulation too, merge results.
        decl = _reformulate_question(q_text)
        if decl:
            extra = mem.query(decl)
            seen_ids = {r.belief_id for r in results}
            for r in extra:
                if r.belief_id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.belief_id)

        llm_fn = self._llm_fn if self._use_llm_answer else None
        return answer_from_beliefs(q_text, results, llm_fn=llm_fn)

    def score(self, predicted: str, gold: str) -> float:
        """Score using token-level F1 (standard for LongMemEval)."""
        return token_f1(predicted, gold)
