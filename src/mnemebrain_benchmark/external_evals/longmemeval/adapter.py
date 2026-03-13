"""LongMemEval-specific benchmark adapter.

Handles ingestion of multi-session conversations with temporal markers
and session boundaries, mapping them to MnemeBrain's belief API.
"""

from __future__ import annotations

from mnemebrain_benchmark.external_evals.answer_generator import answer_from_beliefs
from mnemebrain_benchmark.external_evals.base import ExternalBenchmarkAdapter, Scenario
from mnemebrain_benchmark.external_evals.claim_extractor import extract_claims
from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval
from mnemebrain_benchmark.external_evals.scorer import token_f1


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
        - Extract claims from user messages (assistants are context, not facts).
        - Use session and timestamp as source_ref for provenance.
        - For "knowledge_update" scenarios, newer claims about the same topic
          naturally supersede older ones through MnemeBrain's revise mechanism.
        """
        mem = system  # type: MemorySystem

        prev_claims_by_topic: dict[str, str] = {}  # rough topic -> belief_id

        for turn in scenario.history:
            role = turn.get("role", "user")

            if role != "user":
                continue

            content = turn.get("content", "")
            if not content.strip():
                continue

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
                    "weight": 0.7,
                    "reliability": 0.8,
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
                                "weight": 0.8,
                                "reliability": 0.9,
                            }],
                        )
                    except (NotImplementedError, Exception):
                        pass

                prev_claims_by_topic[topic_key] = result.belief_id

    def answer(self, system: object, question: dict) -> str:
        """Answer a LongMemEval question using belief retrieval."""
        mem = system  # type: MemorySystem
        q_text = question.get("question", "")

        results = mem.query(q_text)

        llm_fn = self._llm_fn if self._use_llm_answer else None
        return answer_from_beliefs(q_text, results, llm_fn=llm_fn)

    def score(self, predicted: str, gold: str) -> float:
        """Score using token-level F1 (standard for LongMemEval)."""
        return token_f1(predicted, gold)
