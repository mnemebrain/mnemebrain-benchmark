"""HotpotQA-specific benchmark adapter.

Requires the full MnemeBrain backend with HippoRAG for multi-hop retrieval.
Paragraphs are ingested as beliefs, then 2-hop queries use graph traversal.
"""

from __future__ import annotations

from mnemebrain_benchmark.external_evals.answer_generator import answer_from_beliefs
from mnemebrain_benchmark.external_evals.base import ExternalBenchmarkAdapter, Scenario
from mnemebrain_benchmark.external_evals.claim_extractor import extract_claims
from mnemebrain_benchmark.external_evals.hotpotqa.loader import load_hotpotqa
from mnemebrain_benchmark.external_evals.scorer import token_f1
from mnemebrain_benchmark.interface import Capability


class HotpotQAAdapter(ExternalBenchmarkAdapter):
    """Adapter for HotpotQA multi-hop QA benchmark.

    Requires a system with HIPPORAG capability for multi-hop retrieval.
    Falls back to single-hop query if HIPPORAG is not available.
    """

    def __init__(
        self,
        *,
        use_llm_extract: bool = False,
        llm_fn: object | None = None,
        use_llm_answer: bool = False,
    ) -> None:
        self._use_llm_extract = use_llm_extract
        self._llm_fn = llm_fn
        self._use_llm_answer = use_llm_answer

    def name(self) -> str:
        return "hotpotqa"

    def load_dataset(self, path: str, subset: str | None = None) -> list[Scenario]:
        return load_hotpotqa(path)

    def ingest(self, system: object, scenario: Scenario) -> None:
        """Ingest HotpotQA context paragraphs as beliefs.

        Each paragraph is ingested as a single belief with its title
        as source_ref, enabling HippoRAG to link related paragraphs
        through shared evidence sources.
        """
        mem = system  # type: MemorySystem

        for entry in scenario.history:
            content = entry.get("content", "")
            title = entry.get("title", "")
            source_ref = entry.get("source_ref", f"hotpotqa:{title}")

            if not content.strip():
                continue

            # Extract claims from the paragraph.
            claims = extract_claims(
                content,
                use_llm=self._use_llm_extract,
                llm_fn=self._llm_fn,
                source_ref=source_ref,
            )

            for claim in claims:
                evidence = [{
                    "source_ref": source_ref,
                    "content": claim.text,
                    "polarity": "supports",
                    "weight": 0.7,
                    "reliability": 0.8,
                }]
                mem.store(claim=claim.text, evidence=evidence)

    def answer(self, system: object, question: dict) -> str:
        """Answer using multi-hop retrieval if available, else single-hop."""
        mem = system  # type: MemorySystem
        q_text = question.get("question", "")

        # Try multi-hop first (HippoRAG).
        if Capability.HIPPORAG in mem.capabilities():
            try:
                results = mem.query_multihop(q_text)
                if results:
                    llm_fn = self._llm_fn if self._use_llm_answer else None
                    return answer_from_beliefs(q_text, results, llm_fn=llm_fn)
            except NotImplementedError:
                pass

        # Fall back to single-hop query.
        results = mem.query(q_text)
        llm_fn = self._llm_fn if self._use_llm_answer else None
        return answer_from_beliefs(q_text, results, llm_fn=llm_fn)

    def score(self, predicted: str, gold: str) -> float:
        """Score using token-level F1 (standard for HotpotQA)."""
        return token_f1(predicted, gold)
