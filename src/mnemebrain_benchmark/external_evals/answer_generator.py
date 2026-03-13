"""Generate natural-language answers from belief query results.

External benchmarks expect free-form text answers, but MnemeBrain returns
structured QueryResult objects. This module bridges the gap.
"""

from __future__ import annotations

from mnemebrain_benchmark.interface import QueryResult


def answer_from_beliefs(
    question: str,
    results: list[QueryResult],
    *,
    llm_fn: object | None = None,
    max_beliefs: int = 5,
) -> str:
    """Convert belief query results into a natural-language answer.

    Two modes:
    - With llm_fn: Uses an LLM to synthesize a coherent answer from beliefs.
    - Without llm_fn: Returns the top belief claim verbatim (fast, no LLM cost).

    Args:
        question: The question being answered.
        results: Query results from the memory system.
        llm_fn: Optional callable(str) -> str for LLM synthesis.
        max_beliefs: Maximum number of beliefs to include in the LLM prompt.

    Returns:
        Natural-language answer string.
    """
    # Filter out FALSE beliefs.
    active = [r for r in results if r.truth_state != "false"]

    if not active:
        return ""

    if llm_fn is None:
        # No-LLM mode: return top claim directly.
        return active[0].claim

    # LLM synthesis mode.
    belief_lines = []
    for i, r in enumerate(active[:max_beliefs], 1):
        conf = f" (confidence: {r.confidence:.2f})" if r.confidence is not None else ""
        state = f" [{r.truth_state}]" if r.truth_state else ""
        belief_lines.append(f"{i}. {r.claim}{conf}{state}")

    beliefs_text = "\n".join(belief_lines)
    prompt = (
        "Based on the following stored beliefs, answer the question concisely.\n\n"
        f"Question: {question}\n\n"
        f"Relevant beliefs:\n{beliefs_text}\n\n"
        "Answer (be concise, use only information from the beliefs above):"
    )

    return llm_fn(prompt).strip()
