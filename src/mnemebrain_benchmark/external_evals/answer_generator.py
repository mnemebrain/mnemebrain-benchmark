"""Generate natural-language answers from belief query results.

External benchmarks expect free-form text answers, but MnemeBrain returns
structured QueryResult objects. This module bridges the gap.
"""

from __future__ import annotations

import re

from mnemebrain_benchmark.interface import QueryResult

# Common English stopwords that carry no answer content.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "so", "yet",
        "both", "either", "neither", "not", "no", "nor", "only", "own", "same",
        "than", "too", "very", "just", "me", "my", "myself", "we", "our",
        "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
        "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "i", "s", "t", "don", "isn", "aren", "wasn", "weren", "hasn",
        "hadn", "doesn", "didn", "won", "wouldn", "shan", "couldn", "mustn",
        "mightn", "needn",
    }
)

# Question words to strip when building the exclusion set.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what", "which", "who", "whom", "whose", "where", "when", "why",
        "how", "is", "are", "was", "were", "did", "do", "does", "has",
        "have", "had", "can", "could", "will", "would", "should", "shall",
        "tell", "me", "you", "describe", "explain", "name", "list",
    }
)

# Regex for a numeric span (digits, commas, decimals, optional units).
_NUMERIC_RE = re.compile(
    r"\b\d[\d,\.]*(?:\s*(?:years?|months?|days?|hours?|minutes?|seconds?"
    r"|km|miles?|meters?|feet|foot|inches?|lbs?|kg|pounds?|%))?\b",
    re.IGNORECASE,
)

# Patterns that suggest a location/place (capitalised multi-word or known
# location prepositions followed by a noun phrase).
_LOCATION_RE = re.compile(
    r"(?:in|at|near|from|to)\s+([A-Z][A-Za-z\s,]+?)(?:[,.]|$)"
    r"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
)


def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z]+", text.lower())


def _question_type(question: str) -> str:
    """Classify the question by its leading wh-word or auxiliary.

    Returns one of: "how_many", "where", "when", "who", "what", "other".
    """
    lower = question.lower().strip()
    if re.match(r"how\s+(many|much|long|far|old|often|tall|wide|deep|big|large|small)", lower):
        return "how_many"
    if lower.startswith("where"):
        return "where"
    if lower.startswith("when"):
        return "when"
    if lower.startswith("who"):
        return "who"
    if lower.startswith("what"):
        return "what"
    return "other"


def _longest_run(tokens: list[str], claim: str, excluded: frozenset[str]) -> str:
    """Find the longest contiguous run of non-excluded tokens and return
    the corresponding substring from the original *claim* text.

    Args:
        tokens: Lowercase tokens from *claim*.
        claim: Original (cased) claim string.
        excluded: Tokens to treat as word-boundaries (skip over).

    Returns:
        The best contiguous substring from *claim*, or the full *claim*
        if no run is found.
    """
    # Build a list of (start_char, end_char, token) for each token in claim.
    token_spans: list[tuple[int, int, str]] = []
    pos = 0
    for tok in re.finditer(r"[A-Za-z0-9']+", claim):
        lowered = tok.group().lower()
        token_spans.append((tok.start(), tok.end(), lowered))

    if not token_spans:
        return claim

    best_start = 0
    best_end = 0
    run_start: int | None = None
    run_start_char = 0

    for i, (char_start, char_end, tok) in enumerate(token_spans):
        if tok in excluded:
            # End any ongoing run.
            if run_start is not None:
                run_end_char = token_spans[i - 1][1]
                if run_end_char - run_start_char > best_end - best_start:
                    best_start = run_start_char
                    best_end = run_end_char
            run_start = None
        else:
            if run_start is None:
                run_start = i
                run_start_char = char_start

    # Close any trailing run.
    if run_start is not None:
        run_end_char = token_spans[-1][1]
        if run_end_char - run_start_char > best_end - best_start:
            best_start = run_start_char
            best_end = run_end_char

    span = claim[best_start:best_end].strip(" ,.;:")
    return span if span else claim


def _extract_answer_span(question: str, claim: str) -> str:
    """Extract the answer-bearing substring from *claim* given *question*.

    Strategy:
    1. Build an exclusion set from question tokens and stopwords.
    2. For "how many/much/long/far" questions, prefer a numeric span.
    3. For "where" questions, prefer a location-like span.
    4. Otherwise, return the longest contiguous run of non-excluded tokens
       from *claim*.
    5. Fallback: return *claim* unchanged if extraction produces nothing.

    Args:
        question: The original question string.
        claim: A stored belief claim to extract the answer from.

    Returns:
        A contiguous substring of *claim* that best answers the question,
        or the full *claim* if extraction fails.
    """
    if not claim:
        return claim

    qtype = _question_type(question)

    # Build exclusion set: question content tokens + stopwords.
    q_tokens = frozenset(_tokenize(question)) | _QUESTION_WORDS | _STOPWORDS

    # --- Numeric preference (how many / how much / how long / how far) -------
    if qtype == "how_many":
        numeric_matches = list(_NUMERIC_RE.finditer(claim))
        if numeric_matches:
            # Return the first numeric span (most prominent quantity).
            return numeric_matches[0].group().strip()

    # --- Location preference (where) -----------------------------------------
    if qtype == "where":
        loc_match = _LOCATION_RE.search(claim)
        if loc_match:
            # group(1) = "in/at/... Location", group(2) = "Title Case Place"
            candidate = (loc_match.group(1) or loc_match.group(2) or "").strip(" ,.")
            if candidate:
                return candidate

    # --- General: longest run of non-excluded tokens --------------------------
    claim_tokens = _tokenize(claim)
    span = _longest_run(claim_tokens, claim, q_tokens)

    # Sanity: if the span is almost as long as the claim (we stripped nothing
    # useful), just return it; if it's empty fall back to full claim.
    return span if span else claim


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
    - Without llm_fn: Extracts the answer-bearing span from the top belief
      claim using :func:`_extract_answer_span` (fast, no LLM cost).

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
        # No-LLM mode: extract the answer-bearing span rather than returning
        # the full claim verbatim.  This improves token-F1 on benchmarks like
        # LongMemEval where gold answers are short (e.g. "Business Administration"
        # vs a full sentence containing that phrase).
        return _extract_answer_span(question, active[0].claim)

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
