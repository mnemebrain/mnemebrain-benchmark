"""Extract discrete claims from raw text for belief ingestion.

Two modes:
- Sentence splitting (default): deterministic, no LLM, reproducible.
- LLM extraction (--llm-extract): uses an LLM for structured claim extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedClaim:
    """A single extracted claim with provenance."""

    text: str
    source_ref: str = ""
    timestamp: str | None = None
    session: str | None = None
    confidence: float = 1.0


# Sentence-ending pattern: period/question/exclamation followed by space or end.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Filter out very short fragments that aren't real claims.
_MIN_CLAIM_LENGTH = 10


def extract_claims_sentence(
    text: str,
    source_ref: str = "",
    timestamp: str | None = None,
    session: str | None = None,
) -> list[ExtractedClaim]:
    """Split text into sentence-level claims (deterministic, no LLM).

    Args:
        text: Raw text to split into claims.
        source_ref: Source reference for provenance tracking.
        timestamp: Optional timestamp string.
        session: Optional session identifier.

    Returns:
        List of ExtractedClaim objects, one per sentence.
    """
    if not text or not text.strip():
        return []

    sentences = _SENT_SPLIT.split(text.strip())
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < _MIN_CLAIM_LENGTH:
            continue
        claims.append(
            ExtractedClaim(
                text=sent,
                source_ref=source_ref,
                timestamp=timestamp,
                session=session,
            )
        )
    return claims


def extract_claims_llm(
    text: str,
    llm_fn: LLMFunction | None = None,
    source_ref: str = "",
    timestamp: str | None = None,
    session: str | None = None,
) -> list[ExtractedClaim]:
    """Extract structured claims using an LLM (better accuracy, non-deterministic).

    Args:
        text: Raw text to extract claims from.
        llm_fn: Callable that takes a prompt string and returns a response string.
            Expected to return one claim per line.
        source_ref: Source reference for provenance tracking.
        timestamp: Optional timestamp string.
        session: Optional session identifier.

    Returns:
        List of ExtractedClaim objects.

    Raises:
        ValueError: If llm_fn is not provided.
    """
    if llm_fn is None:
        raise ValueError(
            "llm_fn is required for LLM-based claim extraction. "
            "Use extract_claims_sentence() for deterministic extraction."
        )

    prompt = (
        "Extract discrete factual claims from the following text. "
        "Return one claim per line. Each claim should be a standalone "
        "statement that can be independently true or false. "
        "Do not include opinions, greetings, or filler.\n\n"
        f"Text:\n{text}\n\n"
        "Claims:"
    )

    response = llm_fn(prompt)
    claims = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Strip leading numbering like "1. " or "- "
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        if len(line) < _MIN_CLAIM_LENGTH:
            continue
        claims.append(
            ExtractedClaim(
                text=line,
                source_ref=source_ref,
                timestamp=timestamp,
                session=session,
            )
        )
    return claims


# Type alias for LLM function signature.
LLMFunction = object  # Callable[[str], str] at runtime; avoid typing import overhead.


def extract_claims(
    text: str,
    *,
    use_llm: bool = False,
    llm_fn: object | None = None,
    source_ref: str = "",
    timestamp: str | None = None,
    session: str | None = None,
) -> list[ExtractedClaim]:
    """Unified entry point for claim extraction.

    Args:
        text: Raw text to extract claims from.
        use_llm: If True, use LLM-based extraction (requires llm_fn).
        llm_fn: LLM callable for LLM mode.
        source_ref: Source reference for provenance.
        timestamp: Optional timestamp.
        session: Optional session identifier.

    Returns:
        List of ExtractedClaim objects.
    """
    if use_llm:
        return extract_claims_llm(
            text,
            llm_fn=llm_fn,
            source_ref=source_ref,
            timestamp=timestamp,
            session=session,
        )
    return extract_claims_sentence(
        text,
        source_ref=source_ref,
        timestamp=timestamp,
        session=session,
    )
