"""Load LongMemEval dataset from HuggingFace or local JSON.

LongMemEval format (from the official release):
- Each example has conversation sessions with temporal markers.
- Questions are categorized into subsets: knowledge_update, temporal,
  information_extraction, multi_session, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemebrain_benchmark.external_evals.base import Scenario

# LongMemEval subset categories we target.
SUPPORTED_SUBSETS = frozenset({
    "knowledge_update",
    "temporal_reasoning",
    "information_extraction",
    "multi_session",
    "single_session",
})


def load_longmemeval(
    path: str,
    subset: str | None = None,
    limit: int | None = None,
) -> list[Scenario]:
    """Load LongMemEval scenarios from a local JSON file or directory.

    Args:
        path: Path to JSON/JSONL file or directory containing them.
        subset: If provided, only load scenarios in this subset category.
        limit: Maximum number of scenarios to load.

    Returns:
        List of Scenario objects.
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        raw_items: list[dict] = []
        for fp in sorted(path_obj.glob("*.json")):
            raw_items.extend(_load_json_file(fp))
        for fp in sorted(path_obj.glob("*.jsonl")):
            raw_items.extend(_load_jsonl_file(fp))
    elif path_obj.suffix == ".jsonl":
        raw_items = _load_jsonl_file(path_obj)
    else:
        raw_items = _load_json_file(path_obj)

    scenarios = []
    for item in raw_items:
        scenario = _parse_scenario(item)
        if scenario is None:
            continue
        if subset and scenario.subset != subset:
            continue
        scenarios.append(scenario)
        if limit and len(scenarios) >= limit:
            break

    return scenarios


def _load_json_file(fp: Path) -> list[dict]:
    """Load a JSON file (list of objects or single object)."""
    with open(fp) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def _load_jsonl_file(fp: Path) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    items = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _parse_scenario(item: dict) -> Scenario | None:
    """Parse a single LongMemEval item into a Scenario."""
    scenario_id = (
        item.get("id") or item.get("example_id") or item.get("question_id") or ""
    )
    if not scenario_id:
        return None

    subset = (
        item.get("category")
        or item.get("question_type")
        or item.get("type")
        or item.get("subset")
        or "unknown"
    )

    history: list[dict] = []
    sessions = (
        item.get("sessions")
        or item.get("haystack_sessions")
        or item.get("conversation")
        or []
    )

    if isinstance(sessions, list) and sessions:
        if isinstance(sessions[0], dict) and "turns" in sessions[0]:
            # Format: [{session_id, timestamp, turns: [{role, content}]}]
            for sess in sessions:
                session_id = sess.get("session_id", "")
                timestamp = sess.get("timestamp", "")
                for turn in sess.get("turns", []):
                    history.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                        "timestamp": timestamp,
                        "session": session_id,
                    })
        elif isinstance(sessions[0], list):
            # Format: [[{role, content}, ...], ...] (haystack_sessions)
            haystack_dates = item.get("haystack_dates", [])
            for sess_idx, sess_turns in enumerate(sessions):
                timestamp = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
                for turn in sess_turns:
                    if isinstance(turn, dict):
                        history.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                            "timestamp": timestamp,
                            "session": str(sess_idx),
                        })
        elif isinstance(sessions[0], dict) and "role" in sessions[0]:
            # Format: [{role, content, timestamp?, session?}]
            for turn in sessions:
                history.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "timestamp": turn.get("timestamp", ""),
                    "session": turn.get("session", ""),
                })
        elif isinstance(sessions[0], str):
            for i, text in enumerate(sessions):
                history.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": text,
                })

    questions: list[dict] = []
    raw_questions = item.get("questions") or item.get("qa_pairs") or []

    if isinstance(raw_questions, list):
        for q in raw_questions:
            if isinstance(q, dict):
                questions.append({
                    "question": q.get("question") or q.get("query") or "",
                    "gold_answer": q.get("answer") or q.get("gold_answer") or "",
                    "type": q.get("type") or subset,
                })
    elif isinstance(raw_questions, dict):
        questions.append({
            "question": raw_questions.get("question") or "",
            "gold_answer": raw_questions.get("answer") or "",
            "type": raw_questions.get("type") or subset,
        })

    if not questions and item.get("question"):
        questions.append({
            "question": item["question"],
            "gold_answer": item.get("answer") or item.get("gold_answer") or "",
            "type": subset,
        })

    if not history and not questions:
        return None

    return Scenario(
        scenario_id=str(scenario_id),
        subset=subset,
        history=history,
        questions=questions,
        metadata={k: v for k, v in item.items() if k not in {
            "id", "example_id", "question_id", "category", "type",
            "question_type", "subset", "sessions", "haystack_sessions",
            "conversation", "questions", "qa_pairs", "question", "answer",
            "gold_answer",
        }},
    )
