"""Load HotpotQA dataset from HuggingFace or local JSON.

HotpotQA format:
- Each example has a question, answer, supporting facts, and context paragraphs.
- Context is a list of [title, sentences] pairs.
- We focus on the "distractor" setting (10 paragraphs, 2 relevant).
"""

from __future__ import annotations

import json
from pathlib import Path

from mnemebrain_benchmark.external_evals.base import Scenario


def load_hotpotqa(
    path: str,
    limit: int | None = None,
    difficulty: str | None = None,
) -> list[Scenario]:
    """Load HotpotQA scenarios from local JSON.

    Args:
        path: Path to JSON file.
        limit: Maximum scenarios to load.
        difficulty: Filter by "easy", "medium", or "hard".

    Returns:
        List of Scenario objects.
    """
    path_obj = Path(path)
    if path_obj.suffix == ".jsonl":
        raw_items = _load_jsonl(path_obj)
    else:
        with open(path_obj) as f:
            data = json.load(f)
        raw_items = data if isinstance(data, list) else [data]

    scenarios = []
    for item in raw_items:
        if difficulty and item.get("level") != difficulty:
            continue

        scenario = _parse_scenario(item)
        if scenario:
            scenarios.append(scenario)
        if limit and len(scenarios) >= limit:
            break

    return scenarios


def _load_jsonl(fp: Path) -> list[dict]:
    items = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _parse_scenario(item: dict) -> Scenario | None:
    """Parse a HotpotQA item into a Scenario."""
    scenario_id = item.get("_id") or item.get("id") or ""
    if not scenario_id:
        return None

    question = item.get("question", "")
    answer = item.get("answer", "")
    q_type = item.get("type", "bridge")
    level = item.get("level", "medium")

    # Parse context paragraphs into history entries.
    history: list[dict] = []
    context = item.get("context", {})

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        for i, title in enumerate(titles):
            sents = sentences_list[i] if i < len(sentences_list) else []
            paragraph = " ".join(sents)
            history.append({
                "role": "document",
                "content": paragraph,
                "title": title,
                "source_ref": f"hotpotqa:{title}",
            })
    elif isinstance(context, list):
        # Alternative format: list of [title, sentences] pairs.
        for ctx in context:
            if isinstance(ctx, list) and len(ctx) >= 2:
                title = ctx[0]
                sents = ctx[1] if isinstance(ctx[1], list) else [ctx[1]]
                paragraph = " ".join(sents)
                history.append({
                    "role": "document",
                    "content": paragraph,
                    "title": title,
                    "source_ref": f"hotpotqa:{title}",
                })

    # Parse supporting facts for metadata.
    supporting_facts = item.get("supporting_facts", {})

    questions = [{
        "question": question,
        "gold_answer": answer,
        "type": q_type,
    }]

    return Scenario(
        scenario_id=str(scenario_id),
        subset=f"hotpotqa_{q_type}",
        history=history,
        questions=questions,
        metadata={
            "level": level,
            "type": q_type,
            "supporting_facts": supporting_facts,
        },
    )
