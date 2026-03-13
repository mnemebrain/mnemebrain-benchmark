"""CLI runner for LongMemEval benchmark.

Usage:
    python -m mnemebrain_benchmark.external_evals longmemeval \
        --data-path /path/to/longmemeval.json \
        --subset knowledge_update \
        --system lite \
        [--llm-extract] [--llm-answer] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from mnemebrain_benchmark.external_evals.longmemeval.adapter import LongMemEvalAdapter
from mnemebrain_benchmark.external_evals.scorer import (
    BenchmarkReport,
    QuestionScore,
    SubsetScore,
    exact_match,
    token_f1,
)


def _create_system(system_type: str, embedding_provider: object | None = None) -> object:
    """Create a memory system adapter by type.

    Args:
        system_type: One of "lite", "full".
        embedding_provider: Optional embedding provider override.

    Returns:
        A MemorySystem-compatible object.
    """
    if system_type == "lite":
        from mnemebrain_benchmark.adapters.mnemebrain_lite_adapter import MnemeBrainLiteAdapter
        from mnemebrain_benchmark.providers import SentenceTransformerProvider

        embedder = embedding_provider or SentenceTransformerProvider()
        return MnemeBrainLiteAdapter(embedder=embedder)
    elif system_type == "full":
        from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter

        return MnemeBrainAdapter()
    else:
        raise ValueError(f"Unknown system type: {system_type!r}. Use 'lite' or 'full'.")


def run_longmemeval(
    data_path: str,
    *,
    system_type: str = "lite",
    subset: str | None = None,
    limit: int | None = None,
    llm_extract: bool = False,
    llm_answer: bool = False,
    llm_fn: object | None = None,
    embedding_provider: object | None = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run LongMemEval benchmark and return a report.

    Args:
        data_path: Path to LongMemEval dataset JSON/JSONL.
        system_type: "lite" or "full".
        subset: Optional subset filter (e.g. "knowledge_update").
        limit: Max scenarios to process.
        llm_extract: Use LLM for claim extraction.
        llm_answer: Use LLM for answer generation.
        llm_fn: LLM callable for extraction/answering.
        embedding_provider: Optional embedding provider.
        verbose: Print progress to stderr.

    Returns:
        BenchmarkReport with per-subset and overall scores.
    """
    adapter = LongMemEvalAdapter(
        use_llm_extract=llm_extract,
        llm_fn=llm_fn,
        use_llm_answer=llm_answer,
    )

    scenarios = adapter.load_dataset(data_path, subset=subset)
    if limit:
        scenarios = scenarios[:limit]

    if verbose:
        print(f"Loaded {len(scenarios)} scenarios", file=sys.stderr)

    system = _create_system(system_type, embedding_provider=embedding_provider)
    report = BenchmarkReport(
        benchmark_name="longmemeval",
        system_name=system.name(),
    )

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"  [{i + 1}/{len(scenarios)}] {scenario.scenario_id} "
                f"({scenario.subset})",
                file=sys.stderr,
            )

        # Reset system for each scenario.
        system.reset()

        # Ingest.
        adapter.ingest(system, scenario)

        # Answer each question.
        for q in scenario.questions:
            t1 = time.monotonic()
            predicted = adapter.answer(system, q)
            answer_time = time.monotonic() - t1

            gold = q.get("gold_answer", "")
            f1 = token_f1(predicted, gold)
            em = exact_match(predicted, gold)

            q_score = QuestionScore(
                question_id=f"{scenario.scenario_id}_{q.get('question', '')[:30]}",
                question=q.get("question", ""),
                gold_answer=gold,
                predicted_answer=predicted,
                f1=f1,
                em=em,
            )

            subset_name = q.get("type") or scenario.subset
            if subset_name not in report.subsets:
                report.subsets[subset_name] = SubsetScore(subset=subset_name)
            report.subsets[subset_name].question_scores.append(q_score)

            if verbose:
                print(
                    f"    Q: {q.get('question', '')[:50]}... "
                    f"F1={f1:.3f} EM={em:.0f} "
                    f"({answer_time:.2f}s)",
                    file=sys.stderr,
                )

    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against MnemeBrain",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to LongMemEval dataset (JSON/JSONL file or directory)",
    )
    parser.add_argument(
        "--system",
        choices=["lite", "full"],
        default="lite",
        help="System to benchmark (default: lite)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Filter to a specific subset (e.g. knowledge_update)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of scenarios to process",
    )
    parser.add_argument(
        "--llm-extract",
        action="store_true",
        help="Use LLM for claim extraction (requires API key)",
    )
    parser.add_argument(
        "--llm-answer",
        action="store_true",
        help="Use LLM for answer generation",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write detailed results to JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )

    args = parser.parse_args(argv)

    report = run_longmemeval(
        data_path=args.data_path,
        system_type=args.system,
        subset=args.subset,
        limit=args.limit,
        llm_extract=args.llm_extract,
        llm_answer=args.llm_answer,
        verbose=args.verbose,
    )

    print(report.summary())

    if args.output_json:
        results = {
            "benchmark": report.benchmark_name,
            "system": report.system_name,
            "overall_f1": report.overall_f1,
            "overall_em": report.overall_em,
            "total_questions": report.total_questions,
            "subsets": {},
        }
        for name, subset_score in report.subsets.items():
            results["subsets"][name] = {
                "count": subset_score.count,
                "avg_f1": subset_score.avg_f1,
                "avg_em": subset_score.avg_em,
                "questions": [
                    {
                        "id": q.question_id,
                        "question": q.question,
                        "gold": q.gold_answer,
                        "predicted": q.predicted_answer,
                        "f1": q.f1,
                        "em": q.em,
                    }
                    for q in subset_score.question_scores
                ],
            }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results written to {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
