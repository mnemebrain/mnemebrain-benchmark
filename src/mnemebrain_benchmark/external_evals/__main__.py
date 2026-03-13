"""Unified CLI for external benchmark runs.

Usage:
    python -m mnemebrain_benchmark.external_evals longmemeval --data-path ... [options]
    python -m mnemebrain_benchmark.external_evals hotpotqa --data-path ... [options]
    python -m mnemebrain_benchmark.external_evals longmemeval --data-path ... --system structured_memory
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from mnemebrain_benchmark.adapter_factory import ALL_ADAPTERS, build_adapters
from mnemebrain_benchmark.external_evals.scorer import (
    BenchmarkReport,
    QuestionScore,
    SubsetScore,
    exact_match,
    token_f1,
)
from mnemebrain_benchmark.providers import EMBEDDER_CHOICES


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all benchmark subcommands."""
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to dataset (JSON/JSONL file or directory)",
    )
    parser.add_argument(
        "--system",
        choices=ALL_ADAPTERS,
        default=None,
        help="Adapter to benchmark (default: all available)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        choices=EMBEDDER_CHOICES,
        help="Embedding provider (default: auto-detect)",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Model name override for the embedding provider",
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
        help="Use LLM for claim extraction",
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


def _write_json_report(report: BenchmarkReport, output_path: str) -> None:
    """Write a BenchmarkReport to JSON."""
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
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results written to {output_path}", file=sys.stderr)


def _build_systems(args: argparse.Namespace) -> list:
    """Build memory system(s) from CLI args, using the shared adapter factory."""
    adapters = build_adapters(args.system, args.embedder, args.embedder_model)
    if not adapters:
        print("No matching adapters found.")
        sys.exit(1)
    return adapters


def _output_path_for(base_path: str, system_name: str, count: int) -> str:
    """Generate output path, appending system name when running multiple."""
    if count <= 1:
        return base_path
    if "." in base_path:
        base, ext = base_path.rsplit(".", 1)
        return f"{base}_{system_name}.{ext}"
    return f"{base_path}_{system_name}.json"


def _run_longmemeval(args: argparse.Namespace) -> None:
    """Run LongMemEval benchmark."""
    from mnemebrain_benchmark.external_evals.longmemeval.run import run_longmemeval

    systems = _build_systems(args)

    for sys_adapter in systems:
        report = run_longmemeval(
            data_path=args.data_path,
            system=sys_adapter,
            subset=args.subset,
            limit=args.limit,
            llm_extract=args.llm_extract,
            llm_answer=args.llm_answer,
            verbose=args.verbose,
        )
        print(report.summary())
        if args.output_json:
            output = _output_path_for(args.output_json, sys_adapter.name(), len(systems))
            _write_json_report(report, output)


def _run_hotpotqa(args: argparse.Namespace) -> None:
    """Run HotpotQA benchmark."""
    from mnemebrain_benchmark.external_evals.hotpotqa.adapter import HotpotQAAdapter

    systems = _build_systems(args)

    for sys_adapter in systems:
        adapter = HotpotQAAdapter(
            use_llm_extract=args.llm_extract,
            use_llm_answer=args.llm_answer,
        )

        scenarios = adapter.load_dataset(args.data_path)
        if args.limit:
            scenarios = scenarios[:args.limit]

        if args.verbose:
            print(f"Loaded {len(scenarios)} scenarios", file=sys.stderr)

        report = BenchmarkReport(
            benchmark_name="hotpotqa",
            system_name=sys_adapter.name(),
        )

        for i, scenario in enumerate(scenarios):
            if args.verbose:
                print(
                    f"  [{i + 1}/{len(scenarios)}] {scenario.scenario_id} "
                    f"({scenario.subset})",
                    file=sys.stderr,
                )

            sys_adapter.reset()
            adapter.ingest(sys_adapter, scenario)

            for q in scenario.questions:
                t1 = time.monotonic()
                predicted = adapter.answer(sys_adapter, q)
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

                if args.verbose:
                    print(
                        f"    Q: {q.get('question', '')[:50]}... "
                        f"F1={f1:.3f} EM={em:.0f} "
                        f"({answer_time:.2f}s)",
                        file=sys.stderr,
                    )

        print(report.summary())
        if args.output_json:
            output = _output_path_for(args.output_json, sys_adapter.name(), len(systems))
            _write_json_report(report, output)


def main(argv: list[str] | None = None) -> None:
    """Unified CLI entry point for external benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run external benchmarks (LongMemEval, HotpotQA) against MnemeBrain",
    )
    subparsers = parser.add_subparsers(dest="benchmark", help="Benchmark to run")

    # LongMemEval subcommand.
    lme_parser = subparsers.add_parser("longmemeval", help="Run LongMemEval benchmark")
    _add_common_args(lme_parser)
    lme_parser.add_argument(
        "--subset",
        default=None,
        help="Filter to a specific subset (e.g. knowledge_update)",
    )

    # HotpotQA subcommand.
    hpqa_parser = subparsers.add_parser("hotpotqa", help="Run HotpotQA benchmark")
    _add_common_args(hpqa_parser)
    hpqa_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level",
    )

    args = parser.parse_args(argv)

    if args.benchmark is None:
        parser.print_help()
        sys.exit(1)

    if args.benchmark == "longmemeval":
        _run_longmemeval(args)
    elif args.benchmark == "hotpotqa":
        _run_hotpotqa(args)


if __name__ == "__main__":
    main()
