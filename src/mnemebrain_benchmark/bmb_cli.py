"""Belief Maintenance Benchmark (BMB) -- CLI entry point.

Runs 48 tasks across 8 categories against memory system adapters.
Optionally includes external benchmarks (LongMemEval, HotpotQA).

Usage:
    mnemebrain-bmb
    mnemebrain-bmb --adapter mnemebrain
    mnemebrain-bmb --include-external --data-path /path/to/data.json
    mnemebrain-bmb --external-only --data-path /path/to/data.json
    mnemebrain-bmb --embedder openai
    python -m mnemebrain_benchmark.bmb_cli
"""

from __future__ import annotations

import argparse
import sys

from mnemebrain_benchmark.adapter_factory import ALL_ADAPTERS, build_adapters
from mnemebrain_benchmark.providers import EMBEDDER_CHOICES
from mnemebrain_benchmark.scenarios.loader import load_bmb_scenarios
from mnemebrain_benchmark.system_report import export_json, format_scorecard
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner

BMB_CATEGORIES = [
    "contradiction",
    "belief_revision",
    "evidence_tracking",
    "temporal",
    "counterfactual",
    "consolidation",
    "multihop_retrieval",
    "pattern_separation",
]

EXTERNAL_BENCHMARKS = ["longmemeval", "hotpotqa", "all"]


def _print_bmb_chart(results: dict[str, list]) -> None:
    from mnemebrain_benchmark.scoring import aggregate_by_category

    total_cats = len(BMB_CATEGORIES)

    print("\n" + "=" * 64)
    print("  BELIEF MAINTENANCE BENCHMARK (BMB)")
    print("  48 tasks | 8 categories | ~100 checks")
    print("=" * 64)

    for system_name, scores in results.items():
        cats = aggregate_by_category(scores)
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if scored:
            avg = sum(c.score or 0.0 for c in scored) / len(scored)
            weighted = avg * len(scored) / total_cats if total_cats > 0 else 0.0
            w_pct = int(weighted * 100)
            filled = "\u2588" * (w_pct // 5)
            coverage = f"[{len(scored)}/{total_cats}]"
            print(f"  {system_name:<20} {filled} {w_pct}% {coverage}")
        else:
            print(f"  {system_name:<20} N/A")

    print("-" * 64)
    print("  Scores are coverage-weighted: score x (categories_attempted / total)")
    print("=" * 64)
    print()


def _run_external_benchmarks(
    adapters,
    *,
    data_path: str,
    external_benchmark: str = "all",
    external_limit: int | None = None,
    verbose: bool = False,
) -> list:
    """Run external benchmarks against the given adapters and return reports."""
    from mnemebrain_benchmark.external_evals.scorer import BenchmarkReport

    reports: list[BenchmarkReport] = []

    for adapter in adapters:
        if external_benchmark in ("longmemeval", "all"):
            report = _run_longmemeval_for_adapter(
                adapter, data_path=data_path, limit=external_limit, verbose=verbose,
            )
            reports.append(report)

        if external_benchmark in ("hotpotqa", "all"):
            report = _run_hotpotqa_for_adapter(
                adapter, data_path=data_path, limit=external_limit, verbose=verbose,
            )
            reports.append(report)

    return reports


def _run_longmemeval_for_adapter(adapter, *, data_path, limit, verbose):
    """Run LongMemEval against a single adapter."""
    import time

    from mnemebrain_benchmark.external_evals.longmemeval.adapter import LongMemEvalAdapter
    from mnemebrain_benchmark.external_evals.scorer import (
        BenchmarkReport,
        QuestionScore,
        SubsetScore,
        exact_match,
        token_f1,
    )

    lme_adapter = LongMemEvalAdapter()
    scenarios = lme_adapter.load_dataset(data_path)
    if limit:
        scenarios = scenarios[:limit]

    report = BenchmarkReport(
        benchmark_name="longmemeval",
        system_name=adapter.name(),
    )

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"  [{i + 1}/{len(scenarios)}] {scenario.scenario_id} ({scenario.subset})",
                file=sys.stderr,
            )

        adapter.reset()
        lme_adapter.ingest(adapter, scenario)

        for q in scenario.questions:
            t1 = time.monotonic()
            predicted = lme_adapter.answer(adapter, q)
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

    return report


def _run_hotpotqa_for_adapter(adapter, *, data_path, limit, verbose):
    """Run HotpotQA against a single adapter."""
    import time

    from mnemebrain_benchmark.external_evals.hotpotqa.adapter import HotpotQAAdapter
    from mnemebrain_benchmark.external_evals.scorer import (
        BenchmarkReport,
        QuestionScore,
        SubsetScore,
        exact_match,
        token_f1,
    )

    hpqa_adapter = HotpotQAAdapter()
    scenarios = hpqa_adapter.load_dataset(data_path)
    if limit:
        scenarios = scenarios[:limit]

    report = BenchmarkReport(
        benchmark_name="hotpotqa",
        system_name=adapter.name(),
    )

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"  [{i + 1}/{len(scenarios)}] {scenario.scenario_id} ({scenario.subset})",
                file=sys.stderr,
            )

        adapter.reset()
        hpqa_adapter.ingest(adapter, scenario)

        for q in scenario.questions:
            t1 = time.monotonic()
            predicted = hpqa_adapter.answer(adapter, q)
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

    return report


def run_bmb(
    adapter_filter: str | None = None,
    category: str | None = None,
    scenario_name: str | None = None,
    output: str = "bmb_report.json",
    embedder_name: str | None = None,
    embedder_model: str | None = None,
    include_external: bool = False,
    external_only: bool = False,
    data_path: str | None = None,
    external_benchmark: str = "all",
    external_limit: int | None = None,
) -> dict[str, list]:
    """Run the BMB benchmark and optionally external benchmarks."""
    adapters = build_adapters(adapter_filter, embedder_name, embedder_model)
    if not adapters:
        print("No matching adapters found.")
        sys.exit(1)

    results: dict[str, list] = {}

    # Run BMB (internal) scenarios unless --external-only.
    if not external_only:
        scenarios = load_bmb_scenarios()

        if category:
            scenarios = [s for s in scenarios if s.category == category]
        if scenario_name:
            scenarios = [s for s in scenarios if s.name == scenario_name]

        if not scenarios:
            print("No matching BMB scenarios found.")
            sys.exit(1)

        print(f"BMB: Running {len(scenarios)} scenarios against {len(adapters)} adapter(s)...\n")

        runner = SystemBenchmarkRunner()
        results = runner.run_all(adapters, scenarios)

        print(format_scorecard(results))
        _print_bmb_chart(results)

    # Run external benchmarks if requested.
    if include_external or external_only:
        if not data_path:
            print("--data-path is required for external benchmarks.")
            sys.exit(1)

        print(
            f"\nExternal: Running {external_benchmark} against "
            f"{len(adapters)} adapter(s)...\n"
        )

        reports = _run_external_benchmarks(
            adapters,
            data_path=data_path,
            external_benchmark=external_benchmark,
            external_limit=external_limit,
        )

        for report in reports:
            print(report.summary())

    export_json(results, output)
    print(f"Report saved to {output}")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MnemeBrain Belief Maintenance Benchmark (BMB)")
    parser.add_argument("--adapter", type=str, default=None, choices=ALL_ADAPTERS)
    parser.add_argument("--category", type=str, default=None, choices=BMB_CATEGORIES)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="bmb_report.json")
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

    # External benchmark flags.
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Run BMB + external benchmarks",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Skip BMB, run external benchmarks only",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to external benchmark dataset (JSON/JSONL file or directory)",
    )
    parser.add_argument(
        "--external-benchmark",
        type=str,
        default="all",
        choices=EXTERNAL_BENCHMARKS,
        help="Which external benchmark to run (default: all)",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        default=None,
        help="Max number of external scenarios to process per benchmark",
    )

    args = parser.parse_args(argv)
    run_bmb(
        adapter_filter=args.adapter,
        category=args.category,
        scenario_name=args.scenario,
        output=args.output,
        embedder_name=args.embedder,
        embedder_model=args.embedder_model,
        include_external=args.include_external,
        external_only=args.external_only,
        data_path=args.data_path,
        external_benchmark=args.external_benchmark,
        external_limit=args.external_limit,
    )


if __name__ == "__main__":
    main()
