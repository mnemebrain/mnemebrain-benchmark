"""CLI entry point for task-level evaluations.

Usage:
    python -m mnemebrain_benchmark.task_evals
    python -m mnemebrain_benchmark.task_evals --eval preference
    python -m mnemebrain_benchmark.task_evals --eval qa --adapter naive_baseline
"""

from __future__ import annotations

import argparse
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.providers import SentenceTransformerProvider
from mnemebrain_benchmark.task_evals.long_horizon_qa import load_qa_scenarios
from mnemebrain_benchmark.task_evals.preference_tracking import load_preference_scenarios
from mnemebrain_benchmark.task_evals.runner import TaskEvalRunner, format_task_eval_table


def _build_adapters(adapter_filter: str | None = None) -> list[MemorySystem]:
    """Build adapters for task evaluation."""
    adapters: list[MemorySystem] = []
    _cached_embedder: list[SentenceTransformerProvider] = []

    def _get_embedder() -> SentenceTransformerProvider:
        if not _cached_embedder:
            try:
                _cached_embedder.append(SentenceTransformerProvider())
            except ImportError:
                print(
                    "sentence-transformers required: pip install mnemebrain-benchmark[embeddings]"
                )
                sys.exit(1)
        return _cached_embedder[0]

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline

            adapters.append(NaiveBaseline(_get_embedder()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print("naive_baseline not available")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "rag_baseline":
        try:
            from mnemebrain_benchmark.adapters.rag_baseline import RAGBaseline

            adapters.append(RAGBaseline(_get_embedder()))
        except ImportError:
            if adapter_filter == "rag_baseline":
                print("rag_baseline not available")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mnemebrain":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter

            adapters.append(MnemeBrainAdapter())
        except ImportError:
            if adapter_filter == "mnemebrain":
                print("mnemebrain adapter requires: pip install mnemebrain-benchmark[mnemebrain]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mnemebrain_lite":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_lite_adapter import (
                MnemeBrainLiteAdapter,
            )

            adapters.append(MnemeBrainLiteAdapter(_get_embedder()))
        except ImportError:
            if adapter_filter == "mnemebrain_lite":
                print("mnemebrain_lite requires: pip install mnemebrain-lite[embeddings]")
                sys.exit(1)

    return adapters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MnemeBrain Task-Level Evaluations")
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        choices=["preference", "qa"],
        help="Run only a specific evaluation (default: both)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        choices=["mnemebrain", "mnemebrain_lite", "naive_baseline", "rag_baseline"],
        help="Run only a specific adapter",
    )

    args = parser.parse_args(argv)
    adapters = _build_adapters(args.adapter)

    if not adapters:
        print("No adapters available.")
        sys.exit(1)

    runner = TaskEvalRunner()

    if args.eval is None or args.eval == "preference":
        scenarios = load_preference_scenarios()
        report = runner.run_all(adapters, scenarios)
        report.eval_name = "preference_tracking"
        print(format_task_eval_table(report))

    if args.eval is None or args.eval == "qa":
        scenarios = load_qa_scenarios()
        report = runner.run_all(adapters, scenarios)
        report.eval_name = "long_horizon_qa"
        print(format_task_eval_table(report))


if __name__ == "__main__":
    main()
