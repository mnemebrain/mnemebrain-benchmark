"""CLI entry point for the system benchmark."""

from __future__ import annotations

import argparse
import os
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.providers import EMBEDDER_CHOICES, build_embedder
from mnemebrain_benchmark.scenarios.loader import load_scenarios
from mnemebrain_benchmark.system_report import export_json, format_scorecard
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner


def _build_adapters(
    adapter_filter: str | None = None,
    embedder_name: str | None = None,
    embedder_model: str | None = None,
) -> list[MemorySystem]:
    adapters: list[MemorySystem] = []

    embedder = None

    def _lazy_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = build_embedder(embedder_name, embedder_model)
        return embedder

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline

            adapters.append(NaiveBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print(
                    "naive_baseline requires sentence-transformers:"
                    " pip install mnemebrain-benchmark[embeddings]"
                )
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mnemebrain":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter

            base_url = os.environ.get("MNEMEBRAIN_URL", "http://localhost:8000")
            adapters.append(MnemeBrainAdapter(base_url=base_url))
        except ImportError:
            if adapter_filter == "mnemebrain":
                print(
                    "mnemebrain adapter requires the SDK: "
                    "pip install mnemebrain-benchmark[mnemebrain]"
                )
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mnemebrain_lite":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_lite_adapter import (
                MnemeBrainLiteAdapter,
            )

            adapters.append(MnemeBrainLiteAdapter(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "mnemebrain_lite":
                print("mnemebrain_lite adapter requires: pip install mnemebrain-lite[embeddings]")
                sys.exit(1)

    return adapters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MnemeBrain System Benchmark")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        choices=["mnemebrain", "mnemebrain_lite", "naive_baseline"],
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["contradiction", "retraction", "decay", "dedup", "extraction", "lifecycle"],
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="system_benchmark_report.json")
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        choices=EMBEDDER_CHOICES,
        help="Embedding provider (default: auto-detect — openai if OPENAI_API_KEY set, else sentence_transformers)",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Model name override for the embedding provider",
    )

    args = parser.parse_args(argv)

    scenarios = load_scenarios()

    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]

    if not scenarios:
        print("No matching scenarios found.")
        sys.exit(1)

    adapters = _build_adapters(args.adapter, args.embedder, args.embedder_model)
    if not adapters:
        print("No matching adapters found.")
        sys.exit(1)

    print(f"Running {len(scenarios)} scenarios against {len(adapters)} adapter(s)...\n")

    runner = SystemBenchmarkRunner()
    results = runner.run_all(adapters, scenarios)

    print(format_scorecard(results))
    export_json(results, args.output)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
