"""CLI entry point for the system benchmark."""

from __future__ import annotations

import argparse
import sys

from mnemebrain_benchmark.adapter_factory import build_adapters
from mnemebrain_benchmark.providers import EMBEDDER_CHOICES
from mnemebrain_benchmark.scenarios.loader import load_scenarios
from mnemebrain_benchmark.system_report import export_json, format_scorecard
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner


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
        help="Embedding provider (default: auto-detect)",
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

    adapters = build_adapters(args.adapter, args.embedder, args.embedder_model)
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
