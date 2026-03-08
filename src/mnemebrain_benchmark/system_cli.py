"""CLI entry point for the system benchmark."""
from __future__ import annotations

import argparse
import os
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.scenarios.loader import load_scenarios
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner
from mnemebrain_benchmark.system_report import format_scorecard, export_json


def _build_adapters(adapter_filter: str | None = None) -> list[MemorySystem]:
    adapters: list[MemorySystem] = []

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline
            from sentence_transformers import SentenceTransformer
            import numpy as np

            class _STProvider:
                def __init__(self) -> None:
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")
                def embed(self, text: str) -> list[float]:
                    return self._model.encode(text).tolist()
                def similarity(self, a: list[float], b: list[float]) -> float:
                    a_arr, b_arr = np.array(a), np.array(b)
                    dot = np.dot(a_arr, b_arr)
                    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
                    return float(dot / norm) if norm > 0 else 0.0

            adapters.append(NaiveBaseline(_STProvider()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print("naive_baseline requires sentence-transformers: pip install mnemebrain-benchmark[embeddings]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mnemebrain":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter
            base_url = os.environ.get("MNEMEBRAIN_URL", "http://localhost:8000")
            adapters.append(MnemeBrainAdapter(base_url=base_url))
        except ImportError:
            if adapter_filter == "mnemebrain":
                print("mnemebrain adapter requires the SDK: pip install mnemebrain-benchmark[mnemebrain]")
                sys.exit(1)

    return adapters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MnemeBrain System Benchmark"
    )
    parser.add_argument("--adapter", type=str, default=None, choices=["mnemebrain", "naive_baseline"])
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["contradiction", "retraction", "decay", "dedup", "extraction", "lifecycle"],
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="system_benchmark_report.json")

    args = parser.parse_args(argv)

    scenarios = load_scenarios()

    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]

    if not scenarios:
        print("No matching scenarios found.")
        sys.exit(1)

    adapters = _build_adapters(args.adapter)
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
