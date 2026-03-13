"""Belief Maintenance Benchmark (BMB) -- CLI entry point.

Runs 48 tasks across 8 categories against memory system adapters.

Usage:
    mnemebrain-bmb
    mnemebrain-bmb --adapter mnemebrain
    mnemebrain-bmb --embedder openai
    mnemebrain-bmb --embedder ollama --embedder-model nomic-embed-text
    python -m mnemebrain_benchmark.bmb_cli
"""

from __future__ import annotations

import argparse
import os
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.providers import EMBEDDER_CHOICES, build_embedder
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


ALL_ADAPTERS = [
    "mnemebrain",
    "mnemebrain_lite",
    "naive_baseline",
    "langchain_buffer",
    "rag_baseline",
    "structured_memory",
    "mem0",
    "openai_rag",
]


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

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline

            adapters.append(NaiveBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print(
                    "naive_baseline requires sentence-transformers: "
                    "pip install mnemebrain-benchmark[embeddings]"
                )
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "langchain_buffer":
        from mnemebrain_benchmark.adapters.langchain_buffer import LangChainBufferBaseline

        adapters.append(LangChainBufferBaseline())

    if adapter_filter is None or adapter_filter == "rag_baseline":
        try:
            from mnemebrain_benchmark.adapters.rag_baseline import RAGBaseline

            adapters.append(RAGBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "rag_baseline":
                print(
                    "rag_baseline requires sentence-transformers: "
                    "pip install mnemebrain-benchmark[embeddings]"
                )
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "structured_memory":
        try:
            from mnemebrain_benchmark.adapters.structured_memory import StructuredMemoryBaseline

            adapters.append(StructuredMemoryBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "structured_memory":
                print(
                    "structured_memory requires sentence-transformers:"
                    " pip install mnemebrain-benchmark[embeddings]"
                )
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "mem0":
        try:
            from mnemebrain_benchmark.adapters.mem0_adapter import Mem0Adapter

            adapters.append(Mem0Adapter())
        except (ImportError, ValueError) as e:
            if adapter_filter == "mem0":
                print(f"mem0 adapter error: {e}")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "openai_rag":
        try:
            from mnemebrain_benchmark.adapters.openai_rag_adapter import OpenAIRAGAdapter

            adapters.append(OpenAIRAGAdapter())
        except (ImportError, ValueError) as e:
            if adapter_filter == "openai_rag":
                print(f"openai_rag adapter error: {e}")
                sys.exit(1)

    return adapters


def _print_bmb_chart(results: dict[str, list]) -> None:
    from mnemebrain_benchmark.scoring import aggregate_by_category

    print("\n" + "=" * 60)
    print("  BELIEF MAINTENANCE BENCHMARK (BMB)")
    print("  48 tasks | 8 categories | ~100 checks")
    print("=" * 60)

    for system_name, scores in results.items():
        cats = aggregate_by_category(scores)
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if scored:
            avg = sum(c.score or 0.0 for c in scored) / len(scored)
            pct = int(avg * 100)
            filled = "\u2588" * (pct // 5)
            print(f"  {system_name:<20} {filled} {pct}%")
        else:
            print(f"  {system_name:<20} N/A")

    print("=" * 60)
    print()


def run_bmb(
    adapter_filter: str | None = None,
    category: str | None = None,
    scenario_name: str | None = None,
    output: str = "bmb_report.json",
    embedder_name: str | None = None,
    embedder_model: str | None = None,
) -> dict[str, list]:
    scenarios = load_bmb_scenarios()

    if category:
        scenarios = [s for s in scenarios if s.category == category]
    if scenario_name:
        scenarios = [s for s in scenarios if s.name == scenario_name]

    if not scenarios:
        print("No matching BMB scenarios found.")
        sys.exit(1)

    adapters = _build_adapters(adapter_filter, embedder_name, embedder_model)
    if not adapters:
        print("No matching adapters found.")
        sys.exit(1)

    print(f"BMB: Running {len(scenarios)} scenarios against {len(adapters)} adapter(s)...\n")

    runner = SystemBenchmarkRunner()
    results = runner.run_all(adapters, scenarios)

    print(format_scorecard(results))
    _print_bmb_chart(results)
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
        help="Embedding provider (default: auto-detect — openai if OPENAI_API_KEY set, else sentence_transformers)",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Model name override for the embedding provider",
    )

    args = parser.parse_args(argv)
    run_bmb(
        adapter_filter=args.adapter,
        category=args.category,
        scenario_name=args.scenario,
        output=args.output,
        embedder_name=args.embedder,
        embedder_model=args.embedder_model,
    )


if __name__ == "__main__":
    main()
