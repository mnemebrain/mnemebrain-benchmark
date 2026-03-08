"""Belief Maintenance Benchmark (BMB) -- CLI entry point.

Runs 48 tasks across 8 categories against memory system adapters.

Usage:
    mnemebrain-bmb
    mnemebrain-bmb --adapter mnemebrain
    python -m mnemebrain_benchmark.bmb_cli
"""
from __future__ import annotations

import argparse
import os
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.scenarios.loader import load_bmb_scenarios
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner
from mnemebrain_benchmark.system_report import format_scorecard, export_json

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
    "naive_baseline",
    "langchain_buffer",
    "rag_baseline",
    "structured_memory",
    "mem0",
    "openai_rag",
]


def _get_embedder():
    """Lazily create a SentenceTransformer embedding provider."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required: pip install mnemebrain-benchmark[embeddings]")

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

    return _STProvider()


def _build_adapters(adapter_filter: str | None = None) -> list[MemorySystem]:
    adapters: list[MemorySystem] = []

    embedder = None
    def _lazy_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = _get_embedder()
        return embedder

    if adapter_filter is None or adapter_filter == "mnemebrain":
        try:
            from mnemebrain_benchmark.adapters.mnemebrain_adapter import MnemeBrainAdapter
            base_url = os.environ.get("MNEMEBRAIN_URL", "http://localhost:8000")
            adapters.append(MnemeBrainAdapter(base_url=base_url))
        except ImportError:
            if adapter_filter == "mnemebrain":
                print("mnemebrain adapter requires the SDK: pip install mnemebrain-benchmark[mnemebrain]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_benchmark.adapters.naive_baseline import NaiveBaseline
            adapters.append(NaiveBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print("naive_baseline requires sentence-transformers: pip install mnemebrain-benchmark[embeddings]")
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
                print("rag_baseline requires sentence-transformers: pip install mnemebrain-benchmark[embeddings]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "structured_memory":
        try:
            from mnemebrain_benchmark.adapters.structured_memory import StructuredMemoryBaseline
            adapters.append(StructuredMemoryBaseline(_lazy_embedder()))
        except ImportError:
            if adapter_filter == "structured_memory":
                print("structured_memory requires sentence-transformers: pip install mnemebrain-benchmark[embeddings]")
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
            avg = sum(c.score for c in scored) / len(scored)
            pct = int(avg * 100)
            bar = "\u2588" * (pct // 5)
            print(f"  {system_name:<20} {bar} {pct}%")
        else:
            print(f"  {system_name:<20} N/A")

    print("=" * 60)
    print()


def run_bmb(
    adapter_filter: str | None = None,
    category: str | None = None,
    scenario_name: str | None = None,
    output: str = "bmb_report.json",
) -> dict[str, list]:
    scenarios = load_bmb_scenarios()

    if category:
        scenarios = [s for s in scenarios if s.category == category]
    if scenario_name:
        scenarios = [s for s in scenarios if s.name == scenario_name]

    if not scenarios:
        print("No matching BMB scenarios found.")
        sys.exit(1)

    adapters = _build_adapters(adapter_filter)
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
    parser = argparse.ArgumentParser(
        description="MnemeBrain Belief Maintenance Benchmark (BMB)"
    )
    parser.add_argument("--adapter", type=str, default=None, choices=ALL_ADAPTERS)
    parser.add_argument("--category", type=str, default=None, choices=BMB_CATEGORIES)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="bmb_report.json")

    args = parser.parse_args(argv)
    run_bmb(
        adapter_filter=args.adapter,
        category=args.category,
        scenario_name=args.scenario,
        output=args.output,
    )


if __name__ == "__main__":
    main()
