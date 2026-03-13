"""Shared adapter factory for all benchmark CLIs.

Centralizes adapter instantiation so bmb_cli, system_cli, external_evals,
and task_evals can all share the same adapter-building logic.
"""

from __future__ import annotations

import os
import sys

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.providers import build_embedder

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


def build_adapters(
    adapter_filter: str | None = None,
    embedder_name: str | None = None,
    embedder_model: str | None = None,
) -> list[MemorySystem]:
    """Build memory system adapters for benchmarking.

    Args:
        adapter_filter: If set, build only this adapter (must be in ALL_ADAPTERS).
        embedder_name: Embedding provider name (see providers.EMBEDDER_CHOICES).
        embedder_model: Model name override for the embedding provider.

    Returns:
        List of instantiated MemorySystem adapters.
    """
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
