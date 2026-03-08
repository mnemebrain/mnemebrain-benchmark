"""Benchmark orchestrator and reporting."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mnemebrain_benchmark.dataset import BenchmarkDataset
from mnemebrain_benchmark.metrics import (
    BenchmarkMetrics,
    PairResult,
    compute_metrics,
    evaluate_pair,
)
from mnemebrain_benchmark.protocols import EmbeddingProvider

# Provider registry: (provider_type, model_name)
PROVIDER_CONFIGS: list[tuple[str, str]] = [
    ("sentence_transformers", "all-MiniLM-L6-v2"),
    ("sentence_transformers", "all-mpnet-base-v2"),
    ("sentence_transformers", "BAAI/bge-small-en-v1.5"),
    ("openai", "text-embedding-3-small"),
    ("openai", "text-embedding-3-large"),
]


def _create_provider(provider_type: str, model_name: str) -> EmbeddingProvider:
    """Instantiate a provider by type and model name."""
    if provider_type == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required: pip install mnemebrain-benchmark[embeddings]"
            ) from exc

        class _STProvider:
            def __init__(self, model: str) -> None:
                self._model = SentenceTransformer(model)

            def embed(self, text: str) -> list[float]:
                return self._model.encode(text).tolist()

            def similarity(self, a: list[float], b: list[float]) -> float:
                import numpy as np
                a_arr, b_arr = np.array(a), np.array(b)
                dot = np.dot(a_arr, b_arr)
                norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
                return float(dot / norm) if norm > 0 else 0.0

        return _STProvider(model_name)

    if provider_type == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai required: pip install mnemebrain-benchmark[openai]") from exc

        import os
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required")

        class _OpenAIProvider:
            def __init__(self, model: str) -> None:
                self._client = OpenAI(api_key=key)
                self._model = model

            def embed(self, text: str) -> list[float]:
                resp = self._client.embeddings.create(input=[text], model=self._model)
                return resp.data[0].embedding

            def similarity(self, a: list[float], b: list[float]) -> float:
                import numpy as np
                a_arr, b_arr = np.array(a), np.array(b)
                dot = np.dot(a_arr, b_arr)
                norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
                return float(dot / norm) if norm > 0 else 0.0

        return _OpenAIProvider(model_name)

    raise ValueError(f"Unknown provider type: {provider_type}")


def _print_metrics(name: str, metrics: BenchmarkMetrics) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print("  Classification (threshold-based):")
    print(
        f"    Precision: {metrics.precision:.3f}"
        f"  Recall: {metrics.recall:.3f}"
        f"  F1: {metrics.f1:.3f}"
    )
    print(
        f"    Accuracy:  {metrics.accuracy:.3f}"
        f"  (TP={metrics.tp} FP={metrics.fp}"
        f" TN={metrics.tn} FN={metrics.fn})"
    )
    print("  Similarity distribution:")
    print(
        f"    Same pairs:      mean={metrics.mean_sim_same:.4f}"
        f"  std={metrics.std_sim_same:.4f}"
    )
    print(
        f"    Different pairs: mean={metrics.mean_sim_different:.4f}"
        f"  std={metrics.std_sim_different:.4f}"
    )
    print(f"    Separation gap:  {metrics.separation_gap:.4f}")
    print("  Latency:")
    print(
        f"    Mean: {metrics.mean_embed_ms:.1f}ms"
        f"  P95: {metrics.p95_embed_ms:.1f}ms"
        f"  P99: {metrics.p99_embed_ms:.1f}ms"
    )
    print(f"    Throughput: {metrics.throughput_pairs_per_sec:.1f} pairs/sec")
    print()


def _serialize_report(report: dict) -> dict:
    def _convert(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj
    return _convert(report)


def save_report(report: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize_report(report), f, indent=2)
    print(f"Report saved to {path}")


def run_benchmark(
    threshold: float = 0.92,
    dataset_path: str | None = None,
    provider_filter: str | None = None,
    model_filter: str | None = None,
    category: str | None = None,
    difficulty: str | None = None,
) -> dict:
    dataset = BenchmarkDataset.load(dataset_path)
    dataset = dataset.filter(category=category, difficulty=difficulty)
    print(f"Dataset: {len(dataset)} pairs")

    configs = PROVIDER_CONFIGS
    if provider_filter:
        configs = [(t, m) for t, m in configs if t == provider_filter]
    if model_filter:
        configs = [(t, m) for t, m in configs if m == model_filter]

    if not configs:
        print("No matching providers found.")
        return {"providers": {}}

    report: dict = {
        "threshold": threshold,
        "dataset_size": len(dataset),
        "providers": {},
    }

    for provider_type, model_name in configs:
        name = f"{provider_type}/{model_name}"
        print(f"\nInitializing {name}...")

        try:
            provider = _create_provider(provider_type, model_name)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        try:
            provider.embed("warmup")
        except Exception as e:
            print(f"  Skipping {name} (warmup failed): {e}")
            continue

        results: list[PairResult] = []
        for pair in dataset.pairs:
            result = evaluate_pair(provider, pair, threshold)
            results.append(result)

        metrics = compute_metrics(results, threshold)
        _print_metrics(name, metrics)

        report["providers"][name] = asdict(metrics)

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MnemeBrain Embedding Provider Benchmark"
    )
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmark_report.json")
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["sentence_transformers", "openai"],
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["fact", "preference", "inference", "prediction"],
    )
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"])

    args = parser.parse_args(argv)

    report = run_benchmark(
        threshold=args.threshold,
        dataset_path=args.dataset,
        provider_filter=args.provider,
        model_filter=args.model,
        category=args.category,
        difficulty=args.difficulty,
    )

    save_report(report, args.output)


if __name__ == "__main__":
    main()
