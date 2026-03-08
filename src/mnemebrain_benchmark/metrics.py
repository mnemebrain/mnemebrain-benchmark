"""Metric computation for the embedding benchmark."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mnemebrain_benchmark.dataset import ClaimPair
from mnemebrain_benchmark.protocols import EmbeddingProvider


@dataclass
class PairResult:
    """Result of evaluating a single claim pair."""

    pair_id: str
    similarity: float
    predicted_same: bool
    actual_same: bool
    embed_time_a_ms: float
    embed_time_b_ms: float


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from a benchmark run."""

    # Classification
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    # Distribution
    mean_sim_same: float = 0.0
    std_sim_same: float = 0.0
    mean_sim_different: float = 0.0
    std_sim_different: float = 0.0
    separation_gap: float = 0.0

    # Latency
    mean_embed_ms: float = 0.0
    p95_embed_ms: float = 0.0
    p99_embed_ms: float = 0.0
    throughput_pairs_per_sec: float = 0.0

    pair_results: list[PairResult] = field(default_factory=list)


def evaluate_pair(
    provider: EmbeddingProvider,
    pair: ClaimPair,
    threshold: float = 0.92,
) -> PairResult:
    """Embed both claims, compute similarity, classify."""
    t0 = time.perf_counter()
    vec_a = provider.embed(pair.claim_a)
    t1 = time.perf_counter()
    vec_b = provider.embed(pair.claim_b)
    t2 = time.perf_counter()

    sim = provider.similarity(vec_a, vec_b)

    return PairResult(
        pair_id=pair.id,
        similarity=sim,
        predicted_same=sim >= threshold,
        actual_same=pair.label == "same",
        embed_time_a_ms=(t1 - t0) * 1000,
        embed_time_b_ms=(t2 - t1) * 1000,
    )


def compute_metrics(
    results: list[PairResult],
    threshold: float = 0.92,
) -> BenchmarkMetrics:
    """Compute aggregate metrics from pair results."""
    if not results:
        return BenchmarkMetrics()

    tp = sum(1 for r in results if r.predicted_same and r.actual_same)
    fp = sum(1 for r in results if r.predicted_same and not r.actual_same)
    tn = sum(1 for r in results if not r.predicted_same and not r.actual_same)
    fn = sum(1 for r in results if not r.predicted_same and r.actual_same)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Similarity distributions
    same_sims = [r.similarity for r in results if r.actual_same]
    diff_sims = [r.similarity for r in results if not r.actual_same]

    mean_sim_same = float(np.mean(same_sims)) if same_sims else 0.0
    std_sim_same = float(np.std(same_sims)) if same_sims else 0.0
    mean_sim_different = float(np.mean(diff_sims)) if diff_sims else 0.0
    std_sim_different = float(np.std(diff_sims)) if diff_sims else 0.0
    separation_gap = mean_sim_same - mean_sim_different

    # Latency
    all_times = []
    total_wall_ms = 0.0
    for r in results:
        all_times.extend([r.embed_time_a_ms, r.embed_time_b_ms])
        total_wall_ms += r.embed_time_a_ms + r.embed_time_b_ms

    mean_embed_ms = float(np.mean(all_times)) if all_times else 0.0
    p95_embed_ms = float(np.percentile(all_times, 95)) if all_times else 0.0
    p99_embed_ms = float(np.percentile(all_times, 99)) if all_times else 0.0
    throughput = (len(results) / (total_wall_ms / 1000)) if total_wall_ms > 0 else 0.0

    return BenchmarkMetrics(
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        mean_sim_same=mean_sim_same, std_sim_same=std_sim_same,
        mean_sim_different=mean_sim_different, std_sim_different=std_sim_different,
        separation_gap=separation_gap,
        mean_embed_ms=mean_embed_ms, p95_embed_ms=p95_embed_ms, p99_embed_ms=p99_embed_ms,
        throughput_pairs_per_sec=throughput,
        pair_results=results,
    )
