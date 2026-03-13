"""External benchmark framework for MnemeBrain.

Supports LongMemEval and HotpotQA benchmarks against any MemorySystem adapter.
"""

from mnemebrain_benchmark.external_evals.base import (
    ExternalBenchmarkAdapter,
    Scenario,
)
from mnemebrain_benchmark.external_evals.scorer import (
    BenchmarkReport,
    SubsetScore,
    exact_match,
    token_f1,
)

__all__ = [
    "BenchmarkReport",
    "ExternalBenchmarkAdapter",
    "Scenario",
    "SubsetScore",
    "exact_match",
    "token_f1",
]
