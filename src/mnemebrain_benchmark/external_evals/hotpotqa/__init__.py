"""HotpotQA benchmark integration."""

from mnemebrain_benchmark.external_evals.hotpotqa.adapter import HotpotQAAdapter
from mnemebrain_benchmark.external_evals.hotpotqa.loader import load_hotpotqa

__all__ = ["HotpotQAAdapter", "load_hotpotqa"]
