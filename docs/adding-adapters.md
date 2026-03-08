# Adding a New Adapter

This guide walks through adding a new memory system adapter to the benchmark suite.

## Overview

An adapter wraps an external memory system so the benchmark can evaluate it. Every adapter implements the `MemorySystem` abstract base class and declares which capabilities it supports. Scenarios requiring capabilities the adapter lacks are automatically skipped.

## Step 1: Create the adapter file

Create `src/mnemebrain_benchmark/adapters/your_adapter.py`:

```python
"""Your memory system adapter."""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


class YourAdapter(MemorySystem):
    """One-line description of what this adapter wraps."""

    def __init__(self) -> None:
        # Initialize your memory system client/state
        self._store: list[dict] = []

    def name(self) -> str:
        return "your_system"

    def capabilities(self) -> set[Capability]:
        # Declare only what your system actually supports
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        belief_id = str(uuid4())
        self._store.append({"id": belief_id, "claim": claim})
        return StoreResult(
            belief_id=belief_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        results = []
        for entry in self._store:
            if self._is_relevant(claim, entry["claim"]):
                results.append(QueryResult(
                    belief_id=entry["id"],
                    claim=entry["claim"],
                    confidence=None,
                    truth_state=None,
                ))
        return results

    def reset(self) -> None:
        self._store.clear()

    def _is_relevant(self, query: str, stored: str) -> bool:
        # Your relevance/similarity logic
        ...
```

### Required methods

Every adapter **must** implement these four abstract methods:

| Method | Returns | Purpose |
|--------|---------|---------|
| `name()` | `str` | Unique identifier used in reports and CLI `--adapter` flag |
| `capabilities()` | `set[Capability]` | Declares which features your system supports |
| `store(claim, evidence)` | `StoreResult` | Ingest a belief with optional evidence |
| `query(claim)` | `list[QueryResult]` | Retrieve relevant beliefs |
| `reset()` | `None` | Clear all state between scenarios |

### Optional methods

Override these to unlock more benchmark scenarios:

| Method | Capability | Unlocks |
|--------|------------|---------|
| `retract(belief_id)` | `RETRACT` | Retraction scenarios |
| `explain(claim)` | `EXPLAIN` | Evidence tracking scenarios |
| `set_time_offset_days(days)` | `DECAY` | Temporal decay scenarios |
| `revise(belief_id, evidence)` | `REVISE` | Belief revision scenarios |
| `sandbox_fork(label)` | `SANDBOX` | Counterfactual reasoning |
| `sandbox_assume(id, belief, state)` | `SANDBOX` | Counterfactual reasoning |
| `sandbox_resolve(id, belief)` | `SANDBOX` | Counterfactual reasoning |
| `sandbox_discard(id)` | `SANDBOX` | Counterfactual reasoning |
| `add_attack(attacker, target, type, weight)` | `ATTACK` | Attack relation scenarios |
| `consolidate()` | `CONSOLIDATION` | Memory consolidation scenarios |
| `get_memory_tier(belief_id)` | `CONSOLIDATION` | Memory tier queries |
| `query_multihop(query)` | `HIPPORAG` | Multi-hop retrieval scenarios |

## Step 2: Add the dependency (if needed)

If your adapter depends on an external package, add an optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
your_system = ["your-package>=1.0"]
all = ["mnemebrain-benchmark[mnemebrain,embeddings,mem0,openai,your_system]"]
```

## Step 3: Write tests

Create `tests/test_your_adapter.py`:

```python
"""Tests for your_adapter."""
from __future__ import annotations

from mnemebrain_benchmark.adapters.your_adapter import YourAdapter
from mnemebrain_benchmark.interface import Capability


class TestYourAdapter:
    def test_name(self):
        adapter = YourAdapter()
        assert adapter.name() == "your_system"

    def test_capabilities(self):
        adapter = YourAdapter()
        caps = adapter.capabilities()
        assert Capability.STORE in caps
        assert Capability.QUERY in caps

    def test_store_new(self):
        adapter = YourAdapter()
        result = adapter.store("test claim", [])
        assert result.belief_id
        assert result.merged is False

    def test_query_empty(self):
        adapter = YourAdapter()
        assert adapter.query("anything") == []

    def test_store_then_query(self):
        adapter = YourAdapter()
        adapter.store("the sky is blue", [])
        results = adapter.query("the sky is blue")
        assert len(results) >= 1
        assert results[0].claim == "the sky is blue"

    def test_reset(self):
        adapter = YourAdapter()
        adapter.store("test", [])
        adapter.reset()
        assert adapter.query("test") == []
```

If your adapter uses embeddings, use the test helpers from `tests/helpers.py`:

```python
from helpers import FakeEmbedder, HighSimEmbedder, LowSimEmbedder

def test_store_merged(self):
    adapter = YourAdapter(HighSimEmbedder(), threshold=0.9)
    r1 = adapter.store("first", [])
    r2 = adapter.store("second", [])
    assert r2.merged is True
```

## Step 4: Register in CLI runners

Add your adapter to the CLI so it can be selected with `--adapter your_system`. Open the relevant CLI file(s) and add your adapter to the adapter registry/factory.

## Step 5: Run the benchmarks

```bash
# Run unit tests
uv run pytest tests/test_your_adapter.py -v

# Run system benchmark with your adapter
mnemebrain-benchmark --adapter your_system

# Run BMB with your adapter
mnemebrain-bmb --adapter your_system

# Run task evaluations
mnemebrain-task-eval --adapter your_system
```

## Capability Matrix

This matrix shows exactly which capabilities each adapter declares. The benchmark uses this to determine which scenarios each system attempts — scenarios requiring a capability marked `--` are skipped.

| Capability | rag_baseline | langchain_buffer | naive_baseline | openai_rag | mem0 | structured_memory | mnemebrain_lite | mnemebrain |
|------------|:------------:|:----------------:|:--------------:|:----------:|:----:|:-----------------:|:---------------:|:----------:|
| store | Y | Y | Y | Y | Y | Y | Y | Y |
| query | Y | Y | Y | Y | Y | Y | Y | Y |
| retract | -- | -- | -- | -- | Y | Y | Y | Y |
| explain | -- | -- | -- | -- | Y | Y | Y | Y |
| revise | -- | -- | -- | -- | Y | Y | Y | Y |
| contradiction | -- | -- | -- | -- | -- | Y | Y | Y |
| decay | -- | -- | -- | -- | -- | -- | Y | Y |
| sandbox | -- | -- | -- | -- | -- | -- | -- | Y |
| attack | -- | -- | -- | -- | -- | -- | -- | Y |
| consolidation | -- | -- | -- | -- | -- | -- | -- | Y |
| hipporag | -- | -- | -- | -- | -- | -- | -- | Y |
| pattern_separation | -- | -- | -- | -- | -- | -- | -- | Y |
| **BMB tasks attempted** | **5** | **5** | **5** | **5** | **18** | **18** | **24** | **48** |

`Y` = declared, `--` = not supported (scenarios requiring this are skipped)

## Tips

- Only declare capabilities your system genuinely supports. The benchmark automatically skips scenarios that require missing capabilities -- false declarations cause test failures.
- Use `StoreResult(truth_state=None, confidence=None)` if your system doesn't track belief states.
- For cloud API adapters, guard initialization behind environment variable checks and document the required keys.
- Keep adapters focused: they should translate the `MemorySystem` interface to your system's API, not implement memory logic themselves.
