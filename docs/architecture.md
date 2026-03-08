# Architecture

## Directory Structure

```
src/mnemebrain_benchmark/
  __init__.py              # Package exports
  __main__.py              # python -m mnemebrain.benchmark
  bmb_cli.py               # BMB CLI entry point
  system_cli.py            # System benchmark CLI entry point
  runner.py                # Embedding benchmark orchestrator
  system_runner.py         # System benchmark orchestrator
  system_report.py         # Scorecard + JSON export
  interface.py             # MemorySystem ABC + result dataclasses
  scoring.py               # Expectation evaluation + aggregation
  metrics.py               # Embedding benchmark metrics
  dataset.py               # Claim pair dataset loader
  protocols.py             # EmbeddingProvider protocol
  adapters/
    mnemebrain_adapter.py  # Full-capability adapter (12 capabilities)
    naive_baseline.py      # Flat vector store (local embeddings)
    langchain_buffer.py    # Append-only text buffer (no embeddings)
    rag_baseline.py        # RAG with local embeddings (overwrites on conflict)
    structured_memory.py   # Mem0-style structured memory (local)
    mem0_adapter.py        # Real Mem0 cloud API (requires API key)
    openai_rag_adapter.py  # RAG with real OpenAI embeddings (requires API key)
  scenarios/
    schema.py              # Action, Expectation, Scenario dataclasses
    loader.py              # JSON loader + validation
    data/
      scenarios.json       # 20 system benchmark scenarios
      bmb_scenarios.json   # 48 BMB scenarios (8 categories)
  task_evals/
    __init__.py            # Package init
    __main__.py            # CLI: mnemebrain-task-eval
    base.py                # TaskAction, TaskQuestion, TaskResult, TaskScenario, score_question()
    runner.py              # TaskEvalRunner, ScenarioTaskScore, TaskEvalReport
    preference_tracking.py # Preference scenario loader
    long_horizon_qa.py     # QA scenario loader
    data/
      preference_scenarios.json  # 10 preference tracking scenarios
      qa_scenarios.json          # 8 long-horizon QA scenarios
  data/
    claim_pairs.json       # Embedding benchmark dataset
```

## Core Abstractions

### MemorySystem ABC (`interface.py`)

Every adapter implements this abstract base class. It defines 12 capabilities:

| Capability | Method | Description |
|------------|--------|-------------|
| `STORE` | `store(claim, evidence)` | Ingest a belief with evidence |
| `QUERY` | `query(claim)` | Retrieve beliefs by similarity |
| `RETRACT` | `retract(belief_id)` | Invalidate belief, recompute state |
| `EXPLAIN` | `explain(claim)` | Return justification chain |
| `CONTRADICTION` | (via store) | Detect BOTH state via Belnap logic |
| `DECAY` | `set_time_offset_days(days)` | Time-based confidence decay |
| `REVISE` | `revise(belief_id, evidence)` | Add evidence, recompute |
| `SANDBOX` | `sandbox_fork/assume/resolve/discard` | Copy-on-write hypothetical reasoning |
| `ATTACK` | `add_attack(attacker, target)` | Explicit attack edges |
| `CONSOLIDATION` | `consolidate()` | Episodic-to-semantic compression |
| `HIPPORAG` | `query_multihop(query)` | Graph traversal with PageRank |
| `PATTERN_SEPARATION` | (via store) | ANN-first orthogonalisation |

### Scenarios (`scenarios/schema.py`)

Each scenario is a multi-step test case:

- **Actions**: 14 types (`store`, `query`, `retract`, `explain`, `wait_days`, `revise`, `sandbox_fork`, `sandbox_assume`, `sandbox_resolve`, `sandbox_discard`, `add_attack`, `consolidate`, `query_multihop`, `get_memory_tier`)
- **Expectations**: 20+ check fields (`truth_state`, `confidence_above`, `contradiction_detected`, `merged`, `query_returns_claim`, etc.)
- **Requires**: capability list -- scenarios are auto-skipped if the adapter lacks them

### Scoring (`scoring.py`)

Binary pass/fail checks per expectation. 20+ check types covering belief state, confidence thresholds, evidence counts, contradiction detection, sandbox isolation, consolidation metrics, pattern separation, and multi-hop retrieval.

### System Runner (`system_runner.py`)

Orchestrates scenario execution:
1. Checks adapter capabilities against scenario requirements
2. Executes actions sequentially, collecting results
3. Evaluates expectations against action results
4. Produces per-scenario and per-category scores

## Four Benchmark Systems

### 1. Embedding Provider Benchmark

Evaluates embedding providers on claim deduplication accuracy and latency. Tests 5 providers (3 sentence-transformers models, 2 OpenAI models) on cosine similarity classification.

### 2. System Benchmark

20 scenarios across 6 categories (contradiction, retraction, decay, dedup, extraction, lifecycle). Tests 7 adapters end-to-end.

### 3. Belief Maintenance Benchmark (BMB)

Flagship evaluation. 48 tasks across 8 categories testing belief dynamics. See [BMB_REPORT.md](../leaderboard/reports/BMB_REPORT.md) for full results.

### 4. Task-Level Evaluations

18 real-world scenarios (preference tracking + long-horizon QA) measuring downstream task improvement from better memory capabilities.

## Adapters

| Adapter | Capabilities | Type |
|---------|-------------|------|
| `mnemebrain` | All 12 | SDK (HTTP to backend) |
| `structured_memory` | store, query, retract, explain, revise, contradiction | Local (Mem0-style) |
| `mem0` | store, query, retract, explain, revise | Cloud API |
| `naive_baseline` | store, query | Local (flat vector store) |
| `rag_baseline` | store, query | Local (overwrites on conflict) |
| `openai_rag` | store, query | Cloud API (OpenAI embeddings) |
| `langchain_buffer` | store, query | Local (append-only text) |

See [adding-adapters.md](adding-adapters.md) for how to implement a new adapter.

## Adapter Lifecycle

Every benchmark run follows this sequence per adapter, per scenario:

```
1. Capability check
   └─ scenario.requires ⊆ adapter.capabilities()?
      ├─ no  → ScenarioScore(skipped=True), skip to next scenario
      └─ yes → continue

2. Reset
   └─ adapter.reset() — clear all state from previous scenario

3. Action execution
   └─ for each action in scenario.actions:
        handler = ACTION_HANDLERS[action.type]
        handler(adapter, action, results_dict)

4. Expectation evaluation
   └─ scoring.evaluate_expectations(scenario.expectations, results_dict)
      └─ produces list[CheckResult] (binary pass/fail each)

5. Score
   └─ ScenarioScore(checks=checks, skipped=False)
      └─ score = passed / total
```

The runner collects `ScenarioScore` objects across all scenarios, then aggregates by category (mean of non-skipped scenario scores) and overall (mean of all non-skipped scenario scores).

Key invariants:
- `reset()` is called before every scenario — adapters must not carry state between scenarios
- Actions execute sequentially — later actions can reference results from earlier actions via `action.target_label`
- Skipped scenarios produce no checks and do not affect the score denominator
