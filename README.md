# MnemeBrain Benchmark Suite

Belief-based memory benchmarks for AI agents. Tests whether memory systems can detect contradictions, revise beliefs, and track evidence over time.

## The Problem

AI agents remember text, not beliefs.

If a user says "I'm vegetarian" and later "I ate steak yesterday", most memory systems either overwrite the first statement or store both with no conflict signal. Agents need belief maintenance, not document retrieval.

## Quickstart

```bash
pip install mnemebrain-benchmark

# Run the flagship benchmark (BMB)
mnemebrain-bmb

# Run with a specific adapter
mnemebrain-bmb --adapter mnemebrain

# Run task-level evaluations
mnemebrain-task-eval
```

## Results

```
Belief Maintenance Benchmark (BMB)
48 tasks | 8 categories | ~100 checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mnemebrain (full)    ████████████████████ 100%
  mnemebrain_lite      ██████████████████   93%
  structured_memory    ███████              36%
  mem0 (real API)      █████                29%
  naive_baseline                             0%
  rag_baseline                               0%
  openai_rag (real API)                      0%
  langchain_buffer                           0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Full analysis with failure breakdowns: [BMB_REPORT.md](BMB_REPORT.md) | Lite vs full comparison: [LITE_VS_FULL_REPORT.md](LITE_VS_FULL_REPORT.md)

## What's Tested

### BMB (48 tasks, 8 categories)

The flagship benchmark. Tests belief dynamics that retrieval benchmarks miss entirely.

| Category | What it tests | Tasks |
|----------|---------------|-------|
| Contradiction Detection | Belnap four-valued logic (BOTH state) | 6 |
| Belief Revision | AGM-style revision with evidence chains | 6 |
| Evidence Tracking | explain() justification chains | 6 |
| Temporal Updates | Decay, staleness, time-validity expiry | 6 |
| Counterfactual Reasoning | Sandbox simulation without canonical mutation | 6 |
| Consolidation | Episodic-to-semantic compression | 6 |
| Multi-hop Retrieval | HippoRAG graph traversal, PageRank | 6 |
| Pattern Separation | ANN-first orthogonalisation | 6 |

### Task-Level Evaluations (18 scenarios, ~59 questions)

Measures downstream impact: does better memory actually produce correct answers?

- **Preference Tracking** (10 scenarios): dietary changes, relocations, schedule shifts, taste evolution
- **Long-Horizon QA** (8 scenarios): project management, medical, legal, scientific knowledge revision

### System Benchmark (20 scenarios)

End-to-end evaluation across contradiction, retraction, decay, dedup, extraction, and lifecycle categories.

### Embedding Provider Benchmark

Evaluates 5 embedding providers on claim deduplication accuracy and latency.

## Adapters

| Adapter | Capabilities | Type |
|---------|-------------|------|
| `mnemebrain` | All 12 | SDK (HTTP) |
| `mnemebrain_lite` | 7 (store, query, retract, explain, contradiction, decay, revise) | Library (embedded) |
| `structured_memory` | store, query, retract, explain, revise, contradiction | Local |
| `mem0` | store, query, retract, explain, revise | Cloud API |
| `naive_baseline` | store, query | Local |
| `rag_baseline` | store, query | Local |
| `openai_rag` | store, query | Cloud API |
| `langchain_buffer` | store, query | Local |

`mnemebrain_lite` wraps `mnemebrain-lite` directly — no server, no SDK, no HTTP. Install with `pip install mnemebrain-lite[embeddings]`.

Scenarios requiring capabilities the adapter lacks are automatically skipped.

## CLI Reference

```bash
# BMB benchmark
mnemebrain-bmb                                    # All adapters
mnemebrain-bmb --adapter mnemebrain               # Full backend (requires server)
mnemebrain-bmb --adapter mnemebrain_lite          # Lite (no server needed)
mnemebrain-bmb --category contradiction            # Single category
mnemebrain-bmb --scenario bmb_vegetarian_contradiction  # Single scenario
mnemebrain-bmb --output results/bmb_report.json    # Custom output

# System benchmark
mnemebrain-benchmark                               # All adapters
mnemebrain-benchmark --adapter mnemebrain --category contradiction

# Task evaluations
mnemebrain-task-eval                               # All evaluations
mnemebrain-task-eval --eval preference             # Preference tracking only
mnemebrain-task-eval --eval qa                     # Long-horizon QA only

# Embedding benchmark
python -m mnemebrain.benchmark                     # All providers
python -m mnemebrain.benchmark --provider sentence_transformers --model all-MiniLM-L6-v2
```

API-based adapters (`mem0`, `openai_rag`) require environment variables or `.env` file.

## Architecture

```
mnemebrain_benchmark/
├── interface.py          # MemorySystem ABC + result dataclasses
├── protocols.py          # EmbeddingProvider protocol
├── providers.py          # Shared SentenceTransformerProvider + cosine_similarity
├── scoring.py            # Expectation evaluation engine
├── system_runner.py      # Scenario executor (dispatch-table based)
├── system_report.py      # Scorecard formatting
├── runner.py             # Embedding benchmark runner
├── adapters/             # 8 memory system implementations
├── scenarios/            # Scenario schema + JSON loader
└── task_evals/           # Task-level evaluation framework
```

## Adding Adapters and Scenarios

See [docs/adding-adapters.md](docs/adding-adapters.md) for implementing new memory system adapters.

BMB scenarios are defined in `src/mnemebrain_benchmark/scenarios/data/bmb_scenarios.json`. Task evaluation scenarios are in `src/mnemebrain_benchmark/task_evals/data/`. See [docs/architecture.md](docs/architecture.md) for the full project structure and design.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

MIT
