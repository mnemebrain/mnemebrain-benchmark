# MnemeBrain Benchmark Suite

Belief-based memory benchmarks for AI agents. Tests whether memory systems can detect contradictions, revise beliefs, and track evidence over time.

## What BMB measures

BMB (Belief Maintenance Benchmark) evaluates **epistemic operations**: contradiction detection via Belnap four-valued logic, belief revision with evidence chains, temporal decay, counterfactual reasoning, memory consolidation, and multi-hop retrieval. These are the operations that separate belief maintenance from document retrieval.

## What BMB does not measure

BMB does not measure retrieval speed, embedding quality, or storage scalability. It does not test conversational ability, summarisation, or general-purpose reasoning. For embedding provider evaluation, see the separate [Embedding Provider Benchmark](#embedding-provider-benchmark).

## How scoring works

BMB is **capability-aware**. Each scenario declares the capabilities it requires. Before execution, the runner checks the adapter's declared capabilities against the scenario's requirements. If the adapter lacks any required capability, the scenario is **skipped** — not counted as a failure, and excluded from the score denominator.

This means a system supporting 2 of 12 capabilities is evaluated on a smaller task set than one supporting all 12. Scores reflect performance on attempted tasks only. See [docs/methodology.md](docs/methodology.md) for the full scoring rules.

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

## Scoring methodology

> Adapters are scored only on scenarios matching their declared capabilities. Unsupported scenarios are skipped and reported separately — not counted as failures. A system supporting 2 of 12 capabilities is evaluated on a smaller task set than one supporting all 12. The "Tasks Attempted" and "Tasks Skipped" columns below make this explicit.

## Results

```
Belief Maintenance Benchmark (BMB)
48 tasks | 8 categories | ~100 checks
```

| System | Score | Tasks Attempted | Tasks Skipped | Capabilities |
|--------|------:|:---------------:|:-------------:|:------------:|
| mnemebrain (full) | **100%** | 48 | 0 | 12 |
| mnemebrain_lite | 93% | 24 | 24 | 7 |
| structured_memory | 36% | 18 | 30 | 6 |
| mem0 (real API) | 29% | 18 | 30 | 5 |
| naive_baseline | 0% | 5 | 43 | 2 |
| rag_baseline | 0% | 5 | 43 | 2 |
| openai_rag (real API) | 0% | 5 | 43 | 2 |
| langchain_buffer | 0% | 5 | 43 | 2 |

Full analysis with failure breakdowns: [BMB_REPORT.md](leaderboard/reports/BMB_REPORT.md) | Lite vs full comparison: [LITE_VS_FULL_REPORT.md](leaderboard/reports/LITE_VS_FULL_REPORT.md)

## Benchmark categories

BMB tests 48 tasks across 8 categories (6 tasks each):

- **Contradiction detection** — can the system represent mutually supported conflict via Belnap four-valued logic?
- **Belief revision** — can new evidence update prior beliefs with traceability (AGM-style)?
- **Evidence tracking** — does `explain()` return justification chains with supporting/attacking counts?
- **Temporal updates** — does confidence decay over time, with staleness and expiry?
- **Counterfactual reasoning** — can the system fork a sandbox, assume hypotheticals, and resolve without mutating canonical state?
- **Consolidation** — does episodic-to-semantic compression produce correct tier assignments?
- **Multi-hop retrieval** — can graph traversal with PageRank surface indirectly connected beliefs?
- **Pattern separation** — does ANN-first orthogonalisation prevent similar-but-distinct beliefs from merging?

## Systems evaluated

Different systems expose different memory abstractions; BMB evaluates them through a common [adapter interface](docs/adding-adapters.md).

**Reference implementations:**
- `mnemebrain` — full research system with all 12 capabilities (SDK, HTTP)
- `mnemebrain_lite` — open-source embedded implementation with 7 capabilities (library, no server)

**External systems:**
- `mem0` — Mem0 cloud API with graph memory enabled (5 capabilities)
- `openai_rag` — RAG with OpenAI embeddings (2 capabilities)

**Local baselines:**
- `structured_memory` — Mem0-style structured memory, local (6 capabilities)
- `naive_baseline` — flat vector store with local embeddings (2 capabilities)
- `rag_baseline` — RAG with local embeddings, overwrites on conflict (2 capabilities)
- `langchain_buffer` — append-only text buffer (2 capabilities)

### Lite vs Full

- **mnemebrain_lite** = open-source embedded reference implementation (7 capabilities). Install with `pip install mnemebrain-lite[embeddings]`.
- **mnemebrain (full)** = research system with extended capabilities (12 capabilities). Requires backend server.
- The benchmark evaluates both via declared capabilities — the 7% gap comes from edge cases in dedup/temporal thresholds, not fundamental design differences.

## Task-Level Evaluations

18 scenarios (~59 questions) measuring whether better memory produces correct downstream answers:

- **Preference Tracking** (10 scenarios): dietary changes, relocations, schedule shifts, taste evolution
- **Long-Horizon QA** (8 scenarios): project management, medical, legal, scientific knowledge revision

## System Benchmark

20 end-to-end scenarios across contradiction, retraction, decay, dedup, extraction, and lifecycle categories.

## Embedding Provider Benchmark

Evaluates 5 embedding providers on claim deduplication accuracy and latency.

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
├── scoring.py            # Expectation evaluation engine
├── system_runner.py      # Scenario executor (capability-aware skip logic)
├── adapters/             # 8 memory system implementations
├── scenarios/            # Scenario schema + JSON loader
└── task_evals/           # Task-level evaluation framework
```

See [docs/architecture.md](docs/architecture.md) for the full project structure and design.

## Documentation

- [docs/methodology.md](docs/methodology.md) — scoring rules, skip logic, capability-aware evaluation
- [docs/scoring.md](docs/scoring.md) — per-check explanations, denominator calculation
- [docs/reproducing-results.md](docs/reproducing-results.md) — exact commands to reproduce each adapter's score
- [docs/architecture.md](docs/architecture.md) — project structure and core abstractions
- [docs/adding-adapters.md](docs/adding-adapters.md) — implementing new memory system adapters

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

MIT
