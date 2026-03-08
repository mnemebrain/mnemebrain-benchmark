# Reproducing Results

## Prerequisites

```bash
# Clone the benchmark repo
git clone https://github.com/mnemebrain/mnemebrain-benchmark.git
cd mnemebrain-benchmark

# Install with embedding support
pip install -e ".[embeddings]"
```

## BMB Benchmark

### All adapters (local only)

```bash
mnemebrain-bmb
```

This runs all locally available adapters: `mnemebrain_lite`, `structured_memory`, `naive_baseline`, `rag_baseline`, `langchain_buffer`.

### MnemeBrain Lite (recommended starting point)

```bash
pip install mnemebrain-lite[embeddings]
mnemebrain-bmb --adapter mnemebrain_lite
```

Expected: ~93% (24 tasks attempted, 24 skipped)

### MnemeBrain Full

Requires the MnemeBrain backend server running locally:

```bash
mnemebrain-bmb --adapter mnemebrain
```

Expected: 100% (48 tasks attempted, 0 skipped)

### Structured Memory

```bash
mnemebrain-bmb --adapter structured_memory
```

Expected: ~36% (18 tasks attempted, 30 skipped)

### Mem0 (cloud API)

Requires `MEM0_API_KEY` environment variable:

```bash
export MEM0_API_KEY=your_key_here
pip install -e ".[mem0]"
mnemebrain-bmb --adapter mem0
```

Expected: ~29% (18 tasks attempted, 30 skipped)

### OpenAI RAG (cloud API)

Requires `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=your_key_here
pip install -e ".[openai]"
mnemebrain-bmb --adapter openai_rag
```

Expected: 0% (5 tasks attempted, 43 skipped)

### Baselines

```bash
mnemebrain-bmb --adapter naive_baseline
mnemebrain-bmb --adapter rag_baseline
mnemebrain-bmb --adapter langchain_buffer
```

Expected: 0% each (5 tasks attempted, 43 skipped)

### Single category

```bash
mnemebrain-bmb --category contradiction
mnemebrain-bmb --category belief_revision
mnemebrain-bmb --category evidence_tracking
mnemebrain-bmb --category temporal
mnemebrain-bmb --category counterfactual
mnemebrain-bmb --category consolidation
mnemebrain-bmb --category multihop
mnemebrain-bmb --category pattern_separation
```

### JSON output

```bash
mnemebrain-bmb --output results/my_run.json
```

## System Benchmark

```bash
mnemebrain-benchmark                                          # All adapters
mnemebrain-benchmark --adapter mnemebrain_lite                 # Lite only
mnemebrain-benchmark --adapter mnemebrain_lite --category contradiction  # Single category
```

## Task-Level Evaluations

```bash
mnemebrain-task-eval                    # All evaluations
mnemebrain-task-eval --eval preference  # Preference tracking only
mnemebrain-task-eval --eval qa          # Long-horizon QA only
```

## Embedding Provider Benchmark

```bash
python -m mnemebrain_benchmark.runner                          # All providers
python -m mnemebrain_benchmark.runner --provider sentence_transformers
python -m mnemebrain_benchmark.runner --provider sentence_transformers --model all-MiniLM-L6-v2
```

OpenAI providers require `OPENAI_API_KEY`.

## Environment Variables

| Variable | Required for | Purpose |
|----------|-------------|---------|
| `MEM0_API_KEY` | mem0 adapter | Mem0 cloud API authentication |
| `OPENAI_API_KEY` | openai_rag adapter, OpenAI embedding providers | OpenAI API authentication |
| `MNEMEBRAIN_URL` | mnemebrain adapter | MnemeBrain backend server URL (default: `http://localhost:8000`) |

## Determinism and Reproducibility

BMB scenarios are **deterministic by design**. There are no random seeds, sampling, or LLM calls in the benchmark loop. Scenarios define fixed action sequences and fixed expectations — the same adapter code on the same input always produces the same checks.

Sources of variation between runs:

| Source | Affects | Magnitude |
|--------|---------|-----------|
| Embedding model version | Similarity-based checks (dedup, query retrieval) | Small — thresholds are calibrated with margin |
| Floating-point platform differences | Confidence scores near thresholds | Rare — only affects edge-case scenarios |
| Cloud API changes (mem0, openai_rag) | Any check for those adapters | Possible — external APIs may change behaviour |
| Local adapters (baselines, lite) | Nothing | Fully deterministic across runs |

For academic reproducibility:
- Pin `sentence-transformers` and model versions in your environment
- Record `pip freeze` output alongside benchmark results
- Use `--output results/run.json` to capture full per-check results for comparison

## Verifying Results

After running benchmarks, compare your output against the published results in [LEADERBOARD.md](../leaderboard/LEADERBOARD.md). Minor variations in embedding-dependent scores (structured_memory, mem0) are expected due to model version differences. Capability-gated scores (contradiction detection, sandbox reasoning) should be deterministic.
