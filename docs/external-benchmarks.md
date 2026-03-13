# External Benchmarks

External benchmarks test MnemeBrain memory systems against established academic datasets. Unlike BMB (which tests belief dynamics with custom scenarios), external benchmarks use 3rd-party datasets with standard QA metrics.

## Supported Benchmarks

### LongMemEval

Multi-session conversation memory benchmark. Tests whether a memory system can retain and retrieve information across extended conversation histories.

**Categories:** knowledge updates, temporal reasoning, multi-session recall, preference tracking, event ordering.

**Dataset format:** JSON/JSONL with session histories and questions. Each scenario contains conversation turns and gold-standard Q&A pairs.

```bash
mnemebrain-external-benchmark longmemeval \
    --data-path /path/to/longmemeval.json \
    --system mnemebrain_lite \
    --subset knowledge_update \
    --limit 50 \
    --llm-extract \
    -v
```

### HotpotQA

Multi-hop QA benchmark requiring reasoning across multiple documents. Tests whether a memory system can connect information from different sources to answer complex questions.

**Question types:** bridge (chain reasoning across documents), comparison (compare entities from different sources).

**Dataset format:** Standard HotpotQA JSON with context paragraphs, questions, and gold answers.

```bash
mnemebrain-external-benchmark hotpotqa \
    --data-path /path/to/hotpotqa.json \
    --system structured_memory \
    --difficulty hard \
    --limit 100 \
    --llm-answer \
    -v
```

## Architecture

```
external_evals/
  __init__.py           # Re-exports: ExternalBenchmarkAdapter, Scenario, scoring
  __main__.py           # Unified CLI (longmemeval, hotpotqa subcommands)
  base.py               # Scenario dataclass + ExternalBenchmarkAdapter ABC
  scorer.py             # token_f1(), exact_match(), BenchmarkReport, SubsetScore
  claim_extractor.py    # ExtractedClaim, sentence/LLM extraction
  answer_generator.py   # answer_from_beliefs() -- QueryResult to natural language
  longmemeval/
    loader.py           # Parse LongMemEval JSON/JSONL/directory
    adapter.py          # LongMemEvalAdapter (ingest sessions, answer questions)
    run.py              # run_longmemeval() + legacy system creation
  hotpotqa/
    loader.py           # Parse HotpotQA JSON/JSONL
    adapter.py          # HotpotQAAdapter (multi-hop with HippoRAG fallback)
```

## Core Abstractions

### ExternalBenchmarkAdapter ABC

```python
class ExternalBenchmarkAdapter(ABC):
    def name(self) -> str: ...
    def load_dataset(self, path, subset=None) -> list[Scenario]: ...
    def ingest(self, system, scenario) -> None: ...
    def answer(self, system, question) -> str: ...
    def score(self, predicted, gold) -> float: ...
```

This is separate from the `MemorySystem` ABC -- adapters here bridge between dataset formats and any `MemorySystem` implementation.

### Scenario

```python
@dataclass
class Scenario:
    scenario_id: str
    subset: str
    history: list[dict]       # Conversation turns or document paragraphs
    questions: list[dict]     # Each has "question" and "gold_answer"
    metadata: dict            # Benchmark-specific metadata
```

### Scoring

Standard QA metrics:

- **Token F1** -- precision/recall over normalized tokens (articles, punctuation, case stripped)
- **Exact Match** -- binary, after normalization

Results are grouped by subset with per-question detail available via `--output-json`.

## Claim Extraction

Two modes for converting raw text into beliefs:

1. **Sentence splitting** (default) -- deterministic, splits on sentence boundaries, filters fragments < 10 chars
2. **LLM extraction** (`--llm-extract`) -- uses an LLM to extract structured claims from text

## Answer Generation

`answer_from_beliefs()` converts `QueryResult` objects into answers:

1. Filters out beliefs with `truth_state="false"`
2. Returns the highest-confidence claim text (default mode)
3. With `--llm-answer`, synthesizes a natural language answer from multiple beliefs

## System Selection

The `--system` flag selects which memory system to benchmark. All 8 adapters from the shared adapter factory are available:

- `mnemebrain_lite` (default) — embedded implementation, no server needed
- `mnemebrain` — full backend (requires running MnemeBrain server)
- `naive_baseline` — flat vector store with local embeddings
- `langchain_buffer` — append-only text buffer
- `rag_baseline` — RAG with local embeddings
- `structured_memory` — Mem0-style structured memory (local)
- `mem0` — Mem0 cloud API (requires API key)
- `openai_rag` — RAG with OpenAI embeddings (requires API key)

Legacy aliases `lite` and `full` are still accepted and map to `mnemebrain_lite` and `mnemebrain` respectively.

Custom embedder selection is available via `--embedder` and `--embedder-model` flags.

## Unified CLI

External benchmarks can also be run via the BMB CLI for a single unified workflow:

```bash
# BMB + external benchmarks in one command
mnemebrain-bmb --include-external --data-path /path/to/data.json

# External benchmarks only (skip BMB)
mnemebrain-bmb --external-only --data-path /path/to/data.json --external-benchmark longmemeval

# External only, specific adapter and limit
mnemebrain-bmb --external-only --data-path /path/to/data.json --adapter mnemebrain_lite --external-limit 50
```

## Adding a New External Benchmark

1. Create a new directory under `external_evals/` (e.g., `musique/`)
2. Implement a `loader.py` that parses the dataset into `Scenario` objects
3. Implement an `adapter.py` subclass of `ExternalBenchmarkAdapter`
4. Add a subcommand in `__main__.py`
5. Add tests in `tests/test_external_evals.py`
