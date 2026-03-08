# Changelog

All notable changes to mnemebrain-benchmark will be documented in this file.

## [0.1.0a2] - 2026-03-08

### Fixed

- `ReviseResult.truth_state` and `ReviseResult.confidence` are now `str | None` and `float | None` — previously declared as non-optional but used with `None` values
- `format_scorecard()` no longer crashes on empty results dict — returns `"No results"` instead of raising `TypeError`
- `aggregate_by_category()` guards against `None` from `ScenarioScore.score()` in sum, preventing `TypeError` on edge cases
- `Mem0Adapter` exception handlers now log warnings instead of silently swallowing errors (`__init__`, `retract`, `revise`, `reset`)
- `Mem0Adapter.store_delay` is now configurable via constructor parameter (was hardcoded `1.5s`)
- `MemorySystem.retract()` parameter renamed from `evidence_id` to `belief_id` — matches actual usage (runner passes `StoreResult.belief_id`)
- `dataset.py` uses `dataclasses.fields()` instead of internal `__dataclass_fields__` accessor
- `sys` imports moved to top-level in `system_runner.py` and `task_evals/runner.py`

### Changed

- **Extracted `providers.py`** — shared `SentenceTransformerProvider` and `cosine_similarity()` replace 5 copy-pasted `_STProvider` classes and 6 duplicated cosine similarity implementations across CLI modules
- **Refactored `system_runner.py`** — 14-branch `if/elif` chain replaced with `_ACTION_HANDLERS` dispatch dict and per-action handler functions
- **Refactored `scoring.py`** — extracted `_extract_truth_state()` and `_extract_confidence()` helpers, reducing 3 repeated `isinstance` cascades to single function calls
- **Refactored `scenarios/loader.py`** — extracted `_parse_scenarios()` to eliminate duplicated parsing logic between `load_scenarios()` and `load_bmb_scenarios()`
- **Refactored task eval loaders** — extracted `_load_task_scenarios()` in `task_evals/base.py`, replacing byte-for-byte duplicate loaders in `preference_tracking.py` and `long_horizon_qa.py`
- Added return type annotations to CLI entry-point functions (`_build_adapters`, `_get_embedder`, `main`)
- Added `warn_unused_ignores = true` to mypy config
- Added upper bounds to optional dependencies: `sentence-transformers<4.0`, `mem0ai<2.0`, `openai<3.0`
- Removed unused imports (`datetime`, `timezone`) from `mnemebrain_lite_adapter.py`

### Tests

- Test coverage improved from 76% to 87% (289 tests, up from 259)
- Added 30 CLI coverage tests for `bmb_cli` (52% → 87%), `system_cli` (28% → 94%), `task_evals.__main__` (0% → 92%)
- Fixed `test_cli.py` adapter count assertion (7 → 8) after `openai_rag` adapter was added

### Docs

- Updated README with architecture overview and fixed embedding benchmark command path
- Expanded CONTRIBUTING.md with development setup and PR guidelines
- Removed BMB_REPORT.md (superseded by JSON reports)

### Known Issues

- `mnemebrain_lite_adapter` accesses private internals (`_store.get()`, `_store._conn.execute()`) — requires upstream API changes to fix

## [0.1.0a1] - 2026-03-08

Initial alpha release.

### Added

- **Belief Maintenance Benchmark (BMB)** — 48 tasks across 8 categories: contradiction detection, belief revision, evidence tracking, temporal updates, counterfactual reasoning, consolidation, multi-hop retrieval, pattern separation
- **System Benchmark** — 20 end-to-end scenarios across contradiction, retraction, decay, dedup, extraction, and lifecycle categories
- **Task-Level Evaluations** — 18 scenarios (~59 questions) for preference tracking (10) and long-horizon QA (8)
- **Embedding Provider Benchmark** — evaluates 5 providers on claim deduplication accuracy and latency
- **8 memory system adapters**: `mnemebrain` (SDK/HTTP), `mnemebrain_lite` (embedded), `structured_memory`, `mem0` (cloud API), `naive_baseline`, `rag_baseline`, `openai_rag` (cloud API), `langchain_buffer`
- `MemorySystem` ABC with `Capability` enum and frozen result dataclasses (`StoreResult`, `QueryResult`, `RetractResult`, `ReviseResult`, `ExplainResult`, etc.)
- `EmbeddingProvider` runtime-checkable protocol
- Scoring engine with expectation evaluation, category aggregation, and scorecard formatting
- JSON report export for all benchmark tracks
- CLI entry points: `mnemebrain-bmb`, `mnemebrain-benchmark`, `mnemebrain-task-eval`
- 259 tests (76% coverage)

### Infrastructure

- CI workflow: lint (ruff), type check (mypy), unit tests, coverage gate (80% min)
- CodeQL and dependency review workflows
- Pylint workflow
- Release workflow: test → build → publish to PyPI + GitHub Release
- GitHub issue templates: bug report, feature request, new adapter proposal
- CONTRIBUTING.md with development setup and PR guidelines
- Documentation: `docs/architecture.md`, `docs/adding-adapters.md`
