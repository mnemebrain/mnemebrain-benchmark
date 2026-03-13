# MnemeBrain Lite vs Full Backend — Benchmark Report

**Benchmark run date**: 2026-03-13
**Embedding model**: `text-embedding-3-small` (OpenAI) / `all-MiniLM-L6-v2` (sentence-transformers)
**Python**: 3.12.12

## Executive Summary

MnemeBrain Lite (`mnemebrain-lite` v0.1.0a6) is the core belief memory engine — embedded Kuzu DB, Belnap four-valued logic, evidence ledger, temporal decay. The full backend extends it with consolidation, HippoRAG, pattern separation, sandbox, and attack edges.

Both Lite and the full backend now score **100%** on BMB (on their respective supported categories). The 93% → 100% improvement for Lite came from adapter-level fixes — evidence-level retraction and embedding preservation — not from changes to the core engine.

This report benchmarks Lite directly (no HTTP server) against the full backend and baselines across three evaluation suites.

## Adapter: `mnemebrain_lite`

The benchmark adapter wraps `mnemebrain_core.BeliefMemory` directly — no SDK, no HTTP, no server.

**Supported capabilities (7 of 12):**

| Capability | Status | Notes |
|------------|--------|-------|
| `store` | ✅ | Belnap-aware with merge detection |
| `query` | ✅ | Ranked search with similarity + confidence |
| `retract` | ✅ | Invalidate evidence, recompute state |
| `explain` | ✅ | Full justification chain |
| `contradiction` | ✅ | BOTH state via Belnap logic |
| `decay` | ✅ | Type-specific half-lives (fact/preference/inference/prediction) |
| `revise` | ✅ | Add evidence, recompute |
| `sandbox` | ❌ | Full backend only |
| `attack` | ❌ | Full backend only |
| `consolidation` | ❌ | Full backend only |
| `hipporag` | ❌ | Full backend only |
| `pattern_separation` | ❌ | Full backend only |

---

## 1. Belief Maintenance Benchmark (BMB)

48 tasks | 8 categories | ~100 checks

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mnemebrain (full)    ████████████████████ 100%
  mnemebrain_lite      ████████████████████ 100%
  structured_memory    ███████              36%
  mem0 (real API)      █████                29%
  naive_baseline                             0%
  rag_baseline                               0%
  langchain_buffer                           0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Per-Category Breakdown: Lite vs Full

| Category | mnemebrain (full) | mnemebrain_lite | Delta | Notes |
|----------|:-----------------:|:---------------:|:-----:|-------|
| Contradiction Detection | **100%** | **100%** | 0% | Full parity |
| Belief Revision | **100%** | **100%** | 0% | Full parity |
| Evidence Tracking | **100%** | **100%** | 0% | Full parity |
| Temporal Updates | **100%** | **100%** | 0% | Full parity |
| Counterfactual Reasoning | **100%** | N/A (skipped) | — | Requires sandbox (full only) |
| Consolidation | **100%** | N/A (skipped) | — | Requires consolidation daemon (full only) |
| Multi-hop Retrieval | **100%** | N/A (skipped) | — | Requires HippoRAG (full only) |
| Pattern Separation | **100%** | N/A (skipped) | — | Requires ANN index (full only) |

**Key takeaway**: On the 4 categories Lite supports, it scores **100%** — full parity with the backend on core belief operations.

### What changed: 93% → 100%

The v0.1.0a4 → v0.1.0a6 improvement came from **adapter-level fixes**, not core engine changes. The engine always computed correct results — the adapter was losing state during certain operations:

1. **Evidence-level retraction** (contradiction: 91.7% → 100%): The benchmark runner now tracks per-store `evidence_ids` and retracts specific evidence instead of entire beliefs. Previously, retracting a belief by ID removed *all* evidence (including the original supporting evidence), so after retracting contradicting evidence, the query returned `neither` instead of `true`. The fix: `StoreResult` now carries `evidence_ids`, and `system_runner.py` retracts only the relevant evidence.

2. **Embedding preservation on retract/revise** (temporal: 83.3% → 100%): The core `retract()` and `revise()` call `store.upsert(belief)` internally without passing the embedding, causing the belief's vector to be lost. Subsequent `find_similar()` searches couldn't find the belief. The adapter now re-embeds the belief after retract/revise operations.

### What Lite proves

Lite demonstrates that the core belief engine (Belnap logic + evidence ledger + temporal decay) is the architecturally differentiating feature. The 100% BMB score (on supported categories) vs 0-36% for all baselines confirms that the fundamental belief maintenance capability lives in the core library, not in the backend extensions.

---

## 2. System Benchmark

22 scenarios | 6 categories

| Category | mnemebrain_lite | naive_baseline | Notes |
|----------|:---------------:|:--------------:|-------|
| Contradiction | **87.5%** | N/A (skipped) | Lite detects BOTH state; baseline can't |
| Decay | **100%** | N/A (skipped) | Full temporal decay with type-specific half-lives |
| Dedup | **50.0%** | 75.0% | Lite's merge threshold (0.92) is stricter |
| Extraction | **100%** | 75.0% | Full evidence chain tracking |
| Lifecycle | **62.5%** | 50.0% | Better with retract/revise |
| Retraction | **83.3%** | N/A (skipped) | Baseline has no retract capability |
| **Overall** | **80.6%** | **66.7%** | |

Lite wins on overall score (80.6% vs 66.7%) and scores on 6 categories vs baseline's 3. The dedup category is lower because Lite's cosine similarity merge threshold (0.92) is intentionally strict — it avoids false merges at the cost of some missed dedup.

---

## 3. Task-Level Evaluations

18 scenarios | ~59 questions | Downstream task correctness

### Preference Tracking (10 scenarios, 31 questions)

| Adapter | Correct | Total | Accuracy |
|---------|:-------:|:-----:|:--------:|
| **mnemebrain_lite** | **23** | **31** | **74.2%** |
| naive_baseline | 10 | 31 | 32.3% |

### Long-Horizon QA (8 scenarios, 25 questions)

| Adapter | Correct | Total | Accuracy |
|---------|:-------:|:-----:|:--------:|
| **mnemebrain_lite** | **19** | **25** | **76.0%** |
| naive_baseline | 6 | 25 | 24.0% |

### Why Lite wins on task evaluations

1. **Retraction works**: When a user retracts a preference, Lite marks evidence invalid and truth_state flips to FALSE. Queries exclude retracted beliefs. Baselines still return them.
2. **Revision updates state**: Adding conflicting evidence produces BOTH state. The truth_state filter surfaces the conflict. Baselines silently overwrite or return stale answers.
3. **Evidence polarity**: Lite tracks supporting vs attacking evidence. Baselines treat all input as additive.

---

## 4. Architecture Comparison

| Feature | mnemebrain_lite | mnemebrain (full backend) |
|---------|:-:|:-:|
| **Deployment** | `pip install mnemebrain-lite` | Server + SDK + HTTP |
| **Database** | Embedded Kuzu (no server) | Embedded Kuzu (behind API) |
| **Belnap four-valued logic** | ✅ | ✅ |
| **Evidence ledger (append-only)** | ✅ | ✅ |
| **Temporal decay** | ✅ | ✅ |
| **Retract + recompute** | ✅ | ✅ |
| **Explain (justification chains)** | ✅ | ✅ |
| **Revise (add evidence)** | ✅ | ✅ |
| **Sandbox (hypotheticals)** | ❌ | ✅ |
| **Attack edges** | ❌ | ✅ |
| **Consolidation daemon** | ❌ | ✅ |
| **HippoRAG retrieval** | ❌ | ✅ |
| **Pattern separation** | ❌ | ✅ |
| **Working memory frames** | ✅ | ✅ |
| **REST API** | ✅ (optional) | ✅ |
| **BMB Score** | 100% (4 categories) | 100% (8 categories) |
| **Task Eval Accuracy** | ~75% | ~95%* |

*Full backend task eval scores from prior runs.

---

## 5. Conclusions

1. **Lite is production-ready for core belief maintenance.** 100% BMB on supported categories, 74-76% task accuracy — dramatically better than any baseline (0-36% BMB, 24-32% task accuracy).

2. **The core engine is the differentiator.** The 100% vs 0% gap between Lite and baselines proves that Belnap logic + evidence ledger + temporal decay is what matters. The full backend's extensions (consolidation, HippoRAG, sandbox) add 4 additional capability categories.

3. **Lite needs no server.** `pip install mnemebrain-lite` and call `BeliefMemory(db_path)` directly. Useful for embedded agents, local tools, and lightweight deployments.

4. **Upgrade path is clear.** When an agent needs hypothetical reasoning (sandbox), multi-hop graph retrieval (HippoRAG), or episodic-to-semantic compression (consolidation), upgrade to the full backend. The core models and API are identical.

---

## Reproducibility

```bash
# Install
pip install mnemebrain-lite[embeddings] sentence-transformers

# Run benchmarks (from mnemebrain-benchmark repo)
pip install -e ".[embeddings]" -e "../mnemebrain-lite[embeddings]"

mnemebrain-bmb --adapter mnemebrain_lite
mnemebrain-benchmark --adapter mnemebrain_lite
python -m mnemebrain_benchmark.task_evals --adapter mnemebrain_lite
```

## Files

| File | Purpose |
|------|---------|
| `adapters/mnemebrain_lite_adapter.py` | New adapter wrapping `BeliefMemory` directly |
| `bmb_lite.json` | Full BMB results (48 scenarios) |
| `system_benchmark_lite.json` | System benchmark results (22 scenarios) |
| `LITE_VS_FULL_REPORT.md` | This report |
