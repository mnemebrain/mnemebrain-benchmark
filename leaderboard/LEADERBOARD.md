# MnemeBrain Benchmark Leaderboard

**Last updated**: 2026-03-08
**Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers)

---

## Belief Maintenance Benchmark (BMB)

48 tasks | 8 categories | ~100 checks

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mnemebrain (full)    ████████████████████ 100%
  mnemebrain_lite      ██████████████████   93%
  structured_memory    ███████              36%
  mem0 (real API)      █████                29%
  naive_baseline                             0%
  rag_baseline                               0%
  openai_rag (real)                          0%
  langchain_buffer                           0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Per-Category Breakdown

| Category | mnemebrain | mnemebrain_lite | structured_memory | mem0 | naive | rag | openai_rag | langchain |
|----------|:----------:|:---------------:|:-----------------:|:----:|:-----:|:---:|:----------:|:---------:|
| Contradiction Detection | **100%** | 91.7% | 0% | 0% | 0% | 0% | 0% | 0% |
| Belief Revision | **100%** | 100% | 39% | 39% | -- | -- | -- | -- |
| Evidence Tracking | **100%** | 100% | 69% | 47% | -- | -- | -- | -- |
| Temporal Updates | **100%** | 83.3% | -- | -- | -- | -- | -- | -- |
| Counterfactual Reasoning | **100%** | -- | -- | -- | -- | -- | -- | -- |
| Consolidation | **100%** | -- | -- | -- | -- | -- | -- | -- |
| Multi-hop Retrieval | **100%** | -- | -- | -- | -- | -- | -- | -- |
| Pattern Separation | **100%** | -- | -- | -- | -- | -- | -- | -- |

`--` = skipped (adapter lacks required capability)

---

## System Benchmark

22 scenarios | 6 categories

| Category | mnemebrain_lite | naive_baseline |
|----------|:---------------:|:--------------:|
| Contradiction | **87.5%** | -- |
| Decay | **100%** | -- |
| Dedup | 50.0% | **75.0%** |
| Extraction | **100%** | 75.0% |
| Lifecycle | **62.5%** | 50.0% |
| Retraction | **83.3%** | -- |
| **Overall** | **80.6%** | 66.7% |

---

## Task-Level Evaluations

18 scenarios | ~59 questions | downstream task correctness

### Preference Tracking (10 scenarios, 31 questions)

| Adapter | Correct | Total | Accuracy |
|---------|:-------:|:-----:|:--------:|
| **mnemebrain_lite** | **23** | 31 | **74.2%** |
| naive_baseline | 10 | 31 | 32.3% |

### Long-Horizon QA (8 scenarios, 25 questions)

| Adapter | Correct | Total | Accuracy |
|---------|:-------:|:-----:|:--------:|
| **mnemebrain_lite** | **19** | 25 | **76.0%** |
| naive_baseline | 6 | 25 | 24.0% |

---

## Capability Matrix

| Capability | mnemebrain | mnemebrain_lite | mem0 | structured | naive/rag/langchain |
|------------|:----------:|:---------------:|:----:|:----------:|:-------------------:|
| store | Y | Y | Y | Y | Y |
| query | Y | Y | Y | Y | Y |
| retract | Y | Y | Y | Y | -- |
| explain | Y | Y | Y | Y | -- |
| contradiction | Y | Y | -- | -- | -- |
| decay | Y | Y | -- | -- | -- |
| revise | Y | Y | Y | Y | -- |
| sandbox | Y | -- | -- | -- | -- |
| attack | Y | -- | -- | -- | -- |
| consolidation | Y | -- | -- | -- | -- |
| hipporag | Y | -- | -- | -- | -- |
| pattern_separation | Y | -- | -- | -- | -- |

---

## Key Findings

1. **Core belief engine is the differentiator**: MnemeBrain Lite scores 93% BMB vs 0-36% for all baselines using only 7 of 12 capabilities.
2. **Baselines fail structurally**: No baseline supports Belnap four-valued logic, temporal decay, or sandbox reasoning.
3. **Mem0 (real API, graph enabled)**: 29% despite graph memory -- no epistemic operations, aggressive deduplication loses evidence.
4. **Task accuracy follows capability**: 74-76% downstream accuracy (Lite) vs 24-32% (baselines) confirms memory capabilities translate to correct answers.

---

## Reports

Detailed benchmark reports are in [`reports/`](reports/):

- [BMB Technical Report](reports/BMB_REPORT.md) -- 48-task belief maintenance benchmark methodology and results
- [Lite vs Full Backend Report](reports/LITE_VS_FULL_REPORT.md) -- mnemebrain-lite vs full backend comparison

## Reproducibility

```bash
pip install mnemebrain-lite[embeddings] sentence-transformers

# From mnemebrain-benchmark repo:
pip install -e ".[embeddings]" -e "../mnemebrain-lite[embeddings]"

mnemebrain-bmb                              # BMB (all adapters)
mnemebrain-bmb --adapter mnemebrain_lite    # BMB (lite only)
mnemebrain-benchmark --adapter mnemebrain_lite  # System benchmark
python -m mnemebrain_benchmark.task_evals --adapter mnemebrain_lite  # Task evals
```
