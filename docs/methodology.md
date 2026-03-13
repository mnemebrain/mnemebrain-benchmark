# Methodology

## Capability-Aware Evaluation

BMB is a capability-aware benchmark. Each scenario declares the capabilities it requires (e.g., `contradiction`, `decay`, `sandbox`). Before executing a scenario, the runner checks the adapter's declared capabilities against the scenario's requirements. If the adapter lacks any required capability, the scenario is **skipped** — not counted as a failure.

This means different adapters are evaluated on different subsets of the 48 BMB tasks:

| System | Capabilities | Tasks Attempted | Tasks Skipped |
|--------|:------------:|:---------------:|:-------------:|
| mnemebrain (full) | 12 | 48 | 0 |
| mnemebrain_lite | 7 | 24 | 24 |
| structured_memory | 6 | 18 | 30 |
| mem0 | 5 | 18 | 30 |
| naive_baseline | 2 | 5 | 43 |
| rag_baseline | 2 | 5 | 43 |
| openai_rag | 2 | 5 | 43 |
| langchain_buffer | 2 | 5 | 43 |

A system scoring 0% on 5 attempted tasks is meaningfully different from one scoring 0% on 48 tasks. The enhanced results table in the README reports both attempted and skipped counts to make this transparent.

## Scenario Structure

Each BMB scenario is a multi-step test case consisting of:

1. **Actions** — sequential operations executed against the memory system (store, query, retract, explain, revise, sandbox operations, etc.)
2. **Expectations** — binary pass/fail checks evaluated against action results
3. **Requires** — capability list that gates execution

### Example scenario flow

```
Scenario: bmb_vegetarian_contradiction
Requires: [store, query, contradiction]

Actions:
  1. store("Alice is vegetarian", evidence=[...])
  2. store("Alice ate steak yesterday", evidence=[...])

Expectations:
  - store_2.contradiction_detected = true
  - store_2.truth_state = "BOTH"
```

If an adapter declares `{store, query}` but not `contradiction`, this scenario is skipped entirely.

## Skip Logic

The skip decision happens in `system_runner.py`:

```python
system_caps = {c.value for c in system.capabilities()}
for req in scenario.requires:
    if req not in system_caps:
        return ScenarioScore(skipped=True)
```

A skipped scenario produces no checks and contributes nothing to the adapter's score — neither positive nor negative.

## Scoring

### Per-scenario scoring

Each scenario produces a list of binary checks. The scenario score is:

```
scenario_score = passed_checks / total_checks
```

If a scenario has 3 checks and 2 pass, the score is 0.667.

### Per-category scoring

Categories group 6 scenarios each. The category score is the mean of non-skipped scenario scores:

```
category_score = mean(scenario_scores where not skipped)
```

If all scenarios in a category are skipped, the category score is `None` (displayed as `--`).

### Overall scoring

BMB reports three metrics per adapter:

**Raw score** — the mean of all non-skipped scenario scores (how well the adapter performs on what it supports):

```
raw_score = mean(all scenario_scores where not skipped)
```

**Coverage** — the fraction of BMB categories the adapter actually runs:

```
coverage = categories_attempted / total_categories
```

**Weighted score** — the headline leaderboard number, penalising adapters that skip categories:

```
weighted_score = raw_score × coverage
```

This prevents an adapter that passes 4/8 categories from appearing equivalent to one that passes 8/8. See [scoring.md](scoring.md) for worked examples.

## Check Types

BMB uses 20+ binary check types:

| Check | What it verifies |
|-------|-----------------|
| `contradiction_detected` | Store operation detected conflicting belief |
| `truth_state` | Belief has expected Belnap state (true/false/both/neither) |
| `confidence_above` / `confidence_below` | Confidence within expected range |
| `query_returns_claim` / `query_returns_nothing` | Query retrieval correctness |
| `explanation_has_evidence` | Explain provides justification |
| `explanation_supporting_count_gte` | Minimum supporting evidence count |
| `explanation_attacking_count_gte` | Minimum attacking evidence count |
| `affected_beliefs` | Retraction propagation count |
| `merged` | Duplicate detection on store |
| `sandbox_resolved_state` | Counterfactual resolution correctness |
| `sandbox_canonical_unchanged` | Sandbox isolation (no side effects) |
| `revision_superseded_count_gte` | Belief revision propagation |
| `semantic_beliefs_created_gte` | Consolidation output count |
| `episodics_pruned_gte` | Consolidation cleanup count |
| `memory_tier` | Correct tier assignment after consolidation |
| `was_separated` | Pattern separation triggered |
| `multihop_returns_claim` / `multihop_returns_nothing` | Multi-hop retrieval correctness |

## Why 0% Scores Happen

Baselines with only `store` + `query` capabilities attempt 5 of 48 BMB tasks. These 5 tasks test basic store-then-query behaviour but include expectations like `contradiction_detected` and `truth_state` that require actual belief maintenance logic. Since baselines don't implement belief states, they return `None` for these fields, causing all checks to fail.

This is by design — BMB measures belief maintenance, not retrieval. A system that stores and retrieves text but doesn't track belief states will score 0% on belief-maintenance checks, even on the few tasks it attempts.

## Fairness Considerations

- Skipped scenarios don't count as failures in the **raw score** — they are excluded from the per-category average
- The **weighted score** penalises low coverage: an adapter scoring 100% raw on 4/8 categories gets a weighted score of 50%, not 100%
- Different adapters may have very different denominators (5 checks vs 100+ checks)
- The leaderboard reports raw score, coverage, and weighted score for full transparency
