# Scoring Details

## Binary Check Evaluation

Every expectation in a BMB scenario produces one binary check: **pass** or **fail**. There are no partial scores within a single check.

## Denominator Calculation

The denominator for an adapter's score is the total number of checks from scenarios that were actually executed (not skipped). Skipped scenarios contribute zero checks.

```
adapter_score = total_passed_checks / total_checks_from_attempted_scenarios
```

### Example

If an adapter attempts 5 scenarios producing 12 total checks and passes 0:
- Score = 0/12 = 0%
- The 43 skipped scenarios and their ~88 checks are excluded entirely

If an adapter attempts all 48 scenarios producing ~100 checks and passes all:
- Score = 100/100 = 100%

## Check Evaluation Rules

### Type matching

Many checks are gated by result type. For example, `contradiction_detected` is only evaluated when the action result is a `StoreResult`. If the result is a different type (or `None`), the check still runs but compares against `None`, which typically causes a failure.

### Comparison semantics

| Check field | Comparison | Example |
|------------|-----------|---------|
| `truth_state` | Exact string match | `"BOTH"`, `"true"`, `"false"` |
| `contradiction_detected` | Boolean equality | `True` / `False` |
| `confidence_above` | `actual > threshold` | `confidence_above: 0.5` passes if confidence is 0.6 |
| `confidence_below` | `actual < threshold` | `confidence_below: 0.3` passes if confidence is 0.2 |
| `*_gte` fields | `actual >= expected` | `affected_beliefs: 1` passes if result is 1 or more |
| `query_returns_claim` | Boolean: `len(results) > 0` | Checks if query returned any results |
| `query_returns_nothing` | Boolean: `len(results) == 0` | Checks if query returned empty |

### Null handling

When a check expects a value but the result is `None` (common for baselines that don't implement belief states):
- `truth_state` check: `None != "BOTH"` → **fail**
- `confidence_above` check: `None is not None` → **fail**
- `contradiction_detected` check: only evaluated if result is `StoreResult` — baselines that return proper `StoreResult` with `contradiction_detected=False` will fail when `True` is expected

## Per-Category Aggregation

```python
category_score = mean(scenario.score() for scenario in category if not scenario.skipped)
```

Categories where all scenarios are skipped show `--` (no score).

## Per-Adapter Aggregation

BMB reports three metrics per adapter:

### 1. Raw score (attempted categories only)

```python
raw_score = mean(cat.score for cat in categories if not cat.skipped)
```

This is the average across non-skipped categories. It answers: "How well does the adapter perform on what it supports?"

### 2. Coverage

```python
coverage = categories_attempted / total_categories
```

The fraction of BMB categories the adapter actually runs. An adapter that skips 4 of 8 categories has 50% coverage.

### 3. Weighted score (overall)

```python
weighted_score = raw_score × coverage
```

This is the headline number on the leaderboard. It penalises adapters that skip categories, preventing a system that passes 4/8 categories from appearing equivalent to one that passes 8/8.

**Why weighted?** Without coverage weighting, an adapter that only implements `store` + `query` and skips 7/8 categories could score 0% (raw) — the same as one that attempts all 8 and fails everything. That's fine. But an adapter that implements 4 capabilities and scores 100% raw on those 4 categories would show 100% — identical to a system that passes all 48 scenarios. The weighted score makes this distinction visible: 100% × 4/8 = **50%** vs 100% × 8/8 = **100%**.

Since each category has 6 scenarios and categories are either fully attempted or fully skipped for a given adapter, the raw score is equivalent to a flat mean across all non-skipped scenarios.

## Worked Example: Full Transparency Breakdown

### rag_baseline (2 capabilities: store, query)

```
Total BMB scenarios:   48
Skipped (missing caps): 43  (require retract/explain/contradiction/decay/sandbox/etc.)
Attempted:              5   (only scenarios requiring just store + query)
Categories attempted:   1/8 (contradiction only)

Checks from attempted scenarios: 12
Passed:   0   (baselines return truth_state=None, contradiction_detected=False)
Failed:  12   (expectations require belief states the baseline doesn't compute)

Raw score:      0 / 12 = 0%
Coverage:       1/8 = 12.5%
Weighted score: 0% × 12.5% = 0%
```

### mnemebrain_lite (7 capabilities)

```
Total BMB scenarios:   48
Skipped (missing caps): 24  (require sandbox/attack/consolidation/hipporag/pattern_separation)
Attempted:             24   (contradiction, belief_revision, evidence_tracking, temporal)
Categories attempted:   4/8

Checks from attempted scenarios: ~50
Passed:  ~50
Failed:    0

Raw score:      ~50 / ~50 = 100%
Coverage:       4/8 = 50%
Weighted score: 100% × 50% = 50%
```

### mnemebrain (12 capabilities)

```
Total BMB scenarios:   48
Skipped:                0
Attempted:             48
Categories attempted:   8/8

Checks from attempted scenarios: ~100
Passed:  ~100
Failed:    0

Raw score:      ~100 / ~100 = 100%
Coverage:       8/8 = 100%
Weighted score: 100% × 100% = 100%
```

The key insight: Lite and Full both score 100% raw, but they cover different amounts of the benchmark. The weighted score makes this visible — Lite at **50%** vs Full at **100%** correctly reflects that Lite covers half the categories.

## Score Interpretation Guide

### Weighted score (leaderboard headline)

| Score Range | Interpretation |
|------------|---------------|
| 100% | All 8 categories attempted and passed |
| 50% | Either passes all of 4/8 categories, or passes half of all 8 |
| 10-15% | Passes some checks on 2-3 categories, skips the rest |
| 0% | No belief maintenance — fails all attempted checks |

### Raw score (per-category detail)

| Score Range | Interpretation |
|------------|---------------|
| 100% | All attempted checks pass — full belief maintenance on declared capabilities |
| 90-99% | Near-complete — minor edge cases in threshold tuning or temporal logic |
| 30-40% | Partial belief support — typically has revise/explain but not contradiction/decay |
| 0% | No belief maintenance — stores and retrieves text but doesn't track belief states |
| `--` | Category skipped — adapter lacks required capabilities |
