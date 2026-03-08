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

```python
overall_score = mean(scenario.score() for scenario in all_scenarios if not scenario.skipped)
```

This is a flat mean across all non-skipped scenarios, not a mean of category scores. Since each category has 6 scenarios and categories are either fully attempted or fully skipped for a given adapter, the two approaches are equivalent in practice.

## Score Interpretation Guide

| Score Range | Interpretation |
|------------|---------------|
| 100% | All attempted checks pass — full belief maintenance on declared capabilities |
| 90-99% | Near-complete — minor edge cases in threshold tuning or temporal logic |
| 30-40% | Partial belief support — typically has revise/explain but not contradiction/decay |
| 0% | No belief maintenance — stores and retrieves text but doesn't track belief states |
| `--` | Category skipped — adapter lacks required capabilities |
