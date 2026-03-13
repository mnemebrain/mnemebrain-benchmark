"""Report generation for the system benchmark."""

from __future__ import annotations

import json

from mnemebrain_benchmark.scoring import (
    CategoryScore,
    ScenarioScore,
    aggregate_by_category,
)


def format_scorecard(results: dict[str, list[ScenarioScore]]) -> str:
    """Format benchmark results as a terminal scorecard."""
    if not results:
        return "No results"

    all_categories: set[str] = set()
    system_cats: dict[str, dict[str, CategoryScore]] = {}

    for system_name, scores in results.items():
        cats = aggregate_by_category(scores)
        system_cats[system_name] = cats
        all_categories.update(cats.keys())

    sorted_cats = sorted(all_categories)
    system_names = list(results.keys())

    col_width = max(15, *(len(n) for n in system_names)) + 2
    cat_width = max(20, *(len(c) for c in sorted_cats)) + 2 if sorted_cats else 22

    separator_width = cat_width + col_width * len(system_names)
    lines: list[str] = []
    lines.append("")
    lines.append("MNEMEBRAIN SYSTEM BENCHMARK")
    lines.append("=" * separator_width)

    header = f"{'Category':<{cat_width}}"
    for name in system_names:
        header += f"{name:<{col_width}}"
    lines.append(header)
    lines.append("-" * separator_width)

    for cat in sorted_cats:
        row = f"{cat:<{cat_width}}"
        for name in system_names:
            cat_score = system_cats.get(name, {}).get(cat)
            if cat_score is None or cat_score.skipped:
                cell = "N/A"
            else:
                cell = f"{cat_score.score * 100:.1f}%" if cat_score.score is not None else "N/A"
            row += f"{cell:<{col_width}}"
        lines.append(row)

    lines.append("-" * separator_width)

    # Raw score (average of attempted categories only)
    raw_row = f"{'Score (attempted)':<{cat_width}}"
    for name in system_names:
        cats = system_cats.get(name, {})
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if scored:
            avg = sum(c.score or 0.0 for c in scored) / len(scored)
            cell = f"{avg * 100:.1f}%"
        else:
            cell = "N/A"
        raw_row += f"{cell:<{col_width}}"
    lines.append(raw_row)

    # Coverage (categories attempted / total)
    coverage_row = f"{'Coverage':<{cat_width}}"
    total_cats = len(sorted_cats)
    for name in system_names:
        cats = system_cats.get(name, {})
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if total_cats > 0:
            cell = f"{len(scored)}/{total_cats}"
        else:
            cell = "N/A"
        coverage_row += f"{cell:<{col_width}}"
    lines.append(coverage_row)

    # Weighted overall (score × coverage)
    weighted_row = f"{'Overall (weighted)':<{cat_width}}"
    for name in system_names:
        cats = system_cats.get(name, {})
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if scored and total_cats > 0:
            avg = sum(c.score or 0.0 for c in scored) / len(scored)
            weighted = avg * len(scored) / total_cats
            cell = f"{weighted * 100:.1f}%"
        else:
            cell = "N/A"
        weighted_row += f"{cell:<{col_width}}"
    lines.append(weighted_row)

    lines.append("=" * separator_width)
    lines.append("")

    return "\n".join(lines)


def export_json(results: dict[str, list[ScenarioScore]], path: str) -> None:
    """Export full benchmark results as JSON."""
    data: dict[str, list[dict[str, object]]] = {}
    for system_name, scores in results.items():
        data[system_name] = []
        for score in scores:
            data[system_name].append(
                {
                    "scenario": score.scenario_name,
                    "category": score.category,
                    "skipped": score.skipped,
                    "score": score.score(),
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "expected": str(c.expected),
                            "actual": str(c.actual),
                        }
                        for c in score.checks
                    ],
                }
            )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
