# Contributing to MnemeBrain

Thank you for your interest in contributing to **MnemeBrain**.

MnemeBrain is an open research and engineering project building
**belief-based memory infrastructure for AI agents**. Contributions from
researchers, engineers, and practitioners are welcome.

This document explains how to get started, how to propose changes, and
the standards we use for contributions.

------------------------------------------------------------------------

# Table of Contents

1.  Philosophy
2.  Ways to Contribute
3.  Development Setup
4.  Running Tests
5.  Project Structure
6.  Benchmark Adapters
7.  Coding Guidelines
8.  Pull Request Process
9.  Reporting Issues
10. Community Guidelines

------------------------------------------------------------------------

# Philosophy

MnemeBrain is built around a simple idea:

> Agents need **belief systems**, not just memory stores.

Contributions should align with the core design principles:

• Deterministic behavior where possible\
• Reproducible evaluation\
• Explicit evidence tracking\
• Transparent belief revision

The project favors **clarity and correctness over cleverness**.

------------------------------------------------------------------------

# Ways to Contribute

There are several ways to contribute. Pick a starting point that matches your experience:

| Difficulty | Contribution | Typical size |
|------------|-------------|-------------|
| Easy | Add benchmark scenarios (JSON) | ~50 lines |
| Easy | Improve documentation | varies |
| Medium | Implement a new adapter | 100-200 lines |
| Medium | Add task evaluation scenarios | ~100 lines |
| Hard | Add new BMB categories | scoring + scenarios |
| Hard | Core architecture improvements | varies |

## 1. Benchmark adapters

Implement adapters for new memory systems so they can be evaluated using
BMB.

Examples we would love to see:

-   Zep
-   MemGPT
-   Graphiti
-   LlamaIndex memory
-   CrewAI memory
-   custom enterprise systems

Adapters are usually **100--200 lines of code**.

------------------------------------------------------------------------

## 2. Benchmark scenarios

Add new evaluation scenarios to the Belief Maintenance Benchmark.

Examples:

-   new contradiction scenarios
-   belief revision chains
-   temporal decay cases
-   counterfactual reasoning tests

------------------------------------------------------------------------

## 3. Core architecture improvements

Contributions to the memory architecture itself:

-   belief graph improvements
-   consolidation algorithms
-   confidence computation
-   retrieval improvements
-   scaling optimizations

------------------------------------------------------------------------

## 4. Documentation

Help improve:

-   tutorials
-   examples
-   diagrams
-   architecture explanations

Good documentation is as valuable as code.

------------------------------------------------------------------------

## 5. Bug fixes

If you find incorrect behavior, please submit a fix.

All bug fixes should include **tests that reproduce the bug**.

------------------------------------------------------------------------

# Development Setup

Clone the repository:

``` bash
git clone https://github.com/mnemebrain/mnemebrain-benchmark.git
cd mnemebrain-benchmark
```

Install with dev dependencies:

``` bash
uv sync --extra dev
```

------------------------------------------------------------------------

# Running Tests

``` bash
# Full test suite with coverage
uv run pytest --cov=mnemebrain_benchmark --cov-report=term-missing

# Quick test run
uv run pytest -q

# Lint and type check
uv run ruff check src/ tests/
uv run mypy src/mnemebrain_benchmark/
```

Run the benchmark locally:

``` bash
uv run mnemebrain-bmb
```

All checks must pass before submitting a pull request. CI enforces lint, type check, tests, and 80% coverage minimum.

------------------------------------------------------------------------

# Project Structure

```
src/mnemebrain_benchmark/
  interface.py         # MemorySystem ABC + result dataclasses
  scoring.py           # Expectation evaluation engine
  system_runner.py     # Scenario executor
  bmb_cli.py           # BMB CLI entry point
  adapters/            # Memory system implementations
  scenarios/data/      # JSON benchmark scenarios
  task_evals/          # Task-level evaluation runner + data
docs/
  adding-adapters.md   # Full adapter implementation guide
  architecture.md      # Detailed project architecture
tests/
  helpers.py           # Shared test utilities (FakeEmbedder, etc.)
  test_*.py            # One test file per module
```

See [docs/architecture.md](docs/architecture.md) for the full directory listing and design details.

------------------------------------------------------------------------

# Benchmark Adapters

Adapters allow external systems to run the benchmark.

Each adapter implements the `MemorySystem` interface.

Example:

``` python
class MyMemorySystem(MemorySystem):

    def store(self, claim, evidence):
        ...

    def query(self, query):
        ...

    def explain(self, belief_id):
        ...
```

Adapters should:

• behave deterministically\
• avoid LLM calls unless necessary\
• expose system behavior faithfully

Do **not modify the benchmark to make a system pass**.

------------------------------------------------------------------------

# Coding Guidelines

## Style

Follow standard Python conventions.

Use:

-   Python 3.12+
-   type hints on public APIs
-   clear function names
-   formatting and linting enforced by [ruff](https://docs.astral.sh/ruff/)

Example:

``` python
def compute_confidence(evidence: list[Evidence]) -> float:
```

Avoid overly clever abstractions.

------------------------------------------------------------------------

## Tests

Every feature must include tests.

Tests should verify:

-   expected outputs
-   edge cases
-   regression cases

------------------------------------------------------------------------

## Determinism

The benchmark must remain deterministic.

Avoid:

-   random sampling
-   nondeterministic LLM calls
-   hidden global state

------------------------------------------------------------------------

# Pull Request Process

1.  Fork the repository
2.  Create a feature branch

``` bash
git checkout -b feature/my-change
```

3.  Commit changes using [Conventional Commits](https://www.conventionalcommits.org/):

``` bash
git commit -m "feat(adapters): add ChromaDB adapter"
git commit -m "fix(scoring): correct confidence threshold comparison"
git commit -m "test(bmb): add contradiction category edge cases"
```

4.  Push your branch

``` bash
git push origin feature/my-change
```

5.  Open a Pull Request

------------------------------------------------------------------------

## Pull Request Requirements

PRs should include:

-   clear description
-   linked issue (if applicable)
-   tests
-   documentation updates (if needed)

Large architectural changes should be discussed **before
implementation**.

------------------------------------------------------------------------

# Reporting Issues

Use GitHub Issues to report problems.

Please include:

• what happened\
• expected behavior\
• reproduction steps\
• environment details

Example:

    System: Mem0 adapter
    Scenario: contradiction detection
    Expected: BOTH
    Observed: TRUE

------------------------------------------------------------------------

# Community Guidelines

We aim to build a welcoming and constructive community.

Please:

• be respectful\
• focus on ideas, not people\
• provide constructive feedback

Research disagreements are welcome --- hostility is not.

------------------------------------------------------------------------

# License

By contributing, you agree that your contributions will be licensed
under the project's open source license.

------------------------------------------------------------------------

# Final Note

MnemeBrain is an experiment in **making AI agents capable of belief
maintenance**.

The project is still evolving. If you see a way to improve it, we would
love your help.
