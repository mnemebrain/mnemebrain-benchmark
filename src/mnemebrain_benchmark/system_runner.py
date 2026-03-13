"""System benchmark runner -- executes scenarios against memory system adapters."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mnemebrain_benchmark.interface import MemorySystem, RetractResult
from mnemebrain_benchmark.scenarios.schema import Action, Scenario
from mnemebrain_benchmark.scoring import (
    CheckResult,
    ScenarioScore,
    evaluate_expectations,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class _HasBeliefId(Protocol):
    belief_id: str


@runtime_checkable
class _HasSandboxId(Protocol):
    sandbox_id: str


def _get_belief_id(action_results: dict[str, object], label: str) -> str | None:
    """Look up a prior result by label and return its belief_id, if any."""
    result = action_results.get(label or "")
    if isinstance(result, _HasBeliefId):
        return result.belief_id
    return None


def _get_sandbox_id(action_results: dict[str, object], label: str) -> str | None:
    """Look up a prior result by label and return its sandbox_id, if any."""
    result = action_results.get(label or "")
    if isinstance(result, _HasSandboxId):
        return result.sandbox_id
    return None


def _handle_store(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.store(
        claim=action.claim or "",
        evidence=action.evidence or [],
    )


def _handle_query(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.query(claim=action.claim or "")


def _get_evidence_ids(action_results: dict[str, object], label: str) -> list[str] | None:
    """Look up a prior StoreResult by label and return its evidence_ids, if any."""
    result = action_results.get(label or "")
    if hasattr(result, "evidence_ids") and result.evidence_ids:
        return result.evidence_ids
    return None


def _handle_retract(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    target = action.target_label or ""
    # Prefer evidence-level retract: only retract evidence added by the target store
    ev_ids = _get_evidence_ids(action_results, target)
    if ev_ids is not None:
        total = RetractResult(affected_beliefs=0, truth_states_changed=0)
        for eid in ev_ids:
            r = system.retract(eid)
            total.affected_beliefs += r.affected_beliefs
            total.truth_states_changed += r.truth_states_changed
        action_results[action.label] = total
        return
    # Fall back to belief-level retract
    bid = _get_belief_id(action_results, target)
    if bid is not None:
        action_results[action.label] = system.retract(bid)


def _handle_explain(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.explain(claim=action.claim or "")


def _handle_wait_days(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    if action.wait_days:
        try:
            system.set_time_offset_days(action.wait_days)
        except NotImplementedError:
            pass


def _handle_revise(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    bid = _get_belief_id(action_results, action.target_label or "")
    if bid is not None:
        action_results[action.label] = system.revise(
            belief_id=bid,
            evidence=action.evidence or [],
        )


def _handle_sandbox_fork(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.sandbox_fork(
        scenario_label=action.scenario_label or "",
    )


def _handle_sandbox_assume(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    sid = _get_sandbox_id(action_results, action.sandbox_label or "")
    bid = _get_belief_id(action_results, action.belief_label or "")
    if sid is not None and bid is not None:
        action_results[action.label] = system.sandbox_assume(
            sandbox_id=sid,
            belief_id=bid,
            truth_state=action.truth_state_override or "false",
        )


def _handle_sandbox_resolve(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    sid = _get_sandbox_id(action_results, action.sandbox_label or "")
    bid = _get_belief_id(action_results, action.belief_label or "")
    if sid is not None and bid is not None:
        action_results[action.label] = system.sandbox_resolve(
            sandbox_id=sid,
            belief_id=bid,
        )


def _handle_sandbox_discard(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    sid = _get_sandbox_id(action_results, action.sandbox_label or "")
    if sid is not None:
        system.sandbox_discard(sandbox_id=sid)
        action_results[action.label] = action_results.get(action.sandbox_label or "")


def _handle_add_attack(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    attacker_bid = _get_belief_id(action_results, action.belief_label or "")
    target_bid = _get_belief_id(action_results, action.target_label or "")
    if attacker_bid is not None and target_bid is not None:
        action_results[action.label] = system.add_attack(
            attacker_id=attacker_bid,
            target_id=target_bid,
            attack_type="undermining",
            weight=0.5,
        )


def _handle_consolidate(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.consolidate()


def _handle_query_multihop(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    action_results[action.label] = system.query_multihop(query=action.claim or "")


def _handle_get_memory_tier(
    system: MemorySystem,
    action: Action,
    action_results: dict[str, object],
) -> None:
    bid = _get_belief_id(action_results, action.belief_ref_label or "")
    if bid is not None:
        action_results[action.label] = system.get_memory_tier(belief_id=bid)


_ACTION_HANDLERS: dict[str, Callable[[MemorySystem, Action, dict[str, object]], None]] = {
    "store": _handle_store,
    "query": _handle_query,
    "retract": _handle_retract,
    "explain": _handle_explain,
    "wait_days": _handle_wait_days,
    "revise": _handle_revise,
    "sandbox_fork": _handle_sandbox_fork,
    "sandbox_assume": _handle_sandbox_assume,
    "sandbox_resolve": _handle_sandbox_resolve,
    "sandbox_discard": _handle_sandbox_discard,
    "add_attack": _handle_add_attack,
    "consolidate": _handle_consolidate,
    "query_multihop": _handle_query_multihop,
    "get_memory_tier": _handle_get_memory_tier,
}


class SystemBenchmarkRunner:
    """Runs benchmark scenarios against MemorySystem adapters."""

    def run_scenario(self, system: MemorySystem, scenario: Scenario) -> ScenarioScore:
        """Execute a single scenario against *system* and return a ScenarioScore."""
        system_caps = {c.value for c in system.capabilities()}
        for req in scenario.requires:
            if req not in system_caps:
                return ScenarioScore(
                    scenario_name=scenario.name,
                    category=scenario.category,
                    checks=[],
                    skipped=True,
                )

        system.reset()
        action_results: dict[str, object] = {}

        for action in scenario.actions:
            handler = _ACTION_HANDLERS.get(action.type)
            if handler is not None:
                handler(system, action, action_results)

        checks = evaluate_expectations(scenario.expectations, action_results)
        return ScenarioScore(
            scenario_name=scenario.name,
            category=scenario.category,
            checks=checks,
            skipped=False,
        )

    def run_all(
        self,
        systems: list[MemorySystem],
        scenarios: list[Scenario],
    ) -> dict[str, list[ScenarioScore]]:
        """Run all scenarios against every system in *systems*."""
        results: dict[str, list[ScenarioScore]] = {}
        for system in systems:
            scores: list[ScenarioScore] = []
            for scenario in scenarios:
                try:
                    scores.append(self.run_scenario(system, scenario))
                except Exception as exc:
                    print(
                        f"  Warning: {system.name()}/{scenario.name}: {exc}",
                        file=sys.stderr,
                    )
                    scores.append(
                        ScenarioScore(
                            scenario_name=scenario.name,
                            category=scenario.category,
                            checks=[
                                CheckResult(
                                    name="runtime_error",
                                    passed=False,
                                    expected="no error",
                                    actual=str(exc),
                                )
                            ],
                            skipped=False,
                        )
                    )
            results[system.name()] = scores
        return results
