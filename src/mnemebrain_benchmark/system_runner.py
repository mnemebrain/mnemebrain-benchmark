"""System benchmark runner -- executes scenarios against memory system adapters."""
from __future__ import annotations

from mnemebrain_benchmark.interface import MemorySystem
from mnemebrain_benchmark.scenarios.schema import Scenario
from mnemebrain_benchmark.scoring import (
    CheckResult,
    ScenarioScore,
    evaluate_expectations,
)


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
            if action.type == "store":
                result = system.store(claim=action.claim or "", evidence=action.evidence or [])
                action_results[action.label] = result
            elif action.type == "query":
                result = system.query(claim=action.claim or "")
                action_results[action.label] = result
            elif action.type == "retract":
                target_result = action_results.get(action.target_label or "")
                if target_result is not None and hasattr(target_result, "belief_id"):
                    result = system.retract(target_result.belief_id)  # type: ignore[union-attr]
                    action_results[action.label] = result
            elif action.type == "explain":
                result = system.explain(claim=action.claim or "")
                action_results[action.label] = result
            elif action.type == "wait_days":
                if action.wait_days:
                    try:
                        system.set_time_offset_days(action.wait_days)
                    except NotImplementedError:
                        pass
            elif action.type == "revise":
                target_result = action_results.get(action.target_label or "")
                if target_result is not None and hasattr(target_result, "belief_id"):
                    result = system.revise(
                        belief_id=target_result.belief_id,
                        evidence=action.evidence or [],
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_fork":
                result = system.sandbox_fork(
                    scenario_label=action.scenario_label or "",
                )
                action_results[action.label] = result
            elif action.type == "sandbox_assume":
                sandbox_result = action_results.get(action.sandbox_label or "")
                belief_result = action_results.get(action.belief_label or "")
                if (
                    sandbox_result is not None
                    and hasattr(sandbox_result, "sandbox_id")
                    and belief_result is not None
                    and hasattr(belief_result, "belief_id")
                ):
                    result = system.sandbox_assume(
                        sandbox_id=sandbox_result.sandbox_id,
                        belief_id=belief_result.belief_id,
                        truth_state=action.truth_state_override or "false",
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_resolve":
                sandbox_result = action_results.get(action.sandbox_label or "")
                belief_result = action_results.get(action.belief_label or "")
                if (
                    sandbox_result is not None
                    and hasattr(sandbox_result, "sandbox_id")
                    and belief_result is not None
                    and hasattr(belief_result, "belief_id")
                ):
                    result = system.sandbox_resolve(
                        sandbox_id=sandbox_result.sandbox_id,
                        belief_id=belief_result.belief_id,
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_discard":
                sandbox_result = action_results.get(action.sandbox_label or "")
                if sandbox_result is not None and hasattr(sandbox_result, "sandbox_id"):
                    system.sandbox_discard(sandbox_id=sandbox_result.sandbox_id)
                    action_results[action.label] = sandbox_result
            elif action.type == "add_attack":
                attacker_result = action_results.get(action.belief_label or "")
                target_result = action_results.get(action.target_label or "")
                if (
                    attacker_result is not None
                    and hasattr(attacker_result, "belief_id")
                    and target_result is not None
                    and hasattr(target_result, "belief_id")
                ):
                    result = system.add_attack(
                        attacker_id=attacker_result.belief_id,
                        target_id=target_result.belief_id,
                        attack_type="undermining",
                        weight=0.5,
                    )
                    action_results[action.label] = result
            elif action.type == "consolidate":
                result = system.consolidate()
                action_results[action.label] = result
            elif action.type == "query_multihop":
                result = system.query_multihop(query=action.claim or "")
                action_results[action.label] = result
            elif action.type == "get_memory_tier":
                target_result = action_results.get(action.belief_ref_label or "")
                if target_result is not None and hasattr(target_result, "belief_id"):
                    result = system.get_memory_tier(
                        belief_id=target_result.belief_id,
                    )
                    action_results[action.label] = result

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
                    import sys
                    print(
                        f"  Warning: {system.name()}/{scenario.name}: {exc}",
                        file=sys.stderr,
                    )
                    scores.append(ScenarioScore(
                        scenario_name=scenario.name,
                        category=scenario.category,
                        checks=[CheckResult(
                            name="runtime_error",
                            passed=False,
                            expected="no error",
                            actual=str(exc),
                        )],
                        skipped=False,
                    ))
            results[system.name()] = scores
        return results
