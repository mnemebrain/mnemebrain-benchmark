"""Tests for mnemebrain_benchmark.system_runner -- SystemBenchmarkRunner."""
from __future__ import annotations

from mnemebrain_benchmark.interface import (
    AttackResult,
    Capability,
    ConsolidateResult,
    ExplainResult,
    MemorySystem,
    MemoryTierResult,
    QueryResult,
    RetractResult,
    ReviseResult,
    SandboxResult,
    StoreResult,
)
from mnemebrain_benchmark.scenarios.schema import Action, Expectation, Scenario
from mnemebrain_benchmark.system_runner import SystemBenchmarkRunner


class FakeSystem(MemorySystem):
    """Full-featured fake for testing the runner."""

    def __init__(self):
        self.reset_count = 0

    def name(self) -> str:
        return "fake"

    def capabilities(self) -> set[Capability]:
        return set(Capability)

    def store(self, claim, evidence):
        return StoreResult("b-1#1", False, False, "true", 0.9)

    def query(self, claim):
        return [QueryResult("b-1#1", claim, 0.9, "true")]

    def retract(self, evidence_id):
        return RetractResult(1, 1)

    def explain(self, claim):
        return ExplainResult(claim, True, 1, 0, "true", 0.9)

    def revise(self, belief_id, evidence):
        return ReviseResult(belief_id, "true", 0.95, 1)

    def set_time_offset_days(self, days):
        pass

    def consolidate(self):
        return ConsolidateResult(2, 1, 1)

    def get_memory_tier(self, belief_id):
        return MemoryTierResult(belief_id, "semantic", 3)

    def query_multihop(self, query):
        return [QueryResult("b-1#1", "multi", 0.8, "true")]

    def sandbox_fork(self, scenario_label=""):
        return SandboxResult("sb-1", "true", True)

    def sandbox_assume(self, sandbox_id, belief_id, truth_state):
        return SandboxResult(sandbox_id, truth_state, True)

    def sandbox_resolve(self, sandbox_id, belief_id):
        return SandboxResult(sandbox_id, "false", True)

    def sandbox_discard(self, sandbox_id):
        pass

    def add_attack(self, attacker_id, target_id, attack_type, weight):
        return AttackResult("e1", attacker_id, target_id)

    def reset(self):
        self.reset_count += 1


class TestRunScenario:
    def setup_method(self):
        self.runner = SystemBenchmarkRunner()
        self.system = FakeSystem()

    def test_store_and_query(self):
        scenario = Scenario(
            name="basic", description="d", category="contradiction",
            requires=["store", "query"],
            actions=[
                Action(label="s1", type="store", claim="test", evidence=[{}]),
                Action(label="q1", type="query", claim="test"),
            ],
            expectations=[
                Expectation(action_label="s1", beliefs_stored=1),
                Expectation(action_label="q1", query_returns_claim=True),
            ],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert not score.skipped
        assert score.score() == 1.0

    def test_skip_missing_capability(self):
        class LimitedSystem(FakeSystem):
            def capabilities(self):
                return {Capability.STORE}

        scenario = Scenario(
            name="needs_retract", description="d", category="retraction",
            requires=["retract"],
            actions=[Action(label="r1", type="retract", target_label="s1")],
            expectations=[],
        )
        score = self.runner.run_scenario(LimitedSystem(), scenario)
        assert score.skipped

    def test_retract_action(self):
        scenario = Scenario(
            name="retract_test", description="d", category="retraction",
            requires=["store", "retract"],
            actions=[
                Action(label="s1", type="store", claim="test", evidence=[{}]),
                Action(label="r1", type="retract", target_label="s1"),
            ],
            expectations=[Expectation(action_label="r1", affected_beliefs=1)],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_explain_action(self):
        scenario = Scenario(
            name="explain_test", description="d", category="extraction",
            requires=["explain"],
            actions=[Action(label="e1", type="explain", claim="test")],
            expectations=[Expectation(action_label="e1", explanation_has_evidence=True)],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_wait_days_action(self):
        scenario = Scenario(
            name="decay_test", description="d", category="decay",
            requires=["decay"],
            actions=[Action(label="w1", type="wait_days", wait_days=30)],
            expectations=[],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert not score.skipped

    def test_wait_days_not_implemented(self):
        class NoDecaySystem(FakeSystem):
            def set_time_offset_days(self, days):
                raise NotImplementedError

        scenario = Scenario(
            name="decay_test", description="d", category="decay",
            requires=["decay"],
            actions=[Action(label="w1", type="wait_days", wait_days=30)],
            expectations=[],
        )
        score = self.runner.run_scenario(NoDecaySystem(), scenario)
        assert not score.skipped

    def test_revise_action(self):
        scenario = Scenario(
            name="revise_test", description="d", category="belief_revision",
            requires=["store", "revise"],
            actions=[
                Action(label="s1", type="store", claim="test", evidence=[{}]),
                Action(label="rv1", type="revise", target_label="s1", evidence=[{}]),
            ],
            expectations=[Expectation(action_label="rv1", truth_state="true")],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_sandbox_fork_assume_resolve_discard(self):
        scenario = Scenario(
            name="sandbox_test", description="d", category="counterfactual",
            requires=["store", "sandbox"],
            actions=[
                Action(label="s1", type="store", claim="test", evidence=[{}]),
                Action(label="sf1", type="sandbox_fork", scenario_label="test"),
                Action(
                    label="sa1", type="sandbox_assume",
                    sandbox_label="sf1", belief_label="s1",
                    truth_state_override="false",
                ),
                Action(label="sr1", type="sandbox_resolve", sandbox_label="sf1", belief_label="s1"),
                Action(label="sd1", type="sandbox_discard", sandbox_label="sf1"),
            ],
            expectations=[
                Expectation(action_label="sr1", sandbox_resolved_state="false"),
            ],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_add_attack_action(self):
        scenario = Scenario(
            name="attack_test", description="d", category="contradiction",
            requires=["store", "attack"],
            actions=[
                Action(label="s1", type="store", claim="claim a", evidence=[{}]),
                Action(label="s2", type="store", claim="claim b", evidence=[{}]),
                Action(label="a1", type="add_attack", belief_label="s1", target_label="s2"),
            ],
            expectations=[],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert not score.skipped

    def test_consolidate_action(self):
        scenario = Scenario(
            name="consolidate_test", description="d", category="consolidation",
            requires=["consolidation"],
            actions=[Action(label="c1", type="consolidate")],
            expectations=[Expectation(action_label="c1", semantic_beliefs_created_gte=1)],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_query_multihop_action(self):
        scenario = Scenario(
            name="multihop_test", description="d", category="multihop_retrieval",
            requires=["hipporag"],
            actions=[Action(label="mh1", type="query_multihop", claim="multi query")],
            expectations=[Expectation(action_label="mh1", multihop_returns_claim=True)],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_get_memory_tier_action(self):
        scenario = Scenario(
            name="tier_test", description="d", category="consolidation",
            requires=["store", "consolidation"],
            actions=[
                Action(label="s1", type="store", claim="test", evidence=[{}]),
                Action(label="mt1", type="get_memory_tier", belief_ref_label="s1"),
            ],
            expectations=[Expectation(action_label="mt1", memory_tier="semantic")],
        )
        score = self.runner.run_scenario(self.system, scenario)
        assert score.score() == 1.0

    def test_reset_called(self):
        scenario = Scenario(
            name="reset_test", description="d", category="contradiction",
            requires=[], actions=[], expectations=[],
        )
        self.runner.run_scenario(self.system, scenario)
        assert self.system.reset_count == 1


class TestRunAll:
    def test_run_all_multiple_systems(self):
        runner = SystemBenchmarkRunner()
        system1 = FakeSystem()
        system2 = FakeSystem()
        scenario = Scenario(
            name="basic", description="d", category="contradiction",
            requires=["store"],
            actions=[Action(label="s1", type="store", claim="x", evidence=[{}])],
            expectations=[Expectation(action_label="s1", beliefs_stored=1)],
        )
        results = runner.run_all([system1, system2], [scenario])
        assert "fake" in results
        assert len(results["fake"]) == 1

    def test_run_all_exception_handling(self):
        runner = SystemBenchmarkRunner()

        class CrashSystem(FakeSystem):
            def store(self, claim, evidence):
                raise RuntimeError("boom")

        scenario = Scenario(
            name="crash", description="d", category="contradiction",
            requires=["store"],
            actions=[Action(label="s1", type="store", claim="x", evidence=[{}])],
            expectations=[Expectation(action_label="s1", beliefs_stored=1)],
        )
        results = runner.run_all([CrashSystem()], [scenario])
        scores = results["fake"]
        assert len(scores) == 1
        assert not scores[0].skipped
        assert scores[0].checks[0].name == "runtime_error"
        assert not scores[0].checks[0].passed
