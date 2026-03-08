"""Scenario data structures for the system benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_ACTION_TYPES = {
    "store", "retract", "query", "explain", "wait_days",
    "revise", "sandbox_fork", "sandbox_assume", "sandbox_resolve", "sandbox_discard",
    "add_attack", "consolidate", "query_multihop", "get_memory_tier",
}
VALID_CATEGORIES = {
    "contradiction", "retraction", "decay", "dedup", "extraction", "lifecycle",
    "belief_revision", "evidence_tracking", "temporal", "counterfactual",
    "consolidation", "multihop_retrieval", "pattern_separation",
}


@dataclass
class Action:
    label: str
    type: str
    claim: str | None = None
    evidence: list[dict] | None = None
    belief_type: str | None = None
    target_label: str | None = None
    wait_days: int | None = None
    belief_label: str | None = None
    sandbox_label: str | None = None
    scenario_label: str | None = None
    truth_state_override: str | None = None
    belief_ref_label: str | None = None


@dataclass
class Expectation:
    action_label: str
    beliefs_stored: int | None = None
    merged: bool | None = None
    contradiction_detected: bool | None = None
    truth_state: str | None = None
    query_returns_claim: bool | None = None
    query_returns_nothing: bool | None = None
    explanation_has_evidence: bool | None = None
    confidence_above: float | None = None
    confidence_below: float | None = None
    affected_beliefs: int | None = None
    explanation_supporting_count_gte: int | None = None
    explanation_attacking_count_gte: int | None = None
    explanation_expired_count_gte: int | None = None
    sandbox_resolved_state: str | None = None
    sandbox_canonical_unchanged: bool | None = None
    revision_superseded_count_gte: int | None = None
    # Phase 5: Consolidation
    semantic_beliefs_created_gte: int | None = None
    semantic_beliefs_created: int | None = None
    episodics_pruned_gte: int | None = None
    memory_tier: str | None = None
    consolidated_from_count_gte: int | None = None
    # Phase 5: Pattern separation
    was_separated: bool | None = None
    # Phase 5: Multi-hop retrieval
    multihop_returns_claim: bool | None = None
    multihop_returns_nothing: bool | None = None


@dataclass
class Scenario:
    name: str
    description: str
    category: str
    requires: list[str]
    actions: list[Action]
    expectations: list[Expectation]
