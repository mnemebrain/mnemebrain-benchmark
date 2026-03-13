"""Tests for the external benchmark evaluation framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mnemebrain_benchmark.external_evals.base import ExternalBenchmarkAdapter, Scenario
from mnemebrain_benchmark.external_evals.claim_extractor import (
    ExtractedClaim,
    extract_claims,
    extract_claims_llm,
    extract_claims_sentence,
)
from mnemebrain_benchmark.external_evals.scorer import (
    BenchmarkReport,
    QuestionScore,
    SubsetScore,
    exact_match,
    token_f1,
)
from mnemebrain_benchmark.external_evals.answer_generator import answer_from_beliefs
from mnemebrain_benchmark.interface import QueryResult


# --- Scorer tests ---


class TestTokenF1:
    def test_identical(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_empty_both(self):
        assert token_f1("", "") == 1.0

    def test_empty_predicted(self):
        assert token_f1("", "hello world") == 0.0

    def test_empty_gold(self):
        assert token_f1("hello world", "") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat", "the cat ran")
        # "the" is stripped (article), so tokens: "cat sat" vs "cat ran"
        # common: "cat" -> precision=1/2, recall=1/2, F1=0.5
        assert 0.4 < f1 < 0.6

    def test_articles_stripped(self):
        assert token_f1("a the an", "a the an") == 1.0  # all stripped = empty = 1.0

    def test_case_insensitive(self):
        assert token_f1("Hello World", "hello world") == 1.0

    def test_punctuation_stripped(self):
        assert token_f1("hello, world!", "hello world") == 1.0


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_case_insensitive(self):
        assert exact_match("Hello World", "hello world") == 1.0

    def test_articles_stripped(self):
        assert exact_match("the cat", "cat") == 1.0


class TestBenchmarkReport:
    def test_overall_scores(self):
        report = BenchmarkReport(
            benchmark_name="test",
            system_name="test_system",
            subsets={
                "a": SubsetScore(
                    subset="a",
                    question_scores=[
                        QuestionScore("q1", "Q?", "yes", "yes", f1=1.0, em=1.0),
                        QuestionScore("q2", "Q?", "no", "yes", f1=0.0, em=0.0),
                    ],
                ),
            },
        )
        assert report.overall_f1 == 0.5
        assert report.overall_em == 0.5
        assert report.total_questions == 2

    def test_empty_report(self):
        report = BenchmarkReport(benchmark_name="test", system_name="sys")
        assert report.overall_f1 == 0.0
        assert report.total_questions == 0

    def test_summary_format(self):
        report = BenchmarkReport(
            benchmark_name="test",
            system_name="sys",
            subsets={
                "sub1": SubsetScore(
                    subset="sub1",
                    question_scores=[
                        QuestionScore("q1", "Q?", "yes", "yes", f1=0.8, em=1.0),
                    ],
                ),
            },
        )
        summary = report.summary()
        assert "TEST BENCHMARK REPORT" in summary
        assert "sub1" in summary
        assert "80.0%" in summary


# --- Claim extractor tests ---


class TestClaimExtractorSentence:
    def test_basic_split(self):
        text = "Alice works at Google. Bob works at Meta. Charlie is a student."
        claims = extract_claims_sentence(text)
        assert len(claims) == 3
        assert claims[0].text == "Alice works at Google."
        assert claims[1].text == "Bob works at Meta."

    def test_empty_text(self):
        assert extract_claims_sentence("") == []
        assert extract_claims_sentence("  ") == []

    def test_short_fragments_filtered(self):
        text = "OK. Sure. Alice works at Google as a software engineer."
        claims = extract_claims_sentence(text)
        # "OK." (3 chars) is below _MIN_CLAIM_LENGTH and not keepable → filtered.
        # "Sure." (5 chars) passes length check → kept.
        # The long sentence is always kept.
        assert any("Alice" in c.text for c in claims)
        assert not any(c.text.strip(".") == "OK" for c in claims)

    def test_metadata_preserved(self):
        claims = extract_claims_sentence(
            "Alice works at Google.",
            source_ref="session:s1@2023-01-15",
            timestamp="2023-01-15",
            session="s1",
        )
        assert len(claims) == 1
        assert claims[0].source_ref == "session:s1@2023-01-15"
        assert claims[0].timestamp == "2023-01-15"
        assert claims[0].session == "s1"


class TestClaimExtractorLLM:
    def test_requires_llm_fn(self):
        with pytest.raises(ValueError, match="llm_fn is required"):
            extract_claims_llm("some text")

    def test_with_mock_llm(self):
        def mock_llm(prompt: str) -> str:
            return "1. Alice works at Google\n2. Bob works at Meta"

        claims = extract_claims_llm("...", llm_fn=mock_llm, source_ref="test")
        assert len(claims) == 2
        assert claims[0].text == "Alice works at Google"
        assert claims[1].text == "Bob works at Meta"


class TestExtractClaimsUnified:
    def test_default_sentence_mode(self):
        claims = extract_claims("Alice works at Google. Bob works at Meta.")
        assert len(claims) == 2

    def test_llm_mode(self):
        def mock_llm(prompt: str) -> str:
            return "Alice works at Google"

        claims = extract_claims("...", use_llm=True, llm_fn=mock_llm)
        assert len(claims) == 1


# --- Answer generator tests ---


class TestAnswerGenerator:
    def test_no_results(self):
        answer = answer_from_beliefs("What?", [])
        assert answer == ""

    def test_false_filtered(self):
        results = [
            QueryResult(belief_id="1", claim="old claim", confidence=0.5, truth_state="false"),
        ]
        answer = answer_from_beliefs("What?", results)
        assert answer == ""

    def test_top_claim_returned(self):
        results = [
            QueryResult(belief_id="1", claim="Alice works at Google", confidence=0.9, truth_state="true"),
        ]
        answer = answer_from_beliefs("Where does Alice work?", results)
        # No-LLM path now extracts an answer span; the key fact must be present.
        assert "Google" in answer

    def test_llm_mode(self):
        def mock_llm(prompt: str) -> str:
            return "Alice works at Google"

        results = [
            QueryResult(belief_id="1", claim="Alice works at Google", confidence=0.9, truth_state="true"),
        ]
        answer = answer_from_beliefs("Where does Alice work?", results, llm_fn=mock_llm)
        assert answer == "Alice works at Google"


# --- Scenario dataclass tests ---


class TestScenario:
    def test_defaults(self):
        s = Scenario(scenario_id="test", subset="knowledge_update")
        assert s.history == []
        assert s.questions == []
        assert s.metadata == {}

    def test_with_data(self):
        s = Scenario(
            scenario_id="lme_001",
            subset="knowledge_update",
            history=[{"role": "user", "content": "Alice works at Google"}],
            questions=[{"question": "Where does Alice work?", "gold_answer": "Google"}],
        )
        assert len(s.history) == 1
        assert len(s.questions) == 1


# --- LongMemEval loader tests ---


class TestLongMemEvalLoader:
    def test_load_structured_sessions(self, tmp_path):
        data = [
            {
                "id": "lme_001",
                "category": "knowledge_update",
                "sessions": [
                    {
                        "session_id": "s1",
                        "timestamp": "2023-01-15",
                        "turns": [
                            {"role": "user", "content": "Alice works at Google."},
                            {"role": "assistant", "content": "Got it."},
                        ],
                    }
                ],
                "questions": [
                    {"question": "Where does Alice work?", "answer": "Google", "type": "knowledge_update"}
                ],
            }
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(fp))
        assert len(scenarios) == 1
        assert scenarios[0].scenario_id == "lme_001"
        assert scenarios[0].subset == "knowledge_update"
        assert len(scenarios[0].history) == 2
        assert len(scenarios[0].questions) == 1
        assert scenarios[0].questions[0]["gold_answer"] == "Google"

    def test_load_flat_turns(self, tmp_path):
        data = [
            {
                "id": "lme_002",
                "category": "temporal_reasoning",
                "sessions": [
                    {"role": "user", "content": "I moved to NYC in 2020."},
                    {"role": "assistant", "content": "Noted."},
                ],
                "questions": [{"question": "Where do I live?", "answer": "NYC"}],
            }
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(fp))
        assert len(scenarios) == 1
        assert len(scenarios[0].history) == 2

    def test_subset_filter(self, tmp_path):
        data = [
            {"id": "1", "category": "knowledge_update", "question": "Q?", "answer": "A"},
            {"id": "2", "category": "temporal_reasoning", "question": "Q2?", "answer": "A2"},
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(fp), subset="knowledge_update")
        assert len(scenarios) == 1
        assert scenarios[0].scenario_id == "1"

    def test_limit(self, tmp_path):
        data = [
            {"id": str(i), "category": "test", "question": f"Q{i}?", "answer": f"A{i}"}
            for i in range(10)
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(fp), limit=3)
        assert len(scenarios) == 3

    def test_load_jsonl(self, tmp_path):
        fp = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"id": "1", "category": "test", "question": "Q?", "answer": "A"}),
            json.dumps({"id": "2", "category": "test", "question": "Q2?", "answer": "A2"}),
        ]
        fp.write_text("\n".join(lines))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(fp))
        assert len(scenarios) == 2

    def test_load_directory(self, tmp_path):
        data1 = [{"id": "1", "category": "test", "question": "Q?", "answer": "A"}]
        data2 = [{"id": "2", "category": "test", "question": "Q2?", "answer": "A2"}]
        (tmp_path / "a.json").write_text(json.dumps(data1))
        (tmp_path / "b.json").write_text(json.dumps(data2))

        from mnemebrain_benchmark.external_evals.longmemeval.loader import load_longmemeval

        scenarios = load_longmemeval(str(tmp_path))
        assert len(scenarios) == 2


# --- HotpotQA loader tests ---


class TestHotpotQALoader:
    def test_load_standard_format(self, tmp_path):
        data = [
            {
                "_id": "hpqa_001",
                "question": "Were Scott and Ed of the same nationality?",
                "answer": "yes",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": {"title": ["Scott", "Ed"], "sent_id": [0, 1]},
                "context": {
                    "title": ["Scott", "Ed", "Decoy"],
                    "sentences": [
                        ["Scott is American.", "He directed films."],
                        ["Ed is also American.", "He made movies."],
                        ["Decoy paragraph here."],
                    ],
                },
            }
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.hotpotqa.loader import load_hotpotqa

        scenarios = load_hotpotqa(str(fp))
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.scenario_id == "hpqa_001"
        assert s.subset == "hotpotqa_comparison"
        assert len(s.history) == 3  # 3 paragraphs
        assert s.questions[0]["gold_answer"] == "yes"

    def test_difficulty_filter(self, tmp_path):
        data = [
            {"_id": "1", "question": "Q1?", "answer": "A1", "type": "bridge", "level": "easy", "context": {}},
            {"_id": "2", "question": "Q2?", "answer": "A2", "type": "bridge", "level": "hard", "context": {}},
        ]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))

        from mnemebrain_benchmark.external_evals.hotpotqa.loader import load_hotpotqa

        scenarios = load_hotpotqa(str(fp), difficulty="hard")
        assert len(scenarios) == 1
        assert scenarios[0].scenario_id == "2"
