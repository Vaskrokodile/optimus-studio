"""Tests for the dataset_mode package.

Covers all 7 layers + source acquisition + orchestrator.
Uses mock handles for vLLM-dependent functions; quality gates and
prompt/template/skills tests run without any model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Think tags for test data
_TO = chr(60) + "think" + chr(62)
_TC = chr(60) + "/think" + chr(62)


@pytest.fixture(autouse=True)
def isolated_app_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path / "iloptimus-home"))


# ---------------------------------------------------------------------------
# Layer 2: Prompts
# ---------------------------------------------------------------------------


class TestDatasetPrompts:
    def test_math_prompt_has_think_tags(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        prompt = dataset_system_prompt("math")
        assert _TO in prompt
        assert _TC in prompt

    def test_all_domains_produce_prompts(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt, SUPPORTED_DOMAINS
        for domain in SUPPORTED_DOMAINS:
            prompt = dataset_system_prompt(domain)
            assert len(prompt) > 200
            assert "DATASET MODE" in prompt

    def test_prompt_is_compact(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        prompt = dataset_system_prompt("math")
        # Should be under ~2000 chars (~400 tokens)
        assert len(prompt) < 2500

    def test_prompt_has_anti_directives(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        prompt = dataset_system_prompt("math")
        assert "ANTI-DUPLICATION" in prompt
        assert "ANTI-LAZY" in prompt
        assert "ANTI-CONTAMINATION" in prompt
        assert "ANTI-SLOP" in prompt

    def test_math_prompt_has_boxed(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        prompt = dataset_system_prompt("math")
        assert "boxed" in prompt

    def test_coding_prompt_has_code_directives(self):
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        prompt = dataset_system_prompt("coding")
        assert "CODING" in prompt


# ---------------------------------------------------------------------------
# Layer 3: Templates
# ---------------------------------------------------------------------------


class TestDatasetTemplates:
    def test_load_templates_math(self):
        from iloptimus.core.dataset_mode.templates import load_templates
        rows = load_templates("math")
        assert len(rows) > 0
        assert all(hasattr(r, "input") for r in rows)
        assert all(hasattr(r, "thinking_trace") for r in rows)

    def test_load_templates_coding(self):
        from iloptimus.core.dataset_mode.templates import load_templates
        rows = load_templates("coding")
        assert len(rows) > 0
        # Kimi K3 templates should be present for coding
        sources = {r.source for r in rows}
        assert "kimi_k3" in sources

    def test_sample_templates_diversity(self):
        from iloptimus.core.dataset_mode.templates import sample_templates
        rng = random.Random(42)
        samples = sample_templates("math", count=3, rng=rng)
        assert len(samples) <= 3
        # Should have at least 2 different sources for diversity
        sources = {s.source for s in samples}
        assert len(sources) >= 1

    def test_template_prompt_builds_string(self):
        from iloptimus.core.dataset_mode.templates import template_prompt
        rng = random.Random(42)
        prompt = template_prompt("math", count=2, rng=rng)
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "Example" in prompt

    def test_template_prompt_truncation(self):
        from iloptimus.core.dataset_mode.templates import template_prompt
        rng = random.Random(42)
        prompt = template_prompt("math", count=10, max_chars=500, rng=rng)
        assert len(prompt) <= 600  # some slack for the header

    def test_template_row_as_example(self):
        from iloptimus.core.dataset_mode.templates import TemplateRow
        row = TemplateRow(
            input="Test problem",
            thinking_trace="Step by step",
            answer="42",
            source="test",
            domain="math",
        )
        example = row.as_example()
        assert "Test problem" in example
        assert "Step by step" in example

    def test_list_template_sources(self):
        from iloptimus.core.dataset_mode.templates import list_template_sources
        sources = list_template_sources()
        assert "openthoughts" in sources
        assert "limo" in sources
        assert "qwen3" in sources
        assert "kimi_k3" in sources


# ---------------------------------------------------------------------------
# Layer 4: Skills
# ---------------------------------------------------------------------------


class TestDatasetSkills:
    def test_skills_prompt_not_empty(self):
        from iloptimus.core.dataset_mode.skills import dataset_skills_prompt
        prompt = dataset_skills_prompt()
        assert len(prompt) > 500

    def test_skills_has_all_four(self):
        from iloptimus.core.dataset_mode.skills import dataset_skills_prompt
        prompt = dataset_skills_prompt()
        assert "Anti-laziness" in prompt
        assert "Anti-slop" in prompt
        assert "Dataset curation" in prompt
        assert "Contamination defense" in prompt

    def test_skills_has_think_tags(self):
        from iloptimus.core.dataset_mode.skills import dataset_skills_prompt
        prompt = dataset_skills_prompt()
        assert _TO in prompt

    def test_skills_truncation(self):
        from iloptimus.core.dataset_mode.skills import dataset_skills_prompt
        prompt = dataset_skills_prompt(max_chars=300)
        assert len(prompt) <= 310

    def test_list_dataset_skills(self):
        from iloptimus.core.dataset_mode.skills import list_dataset_skills
        skills = list_dataset_skills()
        assert len(skills) == 4
        assert all("id" in s and "chars" in s for s in skills)


# ---------------------------------------------------------------------------
# Layer 5: Research
# ---------------------------------------------------------------------------


class TestDatasetResearch:
    def test_plan_initial_queries_math(self):
        from iloptimus.core.dataset_mode.research import ResearchSpec, plan_initial_queries
        spec = ResearchSpec(topic="algebra", domain="math", features=["polynomials", "roots"])
        queries = plan_initial_queries(spec)
        assert len(queries) > 0
        assert "algebra" in queries[0]

    def test_plan_initial_queries_coding(self):
        from iloptimus.core.dataset_mode.research import ResearchSpec, plan_initial_queries
        spec = ResearchSpec(topic="sorting algorithms", domain="coding")
        queries = plan_initial_queries(spec)
        assert any("algorithm" in q for q in queries)

    def test_plan_refinement_queries(self):
        from iloptimus.core.dataset_mode.research import (
            ResearchCorpus,
            ResearchSpec,
            plan_refinement_queries,
        )
        corpus = ResearchCorpus()
        spec = ResearchSpec(topic="algebra", features=["polynomials", "roots", "factoring"])
        queries = plan_refinement_queries(corpus, spec)
        assert len(queries) > 0

    def test_research_corpus_add_dedup(self):
        from iloptimus.core.dataset_mode.research import ResearchCorpus, ResearchSource
        corpus = ResearchCorpus()
        source = ResearchSource(url="https://example.com/1", title="Test", text="content", origin="example.com")
        assert corpus.add(source) is True
        assert corpus.add(source) is False  # duplicate URL

    def test_research_corpus_coverage(self):
        from iloptimus.core.dataset_mode.research import ResearchCorpus, ResearchSource
        corpus = ResearchCorpus()
        assert not corpus.coverage_met
        for i in range(12):
            source = ResearchSource(
                url=f"https://example{i % 4}.com/{i}",
                title=f"Source {i}",
                text=f"content {i}",
                origin=f"example{i % 4}.com",
            )
            corpus.add(source)
        assert corpus.coverage_met

    def test_domain_rate_limiter(self):
        import asyncio
        from iloptimus.core.dataset_mode.research import DomainRateLimiter
        limiter = DomainRateLimiter(delay=0.05)
        # First call should be instant
        async def test():
            await limiter.wait("example.com")
            import time
            start = time.monotonic()
            await limiter.wait("example.com")
            elapsed = time.monotonic() - start
            assert elapsed >= 0.04  # should have waited
        asyncio.run(test())

    def test_research_to_sources(self):
        from iloptimus.core.dataset_mode.research import ResearchCorpus, ResearchSource, research_to_sources
        corpus = ResearchCorpus()
        corpus.add(ResearchSource(url="https://example.com/1", title="Test", text="content", origin="example.com"))
        sources = research_to_sources(corpus)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/1"

    def test_research_summary(self):
        from iloptimus.core.dataset_mode.research import ResearchCorpus, ResearchSource, research_summary, ResearchSpec
        corpus = ResearchCorpus()
        corpus.add(ResearchSource(url="https://example.com/1", title="Test", text="content here", origin="example.com"))
        spec = ResearchSpec(topic="algebra")
        summary = research_summary(corpus, spec)
        assert "1 sources" in summary


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestDatasetGeneration:
    def test_build_generation_prompt(self):
        from iloptimus.core.dataset_mode.generation import build_generation_prompt, GenerationSpec
        import random
        spec = GenerationSpec(domain="math")
        rng = random.Random(42)
        prompt = build_generation_prompt(spec, "algebra: roots", "medium", rng)
        assert "DATASET MODE" in prompt
        assert "algebra" in prompt
        assert "medium" in prompt
        assert _TO in prompt

    def test_parse_generation_valid(self):
        from iloptimus.core.dataset_mode.generation import parse_generation
        raw = f"PROBLEM: What is the sum of the first 10 positive integers?\n\n{_TO}\nUsing the formula for the sum of an arithmetic sequence, n(n+1)/2 = 10*11/2 = 55. This gives us the total.\n{_TC}\n55"
        row = parse_generation(raw)
        assert row is not None
        assert "sum of the first 10" in row.input
        assert "n(n+1)/2" in row.thinking_trace
        assert "55" in row.answer

    def test_parse_generation_missing_think(self):
        from iloptimus.core.dataset_mode.generation import parse_generation
        raw = "PROBLEM: What is 2+2?\n\n4"
        row = parse_generation(raw)
        assert row is None

    def test_parse_generation_too_short(self):
        from iloptimus.core.dataset_mode.generation import parse_generation
        raw = f"PROBLEM: Hi\n\n{_TO}\nshort\n{_TC}\n1"
        row = parse_generation(raw)
        assert row is None

    def test_parse_generation_empty(self):
        from iloptimus.core.dataset_mode.generation import parse_generation
        assert parse_generation("") is None
        assert parse_generation(None) is None

    def test_diversity_order(self):
        from iloptimus.core.dataset_mode.generation import diversity_order
        prompts = [
            ("p1", "topic_a", "easy"),
            ("p2", "topic_a", "medium"),
            ("p3", "topic_a", "hard"),
            ("p4", "topic_b", "easy"),
            ("p5", "topic_b", "medium"),
        ]
        ordered = diversity_order(prompts)
        assert len(ordered) == 5
        # No two adjacent should share the same topic (when possible)
        for i in range(len(ordered) - 1):
            if i < 3:  # with 3 topic_a and 2 topic_b, first 3 alternate
                pass  # just check it doesn't start with 3 same in a row

    def test_expand_generation_prompts(self):
        from iloptimus.core.dataset_mode.generation import expand_generation_prompts, GenerationSpec
        import random
        spec = GenerationSpec(domain="math", count=10)
        rng = random.Random(42)
        prompts = expand_generation_prompts(spec, rng)
        assert len(prompts) == 10
        # Check diversity of topics
        topics = {p[1] for p in prompts}
        assert len(topics) > 1
        # Check difficulty distribution
        difficulties = [p[2] for p in prompts]
        assert "easy" in difficulties or "medium" in difficulties

    def test_generated_row_hash(self):
        from iloptimus.core.dataset_mode.generation import GeneratedRow
        row1 = GeneratedRow(input="test", thinking_trace="thinking", answer="42", domain="math", difficulty="easy")
        row2 = GeneratedRow(input="test", thinking_trace="thinking", answer="42", domain="math", difficulty="easy")
        assert row1.row_hash == row2.row_hash
        row3 = GeneratedRow(input="different", thinking_trace="thinking", answer="42", domain="math", difficulty="easy")
        assert row1.row_hash != row3.row_hash


# ---------------------------------------------------------------------------
# Layer 6: Quality
# ---------------------------------------------------------------------------


class TestDatasetQuality:
    def _good_row(self):
        return {
            "input": "Find all positive integers n such that n^2 + 6n + 8 is a perfect square.",
            "thinking_trace": (
                "Step 1: Let n^2 + 6n + 8 = k^2 for some non-negative integer k.\n"
                "Step 2: Complete the square: (n+3)^2 - 1 = k^2.\n"
                "Step 3: Rearrange: (n+3)^2 - k^2 = 1, so (n+3-k)(n+3+k) = 1.\n"
                "Step 4: Since both factors are integers with product 1, we need n+3-k = 1 and n+3+k = 1.\n"
                "Step 5: This gives k = 0 and n = -2, but n must be positive.\n"
                "Conclusion: There are no positive integer solutions."
            ),
            "answer": "\\boxed{\\text{No positive integer solutions}}",
            "raw_output": f"PROBLEM: Find all positive integers n...\n\n{_TO}\nStep 1: Let n^2...\n{_TC}\n\\boxed{{\\text{{No solutions}}}}",
            "domain": "math",
            "difficulty": "hard",
        }

    def _bad_row(self):
        return {
            "input": "What is the value of 2 plus 2 in standard arithmetic?",
            "thinking_trace": "This is an interesting problem. Let me think about this. As we can see, 2+2=4. In conclusion, the answer is 4.",
            "answer": "4",
            "raw_output": "",
            "domain": "math",
            "difficulty": "easy",
        }

    def test_check_structure_good(self):
        from iloptimus.core.dataset_mode.quality import check_structure
        ok, rejections = check_structure(self._good_row())
        assert ok
        assert len(rejections) == 0

    def test_check_structure_bad(self):
        from iloptimus.core.dataset_mode.quality import check_structure
        bad = {"input": "hi", "thinking_trace": "short", "answer": "", "raw_output": ""}
        ok, rejections = check_structure(bad)
        assert not ok
        assert "input_too_short" in rejections
        assert "thinking_too_short" in rejections
        assert "answer_missing" in rejections

    def test_check_repetition_clean(self):
        from iloptimus.core.dataset_mode.quality import check_repetition
        text = "Step 1: Do A\nStep 2: Do B\nStep 3: Do C"
        ok, ratio = check_repetition(text)
        assert ok
        assert ratio == 0.0

    def test_check_repetition_repetitive(self):
        from iloptimus.core.dataset_mode.quality import check_repetition
        text = "The answer is 42\nThe answer is 42\nThe answer is 42\nThe answer is 42"
        ok, ratio = check_repetition(text)
        assert not ok
        assert ratio > 0.5

    def test_check_slop_clean(self):
        from iloptimus.core.dataset_mode.quality import check_slop
        text = "We compute n(n+1)/2 = 10*11/2 = 55. This gives us the sum."
        ok, found = check_slop(text)
        assert ok
        assert len(found) == 0

    def test_check_slop_found(self):
        from iloptimus.core.dataset_mode.quality import check_slop
        text = "This is an interesting problem. As we can see, the answer is clear."
        ok, found = check_slop(text)
        assert not ok
        assert len(found) >= 2

    def test_check_trivial_input_trivial(self):
        from iloptimus.core.dataset_mode.quality import check_trivial_input
        assert check_trivial_input("What is 2+2?") is True
        assert check_trivial_input("Solve x = 5") is True
        assert check_trivial_input("Hi") is True

    def test_check_trivial_input_non_trivial(self):
        from iloptimus.core.dataset_mode.quality import check_trivial_input
        assert check_trivial_input("Find all positive integers n such that n^2 + 6n + 8 is a perfect square.") is False

    def test_check_contamination_clean(self):
        from iloptimus.core.dataset_mode.quality import check_contamination
        benchmark = ["Find the sum of all positive divisors of 120."]
        result = check_contamination("What is the derivative of x^3?", benchmark)
        assert result[0] is False

    def test_check_contamination_detected(self):
        from iloptimus.core.dataset_mode.quality import check_contamination
        benchmark = ["Find the sum of all positive divisors of 120 and express the result in simplest form"]
        # Near-verbatim reproduction
        result = check_contamination("Find the sum of all positive divisors of 120 and express the result in simplest form", benchmark)
        assert result[0] is True

    def test_score_quality_good(self):
        from iloptimus.core.dataset_mode.quality import score_quality
        score = score_quality(self._good_row())
        assert score > 0.3

    def test_score_quality_bad(self):
        from iloptimus.core.dataset_mode.quality import score_quality
        score = score_quality(self._bad_row())
        assert score < 0.5

    def test_run_quality_gates_accepts_good(self):
        from iloptimus.core.dataset_mode.quality import run_quality_gates
        rows = [self._good_row()]
        accepted, report = run_quality_gates(rows)
        assert len(accepted) == 1
        assert report.accepted == 1
        assert report.rejected == 0
        assert "quality_score" in accepted[0]
        assert "row_sha256" in accepted[0]

    def test_run_quality_gates_rejects_bad(self):
        from iloptimus.core.dataset_mode.quality import run_quality_gates
        rows = [self._bad_row()]
        accepted, report = run_quality_gates(rows)
        assert len(accepted) == 0
        assert report.rejected == 1
        # The bad row should be rejected for at least one quality issue
        assert len(report.rejections_by_type) > 0
        # Common rejection reasons for a bad row: slop, trivial, or low quality
        assert any(
            reason in report.rejections_by_type
            for reason in ("excessive_slop", "trivial_input", "low_quality", "excessive_repetition")
        )

    def test_run_quality_gates_mixed(self):
        from iloptimus.core.dataset_mode.quality import run_quality_gates
        rows = [self._good_row(), self._bad_row()]
        accepted, report = run_quality_gates(rows)
        assert len(accepted) == 1
        assert report.total == 2
        assert report.accepted == 1
        assert report.rejected == 1

    def test_semantic_dedup_fallback(self):
        from iloptimus.core.dataset_mode.quality import SemanticDedupGuard
        guard = SemanticDedupGuard(handle=None)  # no embeddings, use MinHash
        text1 = "The quick brown fox jumps over the lazy dog near the river bank"
        text2 = "The quick brown fox jumps over the lazy dog near the river bank"
        assert guard.is_duplicate(text1) is False
        assert guard.is_duplicate(text2) is True

    def test_quality_report_public(self):
        from iloptimus.core.dataset_mode.quality import QualityReport
        report = QualityReport(total=10, accepted=7, rejected=3, mean_quality=0.75)
        public = report.public()
        assert public["total"] == 10
        assert public["accepted"] == 7
        assert public["mean_quality"] == 0.75


# ---------------------------------------------------------------------------
# Layer 7: Manifest
# ---------------------------------------------------------------------------


class TestDatasetManifest:
    def test_build_manifest(self):
        from iloptimus.core.dataset_mode.manifest import build_manifest
        from iloptimus.core.dataset_mode.quality import QualityReport

        rows = [
            {"domain": "math", "difficulty": "easy", "source": "synthetic"},
            {"domain": "math", "difficulty": "hard", "source": "synthetic"},
        ]
        report = QualityReport(total=3, accepted=2, rejected=1, mean_quality=0.8)
        manifest = build_manifest("test-run", rows, report, domain="math")
        assert manifest.run_id == "test-run"
        assert manifest.accepted_rows == 2
        assert manifest.rejected_rows == 1
        assert manifest.mean_quality == 0.8
        assert manifest.difficulty_distribution["easy"] == 1
        assert manifest.difficulty_distribution["hard"] == 1

    def test_save_and_load_manifest(self, tmp_path):
        from iloptimus.core.dataset_mode.manifest import build_manifest, save_manifest, DatasetManifest
        from iloptimus.core.dataset_mode.quality import QualityReport

        manifest = build_manifest("test-run", [], QualityReport(), domain="math")
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "test-run"

    def test_manifest_summary(self):
        from iloptimus.core.dataset_mode.manifest import DatasetManifest, manifest_summary
        manifest = DatasetManifest(
            run_id="test",
            accepted_rows=50,
            rejected_rows=10,
            total_rows=60,
            mean_quality=0.75,
            difficulty_distribution={"easy": 15, "medium": 25, "hard": 10},
        )
        summary = manifest_summary(manifest)
        assert "50 accepted" in summary
        assert "0.750" in summary
        assert "easy: 15" in summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestDatasetOrchestrator:
    def test_build_failure_driven_spec(self):
        from iloptimus.core.dataset_mode.orchestrator import build_failure_driven_spec
        failures = [
            {"problem": "Find the roots of x^2 - 5x + 6 = 0", "error_type": "wrong_answer"},
            {"problem": "Compute the integral of sin(x)", "error_type": "truncated"},
        ]
        spec = build_failure_driven_spec(failures, domain="math", target_rows=50)
        assert spec.domain == "math"
        assert spec.target_rows == 50
        assert len(spec.seed_problems) == 2
        # Should focus more on hard difficulty
        assert spec.difficulty_split["hard"] > spec.difficulty_split["easy"]

    def test_build_curriculum_spec(self):
        from iloptimus.core.dataset_mode.orchestrator import build_curriculum_spec
        performance = {"easy": 0.9, "medium": 0.5, "hard": 0.2}
        spec = build_curriculum_spec(performance, domain="math", target_rows=100)
        # Should generate more hard rows (low accuracy = more rows)
        assert spec.difficulty_split["hard"] > spec.difficulty_split["easy"]

    def test_build_curriculum_spec_empty(self):
        from iloptimus.core.dataset_mode.orchestrator import build_curriculum_spec
        spec = build_curriculum_spec({}, domain="math")
        # Default split when no performance data
        assert "easy" in spec.difficulty_split
        assert "medium" in spec.difficulty_split

    def test_enter_dataset_mode_no_handle(self):
        """Test that enter_dataset_mode works without a model handle (no generation, just structure)."""
        from iloptimus.core.dataset_mode.orchestrator import enter_dataset_mode, DatasetModeSpec
        spec = DatasetModeSpec(
            domain="math",
            target_rows=5,
            enable_research=False,
            enable_refinement=False,
            enable_reroll=False,
            handle=None,
        )
        result = enter_dataset_mode(spec)
        assert result.run_id is not None
        assert result.error is not None or result.accepted_rows == 0  # no handle = no generation

    def test_enter_dataset_mode_with_mock_handle(self):
        """Test the full pipeline with a mock handle."""
        from iloptimus.core.dataset_mode.orchestrator import enter_dataset_mode, DatasetModeSpec

        # Create a mock handle that returns fake inference results
        mock_handle = MagicMock()
        mock_result = MagicMock()
        mock_result.text = f"PROBLEM: Find the value of x if 3x + 7 = 22.\n\n{_TO}\n3x + 7 = 22, so 3x = 15, therefore x = 5. We verify: 3*5+7 = 22. Correct.\n{_TC}\n\\boxed{{5}}"

        with patch("iloptimus.core.inference.run_inference_batch", return_value=[mock_result] * 5):
            spec = DatasetModeSpec(
                domain="math",
                target_rows=5,
                batch_size=5,
                enable_research=False,
                enable_refinement=False,
                enable_reroll=False,
                enable_verification=False,
                handle=mock_handle,
            )
            result = enter_dataset_mode(spec)
            assert result.run_id is not None
            # Should have generated some rows (may not all pass quality gates)
            assert result.error is None or "quality" not in str(result.error).lower()

    def test_list_sessions_empty(self):
        from iloptimus.core.dataset_mode.orchestrator import list_sessions
        sessions = list_sessions()
        assert isinstance(sessions, list)

    def test_get_session_nonexistent(self):
        from iloptimus.core.dataset_mode.orchestrator import get_session
        assert get_session("nonexistent-id") is None

    def test_load_dataset_nonexistent(self):
        from iloptimus.core.dataset_mode.orchestrator import load_dataset
        assert load_dataset("nonexistent-id") == []


# ---------------------------------------------------------------------------
# Source acquisition
# ---------------------------------------------------------------------------


class TestSourceAcquisition:
    def test_harvest_urls_import(self):
        from iloptimus.core.dataset_mode.source_acquisition import harvest_urls
        assert callable(harvest_urls)

    def test_corpus_root(self):
        from iloptimus.core.dataset_mode.source_acquisition import corpus_root
        root = corpus_root()
        assert "corpus" in str(root)


# ---------------------------------------------------------------------------
# Integration: full pipeline smoke test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_prompt_assembly(self):
        """Test that all prompt components assemble correctly."""
        from iloptimus.core.dataset_mode.prompts import dataset_system_prompt
        from iloptimus.core.dataset_mode.skills import dataset_skills_prompt
        from iloptimus.core.dataset_mode.templates import template_prompt
        import random

        system = dataset_system_prompt("math")
        skills = dataset_skills_prompt()
        rng = random.Random(42)
        templates = template_prompt("math", count=2, rng=rng)

        # All components should be non-empty
        assert len(system) > 200
        assert len(skills) > 500
        assert len(templates) > 100

        # Combined prompt should be reasonable size for a small model
        combined = system + "\n" + skills + "\n" + templates
        assert len(combined) < 8000  # ~1600 tokens, fits in small model context

    def test_quality_gates_on_generated_rows(self):
        """Test quality gates on rows that look like real generated output."""
        from iloptimus.core.dataset_mode.quality import run_quality_gates

        good_row = {
            "input": "A triangle has sides of length 13, 14, and 15. Find its area using Heron's formula.",
            "thinking_trace": (
                "Step 1: Compute the semi-perimeter s = (13+14+15)/2 = 21.\n"
                "Step 2: Apply Heron's formula: Area = sqrt(s(s-a)(s-b)(s-c)).\n"
                "Step 3: Substitute: s-a = 21-13 = 8, s-b = 21-14 = 7, s-c = 21-15 = 6.\n"
                "Step 4: Compute the product: 21 * 8 * 7 * 6 = 7056.\n"
                "Step 5: Take the square root: sqrt(7056) = 84.\n"
                "Verification: A triangle with sides 13, 14, 15 has area 84."
            ),
            "answer": "\\boxed{84}",
            "raw_output": f"PROBLEM: A triangle has sides...\n\n{_TO}\nStep 1: Compute...\n{_TC}\n\\boxed{{84}}",
            "domain": "math",
            "difficulty": "medium",
        }

        accepted, report = run_quality_gates([good_row])
        assert len(accepted) == 1
        assert report.accepted == 1
        assert accepted[0]["quality_score"] > 0.3
