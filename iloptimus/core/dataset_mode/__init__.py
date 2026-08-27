"""Dataset mode: unified dataset building for Optimus Studio.

When the loop enters dataset mode, it spawns parallel vLLM generation sessions
with hardened system prompts, template exemplars, anti-slop/anti-laziness
skills, continuous web research, and quality gates.

Layers:
1. Orchestrator — DatasetModeSession, enter_dataset_mode()
2. Prompts — dataset-mode system prompts (math, coding, reasoning, agentic)
3. Templates — exemplar rows from Kimi K3, Fable 5, Qwen 3
4. Skills — anti-slop, anti-laziness, curation, contamination defense
5. Research — continuous, rate-limited, coverage-aware web search
6. Quality — structural validation, semantic dedup, contamination, slop, scoring
7. Manifest — auto-generated dataset card with coverage audit

Plus source acquisition (web retrieval) migrated from dataset_factory.
"""

from .generation import GeneratedRow, GenerationSpec, generate_batch, reroll_failed
from .manifest import DatasetManifest, build_manifest, manifest_summary, save_manifest
from .orchestrator import (
    DatasetModeResult,
    DatasetModeSpec,
    build_curriculum_spec,
    build_failure_driven_spec,
    enter_dataset_mode,
    get_session as get_dataset_session,
    list_sessions as list_dataset_sessions,
    load_dataset as load_dataset_rows,
)
from .prompts import SUPPORTED_DOMAINS, dataset_system_prompt
from .quality import QualityReport, QualityResult, run_quality_gates, score_quality
from .research import ResearchCorpus, ResearchSpec, continuous_research, research_summary
from .skills import dataset_skills_prompt, list_dataset_skills
from .source_acquisition import harvest_urls
from .templates import TemplateRow, load_templates, sample_templates, template_prompt

__all__ = [
    # Orchestrator
    "DatasetModeSpec", "DatasetModeResult", "enter_dataset_mode",
    "build_failure_driven_spec", "build_curriculum_spec",
    "list_dataset_sessions", "get_dataset_session", "load_dataset_rows",
    # Prompts
    "dataset_system_prompt", "SUPPORTED_DOMAINS",
    # Skills
    "dataset_skills_prompt", "list_dataset_skills",
    # Templates
    "load_templates", "sample_templates", "template_prompt", "TemplateRow",
    # Research
    "ResearchSpec", "ResearchCorpus", "continuous_research", "research_summary",
    # Generation
    "GenerationSpec", "GeneratedRow", "generate_batch", "reroll_failed",
    # Quality
    "QualityResult", "QualityReport", "run_quality_gates", "score_quality",
    # Manifest
    "DatasetManifest", "build_manifest", "save_manifest", "manifest_summary",
    # Source acquisition
    "harvest_urls",
]
