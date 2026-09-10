"""Model registry with hardware compatibility scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hardware import HardwareInfo


@dataclass
class ModelInfo:
    id: str
    name: str
    huggingface_id: str
    params_b: float  # billions of parameters
    # memory requirements in GB for different precisions
    fp16_gb: float
    fp32_gb: float
    int8_gb: float
    int4_gb: float
    family: str  # "deepseek-r1-distill", "qwen2.5", "llama3.2", etc.
    context_length: int = 4096
    backends: list[str] = field(default_factory=lambda: ["mlx", "vllm"])
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # When set, this model is a base-model + LoRA-adapter pair: the base is
    # downloaded from ``huggingface_id`` and the adapter from ``adapter_repo``.
    # On the MLX backend the adapter loads natively; on the vLLM/CUDA backend
    # it is transparently converted from MLX to PEFT format at load time.
    adapter_repo: Optional[str] = None


# ---------------------------------------------------------------------------
# Model registry — popular models for local IL pipelines
# ---------------------------------------------------------------------------

MODELS: list[ModelInfo] = [
    # --- DeepSeek R1 Distill (the IL pipeline's primary model) ---
    ModelInfo(
        id="deepseek-r1-distill-qwen-1.5b",
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        params_b=1.5,
        fp16_gb=3.5, fp32_gb=6.5, int8_gb=2.0, int4_gb=1.2,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="The original IL pipeline model. Reasoning-distilled, perfect for 8GB Macs.",
        tags=["reasoning", "recommended", "il-original"],
    ),
    ModelInfo(
        id="boostedv1-ilv9",
        name="BoostedV1-ILv9",
        huggingface_id="auryn-macmillan/boostedv1-ilv9",
        params_b=1.5,
        fp16_gb=3.5, fp32_gb=6.5, int8_gb=2.0, int4_gb=1.2,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="Continued LoRA training of DeepSeek-R1-Distill-Qwen-1.5B with 3 rounds of IL v9 self-improvement. HumanEval 7.3% -> 11.0%.",
        tags=["reasoning", "recommended", "il-improved"],
    ),
    ModelInfo(
        id="boosted-v1-small",
        name="Boosted-v1-small",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_repo="Akahsizrr/boosted-v1-small",
        params_b=1.5,
        fp16_gb=3.5, fp32_gb=6.5, int8_gb=2.0, int4_gb=1.2,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="MLX LoRA adapter for DeepSeek-R1-Distill-Qwen-1.5B trained via the iloptimus self-improvement pipeline on HumanEval v1. HumanEval 24.0% -> 70.88% (+46.88%).",
        tags=["reasoning", "recommended", "il-improved", "humaneval"],
    ),
    ModelInfo(
        id="deepseek-r1-distill-qwen-7b",
        name="DeepSeek-R1-Distill-Qwen-7B",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        params_b=7.0,
        fp16_gb=15.0, fp32_gb=28.0, int8_gb=8.0, int4_gb=4.5,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="Stronger reasoning, needs 16GB+ for 4-bit. Best on CUDA or M-series 16GB+.",
        tags=["reasoning"],
    ),
    ModelInfo(
        id="deepseek-r1-distill-llama-8b",
        name="DeepSeek-R1-Distill-Llama-8B",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        params_b=8.0,
        fp16_gb=17.0, fp32_gb=32.0, int8_gb=9.0, int4_gb=5.0,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="Llama-based R1 distill. Good reasoning, needs 16GB+ for 4-bit.",
        tags=["reasoning"],
    ),
    ModelInfo(
        id="deepseek-r1-distill-qwen-14b",
        name="DeepSeek-R1-Distill-Qwen-14B",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        params_b=14.0,
        fp16_gb=29.0, fp32_gb=56.0, int8_gb=15.0, int4_gb=8.5,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="Heavy-duty reasoning. Needs 24GB+ VRAM for 4-bit.",
        tags=["reasoning", "large"],
    ),
    # --- Qwen 2.5 ---
    ModelInfo(
        id="qwen2.5-0.5b",
        name="Qwen2.5-0.5B",
        huggingface_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.5,
        fp16_gb=1.2, fp32_gb=2.2, int8_gb=0.7, int4_gb=0.4,
        family="qwen2.5",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Ultra-light. Runs anywhere. Good for quick IL experiments.",
        tags=["lightweight", "recommended"],
    ),
    # --- OmniCoder-9B (the AIME 2025 model) ---
    ModelInfo(
        id="omnicoder-9b",
        name="OmniCoder-9B",
        huggingface_id="Tesslate/OmniCoder-9B",
        params_b=9.0,
        fp16_gb=19.0, fp32_gb=36.0, int8_gb=10.0, int4_gb=6.0,
        family="qwen3.5",
        context_length=131072,
        backends=["vllm"],
        description="Tesslate/OmniCoder-9B — Qwen3.5-based model trained with rank-256 rsLoRA on DistillMax debugging + refactoring datasets. Achieved 19/30 on AIME 2025 (pass@5) with checkpoint-50 adapter. The primary model for continued RL training toward general intelligence.",
        tags=["reasoning", "math", "aime", "large", "rl-target"],
        adapter_repo=None,  # adapter is local at E:\omnicoder-rl-candidates\continuation-50-files\checkpoint-50
    ),
    # --- merged_boosted_v1_small_r1_a100r1 (the 19/30 AIME 2025 model) ---
    ModelInfo(
        id="merged-boosted-r1-a100r1",
        name="merged_boosted_v1_small_r1_a100r1",
        huggingface_id="Akahsizrr/merged_boosted_v1_small_r1_a100r1",
        params_b=1.5,
        fp16_gb=3.5, fp32_gb=6.5, int8_gb=2.0, int4_gb=1.2,
        family="deepseek-r1-distill",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="DeepSeek-R1-Distill-Qwen-1.5B + Akahsizrr/boosted-v1-small (merged) + local RTX3060 round1 LoRA (merged) + A100 round1 LoRA rank-32 (merged). Achieved 19/30 (63.3%) on AIME 2025 — the best recorded score. Score progression: 5/30 baseline → 9/30 (local round1) → 15/30 (A100 pre-training) → 17/30 (A100 round1 post) → 19/30 (A100 round2 pre-training). The primary model for continued RL self-improvement.",
        tags=["reasoning", "math", "aime", "recommended", "rl-target", "self-improved"],
        adapter_repo=None,  # standalone merged checkpoint, no adapter needed
    ),
    ModelInfo(
        id="qwen2.5-1.5b",
        name="Qwen2.5-1.5B",
        huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.5,
        fp16_gb=3.5, fp32_gb=6.5, int8_gb=2.0, int4_gb=1.2,
        family="qwen2.5",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Small and fast. Great for 8GB Macs.",
        tags=["recommended"],
    ),
    ModelInfo(
        id="qwen2.5-3b",
        name="Qwen2.5-3B",
        huggingface_id="Qwen/Qwen2.5-3B-Instruct",
        params_b=3.0,
        fp16_gb=6.5, fp32_gb=12.0, int8_gb=3.5, int4_gb=2.0,
        family="qwen2.5",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Good balance. Fits in 8GB at 4-bit.",
        tags=["recommended"],
    ),
    ModelInfo(
        id="qwen2.5-7b",
        name="Qwen2.5-7B",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.0,
        fp16_gb=15.0, fp32_gb=28.0, int8_gb=8.0, int4_gb=4.5,
        family="qwen2.5",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Capable general model. Needs 16GB+ for 4-bit.",
        tags=[],
    ),
    ModelInfo(
        id="qwen2.5-14b",
        name="Qwen2.5-14B",
        huggingface_id="Qwen/Qwen2.5-14B-Instruct",
        params_b=14.0,
        fp16_gb=29.0, fp32_gb=56.0, int8_gb=15.0, int4_gb=8.5,
        family="qwen2.5",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Strong general model. Needs 24GB+ VRAM for 4-bit.",
        tags=["large"],
    ),
    # --- Llama 3.2 ---
    ModelInfo(
        id="llama-3.2-1b",
        name="Llama-3.2-1B",
        huggingface_id="meta-llama/Llama-3.2-1B-Instruct",
        params_b=1.0,
        fp16_gb=2.5, fp32_gb=4.5, int8_gb=1.5, int4_gb=0.8,
        family="llama3.2",
        context_length=128000,
        backends=["mlx", "vllm"],
        description="Meta's lightweight model. Great for 8GB Macs.",
        tags=["recommended"],
    ),
    ModelInfo(
        id="llama-3.2-3b",
        name="Llama-3.2-3B",
        huggingface_id="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.0,
        fp16_gb=6.5, fp32_gb=12.0, int8_gb=3.5, int4_gb=2.0,
        family="llama3.2",
        context_length=128000,
        backends=["mlx", "vllm"],
        description="Meta's small model. Fits in 8GB at 4-bit.",
        tags=["recommended"],
    ),
    # --- Phi-3.5 ---
    ModelInfo(
        id="phi-3.5-mini",
        name="Phi-3.5-mini",
        huggingface_id="microsoft/Phi-3.5-mini-instruct",
        params_b=3.8,
        fp16_gb=8.0, fp32_gb=15.0, int8_gb=4.5, int4_gb=2.5,
        family="phi",
        context_length=128000,
        backends=["mlx", "vllm"],
        description="Microsoft's small reasoning model. Good for coding tasks.",
        tags=[],
    ),
    # --- Mistral ---
    ModelInfo(
        id="mistral-7b",
        name="Mistral-7B-Instruct",
        huggingface_id="mistralai/Mistral-7B-Instruct-v0.3",
        params_b=7.0,
        fp16_gb=15.0, fp32_gb=28.0, int8_gb=8.0, int4_gb=4.5,
        family="mistral",
        context_length=32768,
        backends=["mlx", "vllm"],
        description="Solid general model. Needs 16GB+ for 4-bit.",
        tags=[],
    ),
    # --- GLM ---
    ModelInfo(
        id="glm-4-9b",
        name="GLM-4-9B",
        huggingface_id="THUDM/glm-4-9b-chat",
        params_b=9.0,
        fp16_gb=19.0, fp32_gb=36.0, int8_gb=10.0, int4_gb=5.5,
        family="glm",
        context_length=131072,
        backends=["mlx", "vllm"],
        description="Zhipu's model. Strong on coding and reasoning.",
        tags=[],
    ),
]


@dataclass
class CompatibilityResult:
    status: str  # "recommended", "feasible", "tight", "not-recommended"
    best_precision: str  # "fp16", "int8", "int4"
    best_precision_gb: float
    reason: str
    score: float  # 0.0 to 1.0


def check_compatibility(model: ModelInfo, hw: HardwareInfo) -> CompatibilityResult:
    """Check if a model fits on the user's hardware."""
    available = hw.total_memory_gb

    # Find the best precision that fits
    precisions = [
        ("fp16", model.fp16_gb),
        ("int8", model.int8_gb),
        ("int4", model.int4_gb),
    ]

    best = None
    for prec, mem in precisions:
        if mem <= available:
            best = (prec, mem)
            break

    if best is None:
        # Even 4-bit doesn't fit
        return CompatibilityResult(
            status="not-recommended",
            best_precision="int4",
            best_precision_gb=model.int4_gb,
            reason=f"Needs {model.int4_gb:.1f}GB (4-bit) but only {available:.1f}GB available",
            score=0.0,
        )

    prec, mem = best
    headroom = available - mem
    headroom_pct = headroom / available if available > 0 else 0

    if prec == "fp16" and headroom_pct > 0.3:
        status = "recommended"
        score = 1.0
        reason = f"Fits in {prec.upper()} ({mem:.1f}GB) with {headroom:.1f}GB headroom"
    elif prec in ("fp16", "int8") and headroom_pct > 0.15:
        status = "recommended"
        score = 0.9
        reason = f"Fits in {prec.upper()} ({mem:.1f}GB) with {headroom:.1f}GB headroom"
    elif headroom_pct > 0.05:
        status = "feasible"
        score = 0.7
        reason = f"Fits in {prec.upper()} ({mem:.1f}GB) — {headroom:.1f}GB headroom"
    else:
        status = "tight"
        score = 0.4
        reason = f"Tight fit in {prec.upper()} ({mem:.1f}GB) — only {headroom:.1f}GB headroom"

    # Backend check
    backend_ok = False
    if hw.recommended_backend == "mlx" and "mlx" in model.backends:
        backend_ok = True
    elif hw.recommended_backend == "vllm" and "vllm" in model.backends:
        backend_ok = True
    elif hw.recommended_backend == "cpu":
        backend_ok = True  # CPU can run anything slowly

    if not backend_ok and hw.recommended_backend != "cpu":
        score *= 0.5
        reason += f". Note: {hw.recommended_backend} backend not listed for this model"

    return CompatibilityResult(
        status=status,
        best_precision=prec,
        best_precision_gb=mem,
        reason=reason,
        score=score,
    )


def get_all_models() -> list[ModelInfo]:
    return MODELS


def get_model(model_id: str) -> Optional[ModelInfo]:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None
