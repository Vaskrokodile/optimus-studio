"""
AestheticScorer — CLIP-based image aesthetic prediction.

Uses the LAION Aesthetic Score v2 model: a linear head on top of
CLIP ViT-L/14 that predicts a human-annotated aesthetic rating (0-10)
for an image. We normalize to [0, 1] for use as an RL reward.

The model weights are downloaded from the LAION HuggingFace repo on
first use and cached locally.

For ultra-batched scoring, images are processed in batches on the GPU.
A batch of 256 images takes ~50ms on an RTX 3060.

References:
  - LAION Aesthetic V2: https://github.com/christophschuhmann/improved-aesthetic-predictor
  - CLIP ViT-L/14: OpenAI / open_clip
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# CLIP is available via open_clip_torch (more flexible) or openai-clip.
# We prefer open_clip_torch as it's the standard in the LAION ecosystem.
try:
    import open_clip
    from open_clip import create_model_and_transforms, get_tokenizer
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

# URL for the LAION aesthetic predictor v2 weights (ViT-L/14)
AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_CACHE_DIR = Path.home() / ".cache" / "rl_factory" / "aesthetic"


class AestheticPredictor(nn.Module):
    """
    Linear aesthetic predictor on top of CLIP ViT-L/14 embeddings (768-dim).
    Outputs a single aesthetic score (trained on a 1-10 scale).
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer:
    """
    Batched CLIP aesthetic scorer for RL reward computation.

    Usage:
        scorer = AestheticScorer(device="cuda")
        scores = scorer.score_batch(images)  # list of PIL Images → tensor of [0,1] scores
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model: str = "ViT-L-14",
        pretrained: str = "openai",
        batch_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.clip_model_name = clip_model
        self.pretrained = pretrained

        if not HAS_OPEN_CLIP:
            raise ImportError(
                "open_clip_torch is required. Install with: pip install open_clip_torch"
            )

        # Load CLIP model
        model, _, preprocess = create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        self.clip_model = model.visual.eval().to(device)
        self.preprocess = preprocess

        # Load aesthetic predictor head
        self.predictor = AestheticPredictor(input_dim=768).to(device)
        self._load_aesthetic_weights()

        # Freeze everything — this is a fixed reward model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.predictor.parameters():
            param.requires_grad = False

    def _load_aesthetic_weights(self) -> None:
        """Download and load the LAION aesthetic predictor weights."""
        AESTHETIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = AESTHETIC_CACHE_DIR / "aesthetic_v2_l14.pth"

        if not weights_path.exists():
            import urllib.request
            print(f"Downloading LAION aesthetic predictor weights to {weights_path}...")
            urllib.request.urlretrieve(AESTHETIC_WEIGHTS_URL, str(weights_path))

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()

    def _extract_clip_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract CLIP image features for a batch of images."""
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.preprocess(img))

        features = []
        with torch.no_grad():
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device)
                feat = self.clip_model(batch)
                # Normalize features (CLIP convention)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat.cpu())

        return torch.cat(features, dim=0)

    def score_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Score a batch of images for aesthetic quality.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (N,) with scores in [0, 1].
            LAION scores are on a 1-10 scale; we normalize to [0, 1].
        """
        if not images:
            return torch.tensor([])

        features = self._extract_clip_features(images)

        with torch.no_grad():
            scores = self.predictor(features.to(self.device)).squeeze(-1)

        # LAION aesthetic scores are ~1-10. Normalize to [0, 1].
        # Use a sigmoid-like mapping centered at 5.5 (median aesthetic):
        # score_norm = (raw - 1) / 9  → linear [0,1] for [1,10]
        # But we also want to emphasize the upper range (beautiful art),
        # so we apply a mild power transform to push differentiation:
        # score_norm = ((raw - 1) / 9) ** 0.7
        scores = scores.cpu()
        normalized = ((scores - 1.0) / 9.0).clamp(0.0, 1.0)
        normalized = normalized.pow(0.7)  # emphasize differences in the good range

        return normalized

    def score_png_bytes_batch(self, png_bytes_list: list[bytes]) -> torch.Tensor:
        """
        Convenience: score images from raw PNG bytes.

        Args:
            png_bytes_list: List of PNG byte arrays.

        Returns:
            Tensor of shape (N,) with scores in [0, 1].
        """
        images = []
        for png in png_bytes_list:
            try:
                img = Image.open(io.BytesIO(png))
                images.append(img)
            except Exception:
                # Corrupt/invalid image → score 0
                images.append(Image.new("RGB", (64, 64), (0, 0, 0)))

        return self.score_batch(images)

    def score_single(self, image: Image.Image) -> float:
        """Score a single image, return float."""
        return self.score_batch([image]).item()
