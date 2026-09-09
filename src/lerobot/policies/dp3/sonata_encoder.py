#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sonata (CVPR 2025 Highlight) encoder for DP3 with frozen pretrained weights.

Sonata is Meta & HKU's self-supervised point cloud transformer with 108.5M parameters.
It's an encoder-only PTv3 variant with superior performance on 3D perception tasks.

Architecture:
    - 5-stage encoder: [48, 96, 192, 384, 512] channels
    - Block depths: [3, 3, 3, 12, 3]
    - Encoder-only (decoder removed for SSL)
    - Output: 512-d per-point features at final stage
"""

import warnings

import torch
from torch import Tensor, nn


class SonataEncoder(nn.Module):
    """Frozen Sonata encoder for point cloud feature extraction.

    This encoder loads pretrained Sonata weights (CVPR 2025 Highlight) and freezes them,
    using only the forward pass to extract features. A trainable projection layer maps
    Sonata's 512-d output features to the desired dimension for the policy.

    Sonata is an encoder-only PTv3 variant with:
    - 108.5M parameters
    - 5 stages: [48, 96, 192, 384, 512] channels
    - Block depths: [3, 3, 3, 12, 3]
    - Superior SSL performance (72.5% on ScanNet linear probing)

    Reference:
        Sonata: Self-Supervised Learning of Reliable Point Representations
        Meta & HKU, CVPR 2025 Highlight
        https://github.com/facebookresearch/sonata
    """

    def __init__(
        self,
        config,
        sonata_model_path: str | None = None,
        sonata_feature_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        """Initialize Sonata encoder with frozen backbone.

        Args:
            config: DP3Config with point cloud settings
            sonata_model_path: Path to pretrained Sonata checkpoint (if None, uses placeholder)
            sonata_feature_dim: Output dimension of Sonata encoder (default 512)
            freeze_backbone: Whether to freeze Sonata weights (recommended: True)
        """
        super().__init__()
        if not config.experimental_allow_encoder_placeholders:
            raise NotImplementedError(
                "Sonata integration is unimplemented; this legacy encoder is an MLP placeholder."
            )
        if sonata_model_path:
            raise ValueError("Real Sonata weights cannot be loaded into the legacy MLP placeholder.")
        warnings.warn("Using an experimental MLP placeholder, not a Sonata Transformer.", stacklevel=2)

        point_cloud = config.point_cloud_feature
        if point_cloud is None:
            raise ValueError("DP3 point-cloud features must be validated before model construction.")

        self.num_points = config.point_cloud_num_points
        self.random_subsample = config.point_cloud_random_subsample
        self.sonata_feature_dim = sonata_feature_dim
        self.freeze_backbone = freeze_backbone

        # Sonata encoder: 5-stage encoder-only PTv3 variant (108.5M params)
        # Output: 512-d per-point features at final stage
        # TODO: Load actual Sonata model
        # For now, create a placeholder that mimics Sonata's 5-stage structure
        # In production, this should load from pretrained weights
        self.sonata_backbone = self._create_sonata_placeholder(point_cloud.shape[1], sonata_feature_dim)

        # Freeze Sonata backbone
        if self.freeze_backbone:
            for param in self.sonata_backbone.parameters():
                param.requires_grad = False
            self.sonata_backbone.eval()

        # Trainable projection layer: Sonata features (512-d) -> policy dimension
        self.output_dim = config.point_cloud_encoder_output_dim
        projection_layers: list[nn.Module] = [nn.Linear(sonata_feature_dim, self.output_dim)]
        if config.point_cloud_use_layer_norm:
            projection_layers.append(nn.LayerNorm(self.output_dim))
        self.projection = nn.Sequential(*projection_layers)

    def _create_sonata_placeholder(self, in_channels: int, out_dim: int) -> nn.Module:
        """Create a placeholder model that mimics Sonata's 5-stage encoder structure.

        This should be replaced with actual Sonata encoder loading.
        For now, it creates a simplified encoder that mimics Sonata's architecture:
        - 5-stage encoder with increasing channels: [48, 96, 192, 384, 512]
        - Mimics the encoder-only design (no decoder)
        - Output: per-point features at 512-d (final stage)

        In production, replace this with:
        ```python
        import sonata

        model = sonata.model.load("sonata")
        # Or from local checkpoint
        model = sonata.model.load(sonata_model_path)
        return model.backbone.encoder  # Only the encoder part
        ```
        """
        # Simplified Sonata-like 5-stage encoder:
        # Mimics: [48, 96, 192, 384, 512] channel progression
        # Input: (batch, N, in_channels) → Output: (batch, N, out_dim)
        return nn.Sequential(
            # Stage 1: 48 channels (3 blocks)
            nn.Linear(in_channels, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            # Stage 2: 96 channels (3 blocks)
            nn.Linear(48, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            # Stage 3: 192 channels (3 blocks)
            nn.Linear(96, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            # Stage 4: 384 channels (12 blocks - deepest)
            nn.Linear(192, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            # Stage 5: 512 channels (3 blocks - final encoder output)
            nn.Linear(384, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def _subsample(self, points: Tensor) -> Tensor:
        """Subsample point cloud to fixed size."""
        if self.num_points is None or points.shape[-2] == self.num_points:
            return points
        if points.shape[-2] < self.num_points:
            raise ValueError(
                f"Sonata encoder received {points.shape[-2]} points, fewer than configured {self.num_points}."
            )

        if self.training and self.random_subsample and not self.freeze_backbone:
            indices = torch.randperm(points.shape[-2], device=points.device)[: self.num_points]
        else:
            # Deterministic subsampling for frozen backbone
            indices = (
                torch.linspace(
                    0,
                    points.shape[-2] - 1,
                    steps=self.num_points,
                    device=points.device,
                )
                .round()
                .long()
            )
        return points.index_select(-2, indices)

    def forward(self, points: Tensor) -> Tensor:
        """Extract features from point cloud using frozen Sonata encoder.

        Args:
            points: Point cloud tensor (batch, num_points, channels)
                    e.g., (8, 2048, 3) for XYZ-only point clouds

        Returns:
            Projected features (batch, output_dim)
                    e.g., (8, 256) after projection

        Pipeline:
            1. Subsample to fixed number of points (e.g., 2048)
            2. Sonata encoder: (B, N, 3) → (B, N, 512) per-point features
               - 5 stages: [48, 96, 192, 384, 512] channels
               - Encoder-only design (no decoder)
            3. Global max pooling: (B, N, 512) → (B, 512)
            4. Trainable projection: (B, 512) → (B, output_dim)
        """
        if points.ndim != 3:
            raise ValueError(f"Sonata encoder expects (batch, points, channels), got {tuple(points.shape)}.")

        # Subsample points to fixed size
        points = self._subsample(points)

        # Handle NaN/Inf values
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract per-point features with frozen Sonata encoder
        with torch.set_grad_enabled(not self.freeze_backbone):
            # Sonata encoder forward pass
            # Input: (batch, num_points, in_channels)
            # Output: (batch, num_points, sonata_feature_dim) - per-point features at final stage
            per_point_features = self.sonata_backbone(points)  # (B, N, 512)

            # Global aggregation: max pooling across all points
            # This gives us a single global feature vector per point cloud
            pooled_features = per_point_features.amax(dim=1)  # (B, 512)

        # Project to policy dimension (trainable projection layer)
        # This is the only trainable part when freeze_backbone=True
        projected_features = self.projection(pooled_features)  # (B, output_dim)

        return projected_features

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.sonata_backbone.eval()
        return self
