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

"""Point Transformer V3 encoder for DP3 with frozen pretrained weights."""

import warnings

import torch
from torch import Tensor, nn


class PTv3Encoder(nn.Module):
    """Frozen Point Transformer V3 encoder for point cloud feature extraction.

    This encoder loads pretrained PTv3 weights and freezes them, using only
    the forward pass to extract features. A trainable projection layer maps
    PTv3's output features to the desired dimension for the policy.

    Expected PTv3 output: (batch, feature_dim) after global pooling
    This module adds: Linear projection to policy's expected dimension
    """

    def __init__(
        self,
        config,
        ptv3_model_path: str | None = None,
        ptv3_feature_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        """Initialize PTv3 encoder with frozen backbone.

        Args:
            config: DP3Config with point cloud settings
            ptv3_model_path: Path to pretrained PTv3 checkpoint (if None, uses placeholder)
            ptv3_feature_dim: Output dimension of PTv3 encoder bottleneck (default 512)
            freeze_backbone: Whether to freeze PTv3 weights
        """
        super().__init__()
        if not config.experimental_allow_encoder_placeholders:
            raise NotImplementedError(
                "PTv3 integration is unimplemented; this legacy encoder is an MLP placeholder."
            )
        if ptv3_model_path:
            raise ValueError("Real PTv3 weights cannot be loaded into the legacy MLP placeholder.")
        warnings.warn("Using an experimental MLP placeholder, not a PTv3 Transformer.", stacklevel=2)

        point_cloud = config.point_cloud_feature
        if point_cloud is None:
            raise ValueError("DP3 point-cloud features must be validated before model construction.")

        self.num_points = config.point_cloud_num_points
        self.random_subsample = config.point_cloud_random_subsample
        self.ptv3_feature_dim = ptv3_feature_dim
        self.freeze_backbone = freeze_backbone

        # PTv3 encoder: extracts bottleneck features (512-d per-point by default)
        # We use only the encoder, not the full encoder-decoder
        # TODO: Load actual PTv3 encoder
        # For now, create a placeholder that mimics PTv3 encoder structure
        # In production, this should load from pretrained weights
        self.ptv3_backbone = self._create_ptv3_placeholder(point_cloud.shape[1], ptv3_feature_dim)

        # Freeze PTv3 backbone
        if self.freeze_backbone:
            for param in self.ptv3_backbone.parameters():
                param.requires_grad = False
            self.ptv3_backbone.eval()

        # Trainable projection layer: PTv3 bottleneck features (512-d) -> policy dimension
        self.output_dim = config.point_cloud_encoder_output_dim
        projection_layers: list[nn.Module] = [nn.Linear(ptv3_feature_dim, self.output_dim)]
        if config.point_cloud_use_layer_norm:
            projection_layers.append(nn.LayerNorm(self.output_dim))
        self.projection = nn.Sequential(*projection_layers)

    def _create_ptv3_placeholder(self, in_channels: int, out_dim: int) -> nn.Module:
        """Create a placeholder model that mimics PTv3 encoder structure.

        This should be replaced with actual PTv3 encoder loading.
        For now, it creates a simplified encoder that mimics PTv3's architecture:
        - Per-point feature extraction through multiple stages
        - Mimics the encoder bottleneck (64 → 128 → 256 → 512)
        - Output: per-point features at the specified dimension (default 512)

        In production, replace this with:
        ```python
        from pointcept.models import build_model

        model = build_model(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        return model.encoder  # Only the encoder part
        ```
        """
        # Simplified PTv3-like encoder:
        # Mimics the 4-stage encoder with increasing channels
        # Input: (batch, N, in_channels) → Output: (batch, N, out_dim)
        return nn.Sequential(
            # Stage 1: 64 channels
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            # Stage 2: 128 channels
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # Stage 3: 256 channels
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            # Stage 4: 512 channels (encoder bottleneck)
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def _subsample(self, points: Tensor) -> Tensor:
        """Subsample point cloud to fixed size."""
        if self.num_points is None or points.shape[-2] == self.num_points:
            return points
        if points.shape[-2] < self.num_points:
            raise ValueError(
                f"PTv3 encoder received {points.shape[-2]} points, fewer than configured {self.num_points}."
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
        """Extract features from point cloud using frozen PTv3 encoder.

        Args:
            points: Point cloud tensor (batch, num_points, channels)
                    e.g., (8, 2048, 3) for XYZ-only point clouds

        Returns:
            Projected features (batch, output_dim)
                    e.g., (8, 256) after projection

        Pipeline:
            1. Subsample to fixed number of points (e.g., 2048)
            2. PTv3 encoder: (B, N, 3) → (B, N, 512) per-point bottleneck features
            3. Global max pooling: (B, N, 512) → (B, 512)
            4. Trainable projection: (B, 512) → (B, output_dim)
        """
        if points.ndim != 3:
            raise ValueError(f"PTv3 encoder expects (batch, points, channels), got {tuple(points.shape)}.")

        # Subsample points to fixed size
        points = self._subsample(points)

        # Handle NaN/Inf values
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract per-point features with frozen PTv3 encoder
        with torch.set_grad_enabled(not self.freeze_backbone):
            # PTv3 encoder forward pass
            # Input: (batch, num_points, in_channels)
            # Output: (batch, num_points, ptv3_feature_dim) - per-point bottleneck features
            per_point_features = self.ptv3_backbone(points)  # (B, N, 512)

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
            self.ptv3_backbone.eval()
        return self
