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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.constants import OBS_POINT_CLOUD

DP3_STATE_DIM = 34
DP3_ACTION_DIM = 20


@PreTrainedConfig.register_subclass("dp3")
@dataclass
class DP3Config(DiffusionConfig):
    """Diffusion Policy conditioned on point cloud, optional wrist RGB, and state.

    The dual-wrist variant is intended for real Franka Duo data collected by
    the local recorder: ZED Mini depth is converted to the point cloud and the
    two D405 wrist RGB streams provide the image inputs. Setting
    ``wrist_image_keys=()`` switches to a point-cloud-only model whose
    observation encoder drops every ResNet-18 wrist encoder and conditions only
    on the point cloud plus robot state.

    The policy consumes a fixed-size ``observation.point_cloud`` tensor in
    ``(num_points, channels)`` layout, zero or more wrist RGB observations, and
    a 34-dimensional ``observation.state``. Point channels are XYZ, or XYZRGB
    when ``use_pc_color`` is enabled. The policy predicts 20-dimensional
    actions (dual-arm XYZ+rot6d targets plus two gripper values).

    The action denoiser is selected by ``diffusion_backbone``: the LeRobot
    1D conv U-Net (default) or a conventional Transformer Diffusion whose
    observation conditioning is a prefix / joint self-attention mask.
    """

    # Shorter real-robot action planning window: predict 32 actions and
    # execute the first 16 before replanning.
    horizon: int = 32
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "POINT_CLOUD": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    point_cloud_key: str = OBS_POINT_CLOUD
    point_cloud_num_points: int | None = 2048
    point_cloud_encoder_hidden_dims: tuple[int, ...] = (64, 128, 256)
    point_cloud_encoder_output_dim: int = 64
    # When True, the point cloud is projected from the pooled MLP width (256 by
    # default) down to ``point_cloud_encoder_output_dim`` before fusion. When
    # False, the pooled point feature (256 dims) is fused directly with the
    # state encoder, skipping the projection Linear+LayerNorm.
    point_cloud_use_projection: bool = True
    state_encoder_hidden_dims: tuple[int, ...] = (64, 64)
    point_cloud_use_layer_norm: bool = True
    # IL uses deterministic sensor inputs: never randomly subsample the
    # configured point cloud during training. Inputs larger than 2048 points
    # use the deterministic fallback in PointNetEncoder.
    point_cloud_random_subsample: bool = False
    use_pc_color: bool = False
    # Legacy encoder experiments are MLP placeholders, not pretrained Transformers.
    experimental_allow_encoder_placeholders: bool = False

    # PTv3 encoder settings
    use_ptv3_encoder: bool = False
    ptv3_model_path: str | None = None
    ptv3_feature_dim: int = 512  # PTv3 encoder bottleneck dimension (Stage 4)
    ptv3_freeze_backbone: bool = True

    # Sonata encoder settings (CVPR 2025 Highlight, Meta & HKU)
    use_sonata_encoder: bool = False
    sonata_model_path: str | None = None
    sonata_feature_dim: int = 512  # Sonata encoder output dimension (Stage 5)
    sonata_freeze_backbone: bool = True

    wrist_image_keys: tuple[str, ...] = (
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    )
    expected_state_dim: int = DP3_STATE_DIM
    expected_action_dim: int = DP3_ACTION_DIM

    # Diffusion denoiser backbone. ``"unet"`` keeps the LeRobot 1D conv U-Net
    # (FiLM-conditioned on the global observation vector). ``"transformer"``
    # swaps in a conventional Transformer Diffusion (standard DDPM/DDIM
    # epsilon prediction) whose observation conditioning is a prefix / joint
    # self-attention: observation + timestep tokens form a prefix, action
    # tokens attend to that prefix and to each other, and the prefix never
    # attends to actions. Conditioning enters only through the attention mask
    # (no AdaLN / FiLM).
    diffusion_backbone: str = "unet"
    transformer_hidden_dim: int = 512
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    transformer_dropout: float = 0.1
    transformer_timestep_embed_dim: int = 256
    transformer_use_rope: bool = True
    transformer_rope_base: float = 10000.0

    # The target model trains one independent ResNet18 per wrist from scratch.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    use_separate_rgb_encoder_per_camera: bool = True
    # Keep image conditioning deterministic in both training and evaluation.
    # (The generic DiffusionConfig defaults to random training crops.)
    crop_is_random: bool = False

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Return only the two wrist images consumed by the policy.

        A source dataset may retain unrelated cameras for provenance or other
        policies. They are intentionally excluded from DP3 conditioning.
        """

        if not self.input_features:
            return {}
        return {
            key: self.input_features[key]
            for key in self.wrist_image_keys
            if key in self.input_features and self.input_features[key].type is FeatureType.VISUAL
        }

    @property
    def point_cloud_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(self.point_cloud_key)

    def validate_features(self) -> None:
        if self.use_ptv3_encoder and self.use_sonata_encoder:
            raise ValueError("Select only one of use_ptv3_encoder and use_sonata_encoder.")
        if (
            self.use_ptv3_encoder or self.use_sonata_encoder
        ) and not self.experimental_allow_encoder_placeholders:
            raise ValueError(
                "PTv3/Sonata integration is unimplemented: legacy encoders are MLP placeholders, "
                "not pretrained Transformers. Use PointNet or explicitly enable "
                "experimental_allow_encoder_placeholders for legacy experiments only."
            )
        if self.robot_state_feature is None:
            raise ValueError("DP3 requires an 'observation.state' input feature.")
        if len(self.robot_state_feature.shape) != 1:
            raise ValueError(
                f"DP3 expects 'observation.state' to be a vector. Got shape {self.robot_state_feature.shape}."
            )
        if self.expected_state_dim != DP3_STATE_DIM:
            raise ValueError(f"DP3 state dimension is fixed at {DP3_STATE_DIM}.")
        if self.robot_state_feature.shape != (self.expected_state_dim,):
            raise ValueError(
                f"DP3 expects a {self.expected_state_dim}-dimensional 'observation.state'. "
                f"Got shape {self.robot_state_feature.shape}."
            )

        point_cloud = self.point_cloud_feature
        if point_cloud is None:
            raise ValueError(f"DP3 requires a '{self.point_cloud_key}' input feature.")
        if point_cloud.type is not FeatureType.POINT_CLOUD:
            raise ValueError(
                f"'{self.point_cloud_key}' must use feature type POINT_CLOUD. Got {point_cloud.type.value}."
            )
        if len(point_cloud.shape) != 2:
            raise ValueError(
                f"DP3 expects '{self.point_cloud_key}' in (num_points, channels) layout. "
                f"Got shape {point_cloud.shape}."
            )

        expected_channels = 6 if self.use_pc_color else 3
        if point_cloud.shape[1] != expected_channels:
            raise ValueError(
                f"DP3 expects {expected_channels} point channels when use_pc_color={self.use_pc_color}. "
                f"Got shape {point_cloud.shape}."
            )
        if self.point_cloud_num_points is not None:
            if self.point_cloud_num_points <= 0:
                raise ValueError("point_cloud_num_points must be positive or None.")
            if self.point_cloud_num_points > point_cloud.shape[0]:
                raise ValueError(
                    "point_cloud_num_points cannot exceed the point-cloud feature size. "
                    f"Got {self.point_cloud_num_points} > {point_cloud.shape[0]}."
                )

        if not self.point_cloud_encoder_hidden_dims or any(
            dim <= 0 for dim in self.point_cloud_encoder_hidden_dims
        ):
            raise ValueError("point_cloud_encoder_hidden_dims must contain positive dimensions.")
        if self.point_cloud_use_projection and self.point_cloud_encoder_output_dim <= 0:
            raise ValueError("point_cloud_encoder_output_dim must be positive when projection is enabled.")
        if not self.state_encoder_hidden_dims or any(dim <= 0 for dim in self.state_encoder_hidden_dims):
            raise ValueError("state_encoder_hidden_dims must contain positive dimensions.")

        if self.diffusion_backbone not in ("unet", "transformer"):
            raise ValueError(
                f"diffusion_backbone must be 'unet' or 'transformer'. Got {self.diffusion_backbone!r}."
            )
        if self.diffusion_backbone == "transformer":
            for name in ("transformer_hidden_dim", "transformer_num_layers", "transformer_num_heads"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer. Got {value!r}.")
            if self.transformer_hidden_dim % self.transformer_num_heads != 0:
                raise ValueError(
                    "transformer_hidden_dim must be divisible by transformer_num_heads. "
                    f"Got {self.transformer_hidden_dim} and {self.transformer_num_heads}."
                )
            if self.transformer_use_rope and (self.transformer_hidden_dim // self.transformer_num_heads) % 2:
                raise ValueError("RoPE requires an even head dimension.")
            if not (0.0 <= self.transformer_dropout <= 1.0):
                raise ValueError(f"transformer_dropout must be in [0, 1]. Got {self.transformer_dropout}.")
            if self.transformer_timestep_embed_dim <= 0 or self.transformer_timestep_embed_dim % 2:
                raise ValueError("transformer_timestep_embed_dim must be a positive even integer.")
            if self.transformer_rope_base <= 0:
                raise ValueError("transformer_rope_base must be positive.")

        if not self.input_features:
            raise ValueError("DP3 input_features must be configured before model construction.")
        if self.wrist_image_keys:
            if len(self.wrist_image_keys) != 2 or len(set(self.wrist_image_keys)) != 2:
                raise ValueError(
                    "wrist_image_keys must contain exactly two distinct feature keys, "
                    "or be empty for a point-cloud-only model."
                )
            for key in self.wrist_image_keys:
                image = self.input_features.get(key)
                if image is None:
                    raise ValueError(f"DP3 requires wrist RGB feature '{key}'.")
                if image.type is not FeatureType.VISUAL:
                    raise ValueError(f"DP3 wrist feature '{key}' must use feature type VISUAL.")
                if len(image.shape) != 3 or image.shape[0] != 3:
                    raise ValueError(
                        f"DP3 wrist RGB feature '{key}' must have (3, height, width) shape. Got {image.shape}."
                    )

        if self.vision_backbone != "resnet18":
            raise ValueError("DP3 wrist RGB encoders must use vision_backbone='resnet18'.")
        if self.pretrained_backbone_weights is not None:
            raise ValueError("DP3 wrist ResNet18 encoders must be initialized without pretrained weights.")
        if not self.use_separate_rgb_encoder_per_camera:
            raise ValueError("DP3 requires a separate ResNet18 encoder for each wrist camera.")
        if self.crop_is_random:
            raise ValueError("DP3 does not use random image crops; set crop_is_random=False.")
        if self.point_cloud_random_subsample:
            raise ValueError(
                "DP3 does not use random point-cloud subsampling; set point_cloud_random_subsample=False."
            )

        # Reuse Diffusion validation for crop bounds and matching wrist image shapes.
        # With wrist_image_keys empty, ``image_features`` is empty so the base
        # diffusion validation has no visuomotor image constraints to check.
        super().validate_features()

        if self.action_feature is None:
            raise ValueError("DP3 requires an 'action' output feature.")
        if self.expected_action_dim != DP3_ACTION_DIM:
            raise ValueError(f"DP3 action dimension is fixed at {DP3_ACTION_DIM}.")
        if self.action_feature.shape != (self.expected_action_dim,):
            raise ValueError(
                f"DP3 expects a {self.expected_action_dim}-dimensional 'action'. "
                f"Got shape {self.action_feature.shape}."
            )
