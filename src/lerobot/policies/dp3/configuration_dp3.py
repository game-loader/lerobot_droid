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


@PreTrainedConfig.register_subclass("dp3")
@dataclass
class DP3Config(DiffusionConfig):
    """Diffusion Policy conditioned on a DP3-style point-cloud encoder.

    The policy consumes a fixed-size ``observation.point_cloud`` tensor in
    ``(num_points, channels)`` layout together with ``observation.state``.
    Point channels are XYZ, or XYZRGB when ``use_pc_color`` is enabled.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "POINT_CLOUD": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    point_cloud_key: str = OBS_POINT_CLOUD
    point_cloud_num_points: int | None = 512
    point_cloud_encoder_hidden_dims: tuple[int, ...] = (64, 128, 256)
    point_cloud_encoder_output_dim: int = 64
    state_encoder_hidden_dims: tuple[int, ...] = (64, 64)
    point_cloud_use_layer_norm: bool = True
    point_cloud_random_subsample: bool = True
    use_pc_color: bool = False

    @property
    def point_cloud_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(self.point_cloud_key)

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("DP3 requires an 'observation.state' input feature.")
        if len(self.robot_state_feature.shape) != 1:
            raise ValueError(
                f"DP3 expects 'observation.state' to be a vector. Got shape {self.robot_state_feature.shape}."
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
        if self.point_cloud_encoder_output_dim <= 0:
            raise ValueError("point_cloud_encoder_output_dim must be positive.")
        if not self.state_encoder_hidden_dims or any(dim <= 0 for dim in self.state_encoder_hidden_dims):
            raise ValueError("state_encoder_hidden_dims must contain positive dimensions.")
        if self.action_feature is None:
            raise ValueError("DP3 requires an 'action' output feature.")
