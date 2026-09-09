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

"""LeRobot-native point-cloud (optionally dual-wrist RGB) Diffusion Policy.

The point encoder follows the RL-100/DP3 structure: shared per-point MLP,
global max pooling, and an optional projection layer. Its feature is
concatenated with zero or two randomly initialized ResNet18 wrist features and
encoded robot state. The action denoiser is either the LeRobot Diffusion 1D
U-Net or a prefix-attention Transformer Diffusion (``diffusion_backbone``);
the checkpoint contract reuses LeRobot Diffusion. The intended deployment
source is the real Franka Duo recorder's ZED depth sidecar plus optional
left/right wrist RGB streams.
"""

from collections import deque

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
    resize_images_for_stacking,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_dp3 import DP3Config
from .modeling_dp3_transformer import DP3DiffusionTransformer
from .ptv3_encoder import PTv3Encoder
from .sonata_encoder import SonataEncoder


class PointNetEncoder(nn.Module):
    """Encode an unordered XYZ or XYZRGB point set with global max pooling."""

    def __init__(self, config: DP3Config):
        super().__init__()
        point_cloud = config.point_cloud_feature
        if point_cloud is None:
            raise ValueError("DP3 point-cloud features must be validated before model construction.")

        layers: list[nn.Module] = []
        in_dim = point_cloud.shape[1]
        for out_dim in config.point_cloud_encoder_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if config.point_cloud_use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.point_mlp = nn.Sequential(*layers)
        if config.point_cloud_use_projection:
            projection: list[nn.Module] = [nn.Linear(in_dim, config.point_cloud_encoder_output_dim)]
            if config.point_cloud_use_layer_norm:
                projection.append(nn.LayerNorm(config.point_cloud_encoder_output_dim))
            self.projection = nn.Sequential(*projection)
            self.output_dim = config.point_cloud_encoder_output_dim
        else:
            self.projection = None
            self.output_dim = in_dim
        self.num_points = config.point_cloud_num_points
        self.random_subsample = config.point_cloud_random_subsample

    def _subsample(self, points: Tensor) -> Tensor:
        if self.num_points is None or points.shape[-2] == self.num_points:
            return points
        if points.shape[-2] < self.num_points:
            raise ValueError(
                f"DP3 received {points.shape[-2]} points, fewer than configured {self.num_points}."
            )

        if self.training and self.random_subsample:
            indices = torch.randperm(points.shape[-2], device=points.device)[: self.num_points]
        else:
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
        if points.ndim != 3:
            raise ValueError(f"DP3 PointNet expects (batch, points, channels), got {tuple(points.shape)}.")
        points = self._subsample(points)
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        point_features = self.point_mlp(points)
        pooled_features = point_features.amax(dim=1)
        if self.projection is None:
            return pooled_features
        return self.projection(pooled_features)


class StateEncoder(nn.Module):
    """Encode proprioception before fusing it with the point-cloud feature."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for index, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if index != len(hidden_dims) - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        return self.mlp(state)


class DP3ObservationEncoder(nn.Module):
    """Fuse RL-100 PointNet, dual-wrist ResNet18, and robot state features."""

    def __init__(self, config: DP3Config):
        super().__init__()
        state_feature = config.robot_state_feature
        if state_feature is None:
            raise ValueError("DP3 state features must be validated before model construction.")

        # Choose point cloud encoder: Sonata > PTv3 > PointNet
        if config.use_sonata_encoder:
            self.point_net = SonataEncoder(
                config,
                sonata_model_path=config.sonata_model_path,
                sonata_feature_dim=config.sonata_feature_dim,
                freeze_backbone=config.sonata_freeze_backbone,
            )
        elif config.use_ptv3_encoder:
            self.point_net = PTv3Encoder(
                config,
                ptv3_model_path=config.ptv3_model_path,
                ptv3_feature_dim=config.ptv3_feature_dim,
                freeze_backbone=config.ptv3_freeze_backbone,
            )
        else:
            self.point_net = PointNetEncoder(config)
        self.rgb_encoders = (
            nn.ModuleList(DiffusionRgbEncoder(config) for _ in config.wrist_image_keys)
            if config.wrist_image_keys
            else nn.ModuleList()
        )
        self.state_encoder = StateEncoder(state_feature.shape[0], config.state_encoder_hidden_dims)
        self.output_dim = (
            self.point_net.output_dim
            + sum(encoder.feature_dim for encoder in self.rgb_encoders)
            + config.state_encoder_hidden_dims[-1]
        )

    def forward(self, state: Tensor, point_cloud: Tensor, wrist_images: Tensor | None = None) -> Tensor:
        if state.ndim != 2:
            raise ValueError(f"DP3 state encoder expects (batch, state_dim), got {tuple(state.shape)}.")
        if state.shape[0] != point_cloud.shape[0]:
            raise ValueError("DP3 state and point-cloud batch dimensions must match.")

        features = [self.point_net(point_cloud)]
        if self.rgb_encoders:
            if wrist_images is None or wrist_images.ndim != 5:
                raise ValueError(
                    "DP3 wrist encoder expects (batch, cameras, channels, height, width), "
                    f"got {None if wrist_images is None else tuple(wrist_images.shape)}."
                )
            if wrist_images.shape[1] != len(self.rgb_encoders):
                raise ValueError(
                    f"DP3 expects {len(self.rgb_encoders)} wrist RGB views, got {wrist_images.shape[1]}."
                )
            if state.shape[0] != wrist_images.shape[0]:
                raise ValueError("DP3 state and wrist RGB batch dimensions must match.")
            features.extend(
                encoder(wrist_images[:, index]) for index, encoder in enumerate(self.rgb_encoders)
            )
        features.append(self.state_encoder(state))
        return torch.cat(features, dim=-1)


class DP3DiffusionModel(DiffusionModel):
    """LeRobot Diffusion denoiser with DP3 observation conditioning.

    The denoiser stored at ``self.unet`` is either the LeRobot 1D conv U-Net
    or the prefix-attention Transformer Diffusion, selected by
    ``config.diffusion_backbone``. Both share the
    ``forward(x, timestep, global_cond)`` contract, so ``compute_loss``,
    ``conditional_sample`` and the RL adapters are backbone-agnostic.
    """

    def __init__(self, config: DP3Config):
        nn.Module.__init__(self)
        self.config = config
        self.observation_encoder = DP3ObservationEncoder(config)
        global_cond_dim = self.observation_encoder.output_dim * config.n_obs_steps
        if config.diffusion_backbone == "transformer":
            self.unet = DP3DiffusionTransformer(config, global_cond_dim=global_cond_dim)
        else:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        point_cloud_key = self.config.point_cloud_key
        required = (OBS_STATE, point_cloud_key, *self.config.wrist_image_keys)
        missing = [key for key in required if key not in batch]
        if missing and (OBS_IMAGES not in batch or not self.config.wrist_image_keys):
            raise ValueError(f"DP3 batch is missing required feature(s): {missing}.")
        if OBS_STATE not in batch or point_cloud_key not in batch:
            missing_nonvisual = [key for key in (OBS_STATE, point_cloud_key) if key not in batch]
            raise ValueError(f"DP3 batch is missing required feature(s): {missing_nonvisual}.")

        if OBS_IMAGES not in batch:
            batch = resize_images_for_stacking(batch, self.config)
        state = batch[OBS_STATE]
        point_cloud = batch[point_cloud_key]
        if state.ndim != 3:
            raise ValueError(
                f"DP3 expects batched observation history for '{OBS_STATE}', got {tuple(state.shape)}."
            )
        if point_cloud.ndim != 4:
            raise ValueError(
                "DP3 expects point-cloud history in (batch, obs_steps, points, channels) layout. "
                f"Got {tuple(point_cloud.shape)}."
            )
        if state.shape[:2] != point_cloud.shape[:2]:
            raise ValueError("DP3 state and point-cloud history dimensions must match.")

        batch_size, n_obs_steps = state.shape[:2]
        if self.config.wrist_image_keys:
            if OBS_IMAGES in batch:
                wrist_images = batch[OBS_IMAGES]
            else:
                image_histories: list[Tensor] = []
                for key in self.config.wrist_image_keys:
                    image = batch[key]
                    if self.config.n_obs_steps == 1 and image.ndim == 4:
                        image = image.unsqueeze(1)
                    if image.ndim != 5:
                        raise ValueError(
                            f"DP3 expects '{key}' history in (batch, obs_steps, channels, height, width) "
                            f"layout. Got {tuple(image.shape)}."
                        )
                    image_histories.append(image)
                wrist_images = torch.stack(image_histories, dim=2)

            if wrist_images.ndim != 6:
                raise ValueError(
                    "DP3 expects wrist RGB history in "
                    "(batch, obs_steps, cameras, channels, height, width) layout. "
                    f"Got {tuple(wrist_images.shape)}."
                )
            if wrist_images.shape[:2] != state.shape[:2]:
                raise ValueError("DP3 state and wrist RGB history dimensions must match.")
            if wrist_images.shape[2] != len(self.config.wrist_image_keys):
                raise ValueError(
                    f"DP3 expects {len(self.config.wrist_image_keys)} wrist RGB views, "
                    f"got {wrist_images.shape[2]}."
                )

            encoded = self.observation_encoder(
                state.reshape(batch_size * n_obs_steps, state.shape[-1]),
                point_cloud.reshape(batch_size * n_obs_steps, *point_cloud.shape[-2:]),
                wrist_images.reshape(batch_size * n_obs_steps, *wrist_images.shape[-4:]),
            )
        else:
            encoded = self.observation_encoder(
                state.reshape(batch_size * n_obs_steps, state.shape[-1]),
                point_cloud.reshape(batch_size * n_obs_steps, *point_cloud.shape[-2:]),
            )
        return encoded.reshape(batch_size, n_obs_steps, -1).flatten(start_dim=1)


class DP3Policy(DiffusionPolicy):
    """DP3-style point-cloud policy using LeRobot's Diffusion action denoiser."""

    config_class = DP3Config
    name = "dp3"

    def __init__(self, config: DP3Config, **kwargs):
        require_package("diffusers", extra="dp3")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = DP3DiffusionModel(config)
        self.reset()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            self.config.point_cloud_key: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._queues.update(
            {key: deque(maxlen=self.config.n_obs_steps) for key in self.config.wrist_image_keys}
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select one action from point cloud, optional dual-wrist RGB, and robot state."""

        required = (OBS_STATE, self.config.point_cloud_key, *self.config.wrist_image_keys)
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"DP3 inference batch is missing required feature(s): {missing}.")
        conditioning_batch = {key: batch[key] for key in required}
        self._queues = populate_queues(self._queues, conditioning_batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(conditioning_batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Compute Diffusion loss from the complete multimodal observation."""

        return self.diffusion.compute_loss(batch), None
