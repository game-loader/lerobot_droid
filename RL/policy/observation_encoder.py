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

"""Observation feature interfaces for state critics and future image critics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from RL.types import ObservationBatch


class ImageEncoderRequiredError(RuntimeError):
    """Raised when image observations reach a state-only feature encoder."""


class ObservationFeatureEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observation: ObservationBatch) -> Tensor:
        raise NotImplementedError


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def is_image_feature(key: str, value: Tensor) -> bool:
    """Recognize canonical and common Gym image keys without matching metadata."""

    lowered = key.lower()
    if "metadata" in lowered or value.ndim < 4:
        return False
    return (
        key == "observation.image"
        or key.startswith("observation.image.")
        or key.startswith("observation.images.")
        or any(token in lowered for token in ("pixel", "rgb", "camera", "front", "rear"))
    )


def _image_keys(observation: ObservationBatch, *, state_key: str) -> list[str]:
    keys: list[str] = []
    for key in observation.features:
        if key == state_key:
            continue
        if is_image_feature(key, observation.features[key]):
            keys.append(key)
    return sorted(keys)


class StateFeatureEncoder(ObservationFeatureEncoder):
    """MLP over flattened state history with an explicit image extension point."""

    def __init__(
        self,
        *,
        state_dim: int,
        n_obs_steps: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        state_key: str = "observation.state",
        image_encoder: ObservationFeatureEncoder | None = None,
    ) -> None:
        super().__init__()
        _positive_int("state_dim", state_dim)
        _positive_int("n_obs_steps", n_obs_steps)
        _positive_int("output_dim", output_dim)
        if not isinstance(state_key, str) or not state_key:
            raise ValueError(f"state_key must be a nonempty string, got {state_key!r}")
        hidden_dims = tuple(hidden_dims)
        for index, hidden_dim in enumerate(hidden_dims):
            _positive_int(f"hidden_dims[{index}]", hidden_dim)
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.state_key = state_key
        self.image_encoder = image_encoder
        self._output_dim = output_dim

        layers: list[nn.Module] = []
        input_dim = state_dim * n_obs_steps
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.state_network = nn.Sequential(*layers)
        self.fusion = (
            nn.Linear(output_dim + image_encoder.output_dim, output_dim)
            if image_encoder is not None
            else None
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, observation: ObservationBatch) -> Tensor:
        if not isinstance(observation, ObservationBatch):
            raise ValueError(
                f"observation must be an ObservationBatch, got {type(observation).__name__}"
            )
        if self.state_key not in observation.features:
            raise ValueError(
                f"observation is missing state_key={self.state_key!r}: "
                f"actual={sorted(observation.features)}"
            )
        state = observation.features[self.state_key]
        expected_shape = (observation.batch_size(), self.n_obs_steps, self.state_dim)
        if state.shape != expected_shape:
            raise ValueError(
                f"{self.state_key} must have shape {expected_shape}, got {tuple(state.shape)}"
            )
        if not state.is_floating_point():
            raise ValueError(f"{self.state_key} must have a floating-point dtype, got {state.dtype}")
        state_features = self.state_network(state.flatten(start_dim=1))

        image_keys = _image_keys(observation, state_key=self.state_key)
        if not image_keys:
            if self.image_encoder is not None:
                raise ImageEncoderRequiredError(
                    "image_encoder is configured but the observation has no image features"
                )
            return state_features
        if self.image_encoder is None or self.fusion is None:
            raise ImageEncoderRequiredError(
                f"image feature encoder is required for keys {image_keys}"
            )
        image_observation = ObservationBatch(
            {key: observation.features[key] for key in image_keys}
        )
        image_features = self.image_encoder(image_observation)
        expected_image_shape = (observation.batch_size(), self.image_encoder.output_dim)
        if image_features.shape != expected_image_shape:
            raise ValueError(
                "image encoder output must have shape "
                f"{expected_image_shape}, got {tuple(image_features.shape)}"
            )
        if image_features.device != state_features.device:
            raise ValueError(
                "image and state features must be on the same device, "
                f"got {image_features.device} and {state_features.device}"
            )
        if image_features.dtype != state_features.dtype:
            image_features = image_features.to(dtype=state_features.dtype)
        return self.fusion(torch.cat((state_features, image_features), dim=-1))
