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

import pytest
import torch

from RL.policy.observation_encoder import (
    ImageEncoderRequiredError,
    ObservationFeatureEncoder,
    StateFeatureEncoder,
)
from RL.types import ObservationBatch


def test_state_encoder_outputs_fixed_feature_size() -> None:
    encoder = StateFeatureEncoder(
        state_dim=39,
        n_obs_steps=2,
        hidden_dims=(128, 128),
        output_dim=128,
    )

    output = encoder(ObservationBatch({"observation.state": torch.zeros(4, 2, 39)}))

    assert output.shape == (4, 128)
    assert torch.isfinite(output).all()


def test_state_encoder_supports_custom_state_key() -> None:
    encoder = StateFeatureEncoder(
        state_dim=3,
        n_obs_steps=2,
        hidden_dims=(8,),
        output_dim=4,
        state_key="robot.state",
    )

    output = encoder(ObservationBatch({"robot.state": torch.zeros(2, 2, 3)}))

    assert output.shape == (2, 4)


def test_images_require_registered_feature_encoder() -> None:
    encoder = StateFeatureEncoder(state_dim=39, n_obs_steps=2, hidden_dims=(64,), output_dim=32)
    observation = ObservationBatch(
        {
            "observation.state": torch.zeros(1, 2, 39),
            "observation.images.front": torch.zeros(1, 2, 3, 16, 16),
        }
    )

    with pytest.raises(ImageEncoderRequiredError, match="observation.images.front"):
        encoder(observation)


class _MeanImageEncoder(ObservationFeatureEncoder):
    @property
    def output_dim(self) -> int:
        return 3

    def forward(self, observation: ObservationBatch) -> torch.Tensor:
        assert set(observation.features) == {"observation.images.front"}
        image = observation.features["observation.images.front"]
        pooled = image.float().mean(dim=(1, 2, 3, 4))
        return pooled.unsqueeze(-1).expand(-1, self.output_dim)


def test_registered_image_encoder_is_composed_with_state_features() -> None:
    encoder = StateFeatureEncoder(
        state_dim=3,
        n_obs_steps=2,
        hidden_dims=(8,),
        output_dim=5,
        image_encoder=_MeanImageEncoder(),
    )
    observation = ObservationBatch(
        {
            "observation.state": torch.zeros(2, 2, 3),
            "observation.images.front": torch.ones(2, 2, 3, 4, 4),
        }
    )

    output = encoder(observation)

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()


def test_state_encoder_rejects_wrong_history_shape() -> None:
    encoder = StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4)

    with pytest.raises(ValueError, match="observation.state"):
        encoder(ObservationBatch({"observation.state": torch.zeros(2, 1, 3)}))
