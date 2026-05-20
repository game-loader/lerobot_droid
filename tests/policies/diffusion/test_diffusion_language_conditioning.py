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

import pytest
import torch
from torch import nn

pytest.importorskip("diffusers")

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def make_config(*, use_language: bool = True, language_projection_dim: int = 5) -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        down_dims=(16, 32),
        diffusion_step_embed_dim=8,
        n_groups=8,
        num_train_timesteps=4,
        num_inference_steps=2,
        use_smolvlm_language_conditioning=use_language,
        language_projection_dim=language_projection_dim,
        load_language_encoder_weights=False,
    )


class FakeLanguageConditioner(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        *,
        freeze_language_encoder: bool = True,
        load_language_encoder_weights: bool = True,
        torch_dtype: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze_language_encoder = freeze_language_encoder
        self.load_language_encoder_weights = load_language_encoder_weights
        self.torch_dtype = torch_dtype
        self.dummy = nn.Parameter(torch.zeros(()))
        self.last_input_ids = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.last_input_ids = input_ids.detach().clone()
        batch_size = input_ids.shape[0]
        return torch.arange(
            batch_size * self.projection_dim,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(batch_size, self.projection_dim)


def patch_language_conditioner(monkeypatch):
    import lerobot.policies.diffusion.modeling_diffusion as modeling_diffusion

    monkeypatch.setattr(modeling_diffusion, "SmolVLMTextConditioner", FakeLanguageConditioner)


def test_smolvlm_language_condition_is_appended_to_unet_film_condition(monkeypatch):
    patch_language_conditioner(monkeypatch)

    config = make_config(use_language=True, language_projection_dim=5)
    model = DiffusionModel(config)

    assert isinstance(model.noise_scheduler, DDPMScheduler)
    assert model.language_encoder.model_name == config.language_model_name
    assert model.language_encoder.freeze_language_encoder == config.freeze_language_encoder
    assert model.language_encoder.load_language_encoder_weights == config.load_language_encoder_weights
    assert model.language_encoder.torch_dtype == config.language_encoder_torch_dtype

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(batch_size, config.n_obs_steps, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 7, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 7, dtype=torch.bool),
    }

    global_cond = model._prepare_global_conditioning(batch)

    per_step_dim = 3 + 2 + config.language_projection_dim
    assert global_cond.shape == (batch_size, config.n_obs_steps * per_step_dim)

    global_cond_by_step = global_cond.view(batch_size, config.n_obs_steps, per_step_dim)
    language_by_step = global_cond_by_step[:, :, -config.language_projection_dim :]
    expected_language = model.language_encoder(
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    ).unsqueeze(1)
    assert torch.equal(language_by_step, expected_language.expand(-1, config.n_obs_steps, -1))

    # The enlarged global condition is the input to every U-Net FiLM encoder.
    first_film_linear = model.unet.down_modules[0][0].cond_encoder[1]
    assert first_film_linear.in_features == config.diffusion_step_embed_dim + global_cond.shape[-1]


def test_smolvlm_language_condition_uses_latest_queued_tokens(monkeypatch):
    patch_language_conditioner(monkeypatch)

    config = make_config(use_language=True, language_projection_dim=4)
    model = DiffusionModel(config)

    latest_tokens = torch.full((2, 6), 9, dtype=torch.long)
    batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(2, config.n_obs_steps, 2),
        OBS_LANGUAGE_TOKENS: torch.stack([torch.ones_like(latest_tokens), latest_tokens], dim=1),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, config.n_obs_steps, 6, dtype=torch.bool),
    }

    model._prepare_global_conditioning(batch)

    assert torch.equal(model.language_encoder.last_input_ids, latest_tokens)


def test_smolvlm_language_condition_requires_tokenized_task(monkeypatch):
    patch_language_conditioner(monkeypatch)

    config = make_config(use_language=True)
    model = DiffusionModel(config)
    batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(2, config.n_obs_steps, 2),
    }

    with pytest.raises(ValueError, match="tokenized task inputs are missing"):
        model._prepare_global_conditioning(batch)


def test_diffusion_global_condition_is_unchanged_when_language_disabled():
    config = make_config(use_language=False)
    model = DiffusionModel(config)

    batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(2, config.n_obs_steps, 2),
    }

    global_cond = model._prepare_global_conditioning(batch)

    assert model.language_encoder is None
    assert global_cond.shape == (2, config.n_obs_steps * (3 + 2))
