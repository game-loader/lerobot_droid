#!/usr/bin/env python

"""Tests for the SmolVLM visual-language encoder path in IMF-AttnRes."""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.imf_attnres.modeling_imf_attnres import IMFAttnResModel, IMFAttnResPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

POLICY_NAME = "imf-attnres"
STATE_DIM = 8
ACTION_DIM = 8
IMAGE_SIZE = 16
IMAGE_KEYS = (
    f"{OBS_IMAGES}.agentview",
    f"{OBS_IMAGES}.eye_in_hand",
)


def make_tiny_config():
    config = make_policy_config(
        POLICY_NAME,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        n_layer=1,
        n_emb=32,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    config.device = "cpu"
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
            for key in IMAGE_KEYS
        },
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    return config


def enable_fake_smolvlm(config):
    config.use_smolvlm_vl_encoder = True
    config.load_vlm_weights = True
    config.freeze_vlm_encoder = True
    config.vlm_hidden_size = 12
    config.vlm_tokens_per_step = 5
    config.vlm_tokenizer_max_length = 7
    return config


def make_stacked_batch(*, batch_size: int = 2, token_shape: str = "batch"):
    n_obs_steps = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, n_obs_steps, STATE_DIM),
        OBS_IMAGES: torch.rand(batch_size, n_obs_steps, len(IMAGE_KEYS), 3, IMAGE_SIZE, IMAGE_SIZE),
        ACTION: torch.randn(batch_size, 4, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }
    if token_shape == "batch":
        batch[OBS_LANGUAGE_TOKENS] = torch.arange(batch_size * 7, dtype=torch.long).view(batch_size, 7)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(batch_size, 7, dtype=torch.bool)
    elif token_shape == "history":
        latest = torch.full((batch_size, 7), 9, dtype=torch.long)
        batch[OBS_LANGUAGE_TOKENS] = torch.stack([torch.ones_like(latest), latest], dim=1)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(batch_size, n_obs_steps, 7, dtype=torch.bool)
    else:
        raise ValueError(token_shape)
    return batch


def make_policy_batch(*, batch_size: int = 2):
    batch = make_stacked_batch(batch_size=batch_size)
    stacked_images = batch.pop(OBS_IMAGES)
    for camera_index, key in enumerate(IMAGE_KEYS):
        batch[key] = stacked_images[:, :, camera_index]
    return batch


class FakeSmolVLMVLEncoder(nn.Module):
    last_instance = None

    def __init__(self, config):
        super().__init__()
        FakeSmolVLMVLEncoder.last_instance = self
        self.model_name = config.vlm_model_name
        self.freeze_vlm_encoder = config.freeze_vlm_encoder
        self.load_vlm_weights = config.load_vlm_weights
        self.feature_dim = config.vlm_hidden_size
        self.tokens_per_step = config.vlm_tokens_per_step
        self.dummy = nn.Parameter(torch.zeros(()))
        self.last_images = None
        self.last_state = None
        self.last_input_ids = None
        self.last_attention_mask = None

    def forward(self, images, state, lang_tokens, lang_masks, env_state=None):
        self.last_images = images.detach().clone()
        self.last_state = state.detach().clone()
        self.last_input_ids = lang_tokens.detach().clone()
        self.last_attention_mask = lang_masks.detach().clone()
        flat_batch = images.shape[0]
        values = torch.arange(
            flat_batch * self.tokens_per_step * self.feature_dim,
            dtype=torch.float32,
            device=images.device,
        )
        return values.view(flat_batch, self.tokens_per_step, self.feature_dim) + self.dummy


def patch_fake_encoder(monkeypatch):
    import lerobot.policies.imf_attnres.modeling_imf_attnres as modeling_imf_attnres

    monkeypatch.setattr(modeling_imf_attnres, "IMFAttnResSmolVLMVLEncoder", FakeSmolVLMVLEncoder)


def test_default_imf_attnres_keeps_resnet_path():
    config = make_tiny_config()

    assert config.use_smolvlm_vl_encoder is False

    model = IMFAttnResModel(config)

    assert model.rgb_encoder is not None
    assert model.vl_encoder is None


def test_smolvlm_vl_conditioning_replaces_resnet_and_returns_condition_tokens(monkeypatch):
    patch_fake_encoder(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())

    model = IMFAttnResModel(config)
    cond = model._prepare_conditioning(make_stacked_batch())

    assert model.rgb_encoder is None
    assert isinstance(model.vl_encoder, FakeSmolVLMVLEncoder)
    assert model.vl_encoder.model_name == config.vlm_model_name
    assert model.vl_encoder.freeze_vlm_encoder is True
    assert model.vl_encoder.load_vlm_weights is True
    assert model.cond_dim == config.vlm_hidden_size
    assert model.condition_tokens_per_step == config.vlm_tokens_per_step
    assert model.head.T_cond == 2 + config.n_obs_steps * config.vlm_tokens_per_step
    assert cond.shape == (2, config.n_obs_steps * config.vlm_tokens_per_step, config.vlm_hidden_size)


def test_smolvlm_vl_conditioning_flattens_language_history(monkeypatch):
    patch_fake_encoder(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    model = IMFAttnResModel(config)
    batch = make_stacked_batch(token_shape="history")

    model._prepare_conditioning(batch)

    expected_tokens = batch[OBS_LANGUAGE_TOKENS].reshape(-1, batch[OBS_LANGUAGE_TOKENS].shape[-1])
    assert torch.equal(model.vl_encoder.last_input_ids, expected_tokens)


def test_smolvlm_vl_conditioning_repeats_current_language_for_history(monkeypatch):
    patch_fake_encoder(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    model = IMFAttnResModel(config)
    batch = make_stacked_batch(token_shape="batch")

    model._prepare_conditioning(batch)

    expected_tokens = batch[OBS_LANGUAGE_TOKENS].repeat_interleave(config.n_obs_steps, dim=0)
    assert torch.equal(model.vl_encoder.last_input_ids, expected_tokens)


def test_smolvlm_vl_conditioning_requires_tokenized_language(monkeypatch):
    patch_fake_encoder(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    model = IMFAttnResModel(config)
    batch = make_stacked_batch()
    batch.pop(OBS_LANGUAGE_TOKENS)

    with pytest.raises(ValueError, match="language tokens"):
        model._prepare_conditioning(batch)


def test_smolvlm_vl_policy_forward_returns_finite_scalar_loss(monkeypatch):
    patch_fake_encoder(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    policy = IMFAttnResPolicy(config)

    loss, diagnostics = policy.forward(make_policy_batch())

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert diagnostics is None
