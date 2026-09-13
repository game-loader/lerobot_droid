#!/usr/bin/env python

"""Tests for the SmolVLM visual-language encoder path in IMF-AttnRes."""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.imf_attnres.imf_transformer1d import IMFTransformer1D
from lerobot.policies.imf_attnres.modeling_imf_attnres import (
    IMFAttnResModel,
    IMFAttnResPolicy,
    IMFAttnResSmolVLMVLEncoder,
)
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
    assert config.vlm_text_encoder_mode == "embedding"
    assert config.vlm_text_num_layers == 16

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


class _FakeTextConfig:
    hidden_size = 6
    pad_token_id = 0

    def __init__(self, num_hidden_layers: int = 4):
        self.num_hidden_layers = num_hidden_layers


class _FakeVisionConfig:
    patch_size = 4
    image_size = 8


class _FakeSmolVLMConfig:
    pad_token_id = 0
    scale_factor = 1
    vision_config = _FakeVisionConfig()

    def __init__(self, num_hidden_layers: int = 4):
        self.text_config = _FakeTextConfig(num_hidden_layers=num_hidden_layers)


class _FakeTokenEmbedding(nn.Module):
    embedding_dim = 6

    def forward(self, tokens):
        return tokens.to(dtype=torch.float32).unsqueeze(-1).expand(*tokens.shape, self.embedding_dim)


class _FakeTextLayer(nn.Module):
    def __init__(self, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states + float(self.layer_index + 1)


class _FakeTextModel(nn.Module):
    def __init__(self, num_hidden_layers: int):
        super().__init__()
        self.config = _FakeTextConfig(num_hidden_layers=num_hidden_layers)
        self.embedding = _FakeTokenEmbedding()
        self.layers = nn.ModuleList([_FakeTextLayer(index) for index in range(num_hidden_layers)])
        self.forward_calls = 0
        self.last_input_shape = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self, *, inputs_embeds, attention_mask=None, use_cache=False, output_hidden_states=False, **kwargs
    ):
        self.forward_calls += 1
        self.last_input_shape = tuple(inputs_embeds.shape)
        hidden_states = inputs_embeds
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            all_hidden_states.append(hidden_states)
        return type(
            "FakeTextOutput",
            (),
            {
                "last_hidden_state": hidden_states,
                "hidden_states": tuple(all_hidden_states) if output_hidden_states else None,
            },
        )()


class _FakeVisionModel(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, *, pixel_values, patch_attention_mask=None):
        batch_size = pixel_values.shape[0]
        self.batch_sizes.append(batch_size)
        image_values = pixel_values.mean(dim=(1, 2, 3), keepdim=False).view(batch_size, 1, 1)
        hidden_states = image_values.expand(batch_size, 4, 6).to(dtype=pixel_values.dtype)
        return type("FakeVisionOutput", (), {"last_hidden_state": hidden_states})()


class _FakeConnector(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _FakeSmolVLMCore(nn.Module):
    def __init__(self, num_hidden_layers: int):
        super().__init__()
        self.vision_model = _FakeVisionModel()
        self.connector = _FakeConnector()
        self.text_model = _FakeTextModel(num_hidden_layers)


class _FakeSmolVLMForConditionalGeneration(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or _FakeSmolVLMConfig()
        self.model = _FakeSmolVLMCore(self.config.text_config.num_hidden_layers)
        self.lm_head = nn.Linear(self.config.text_config.hidden_size, 10, bias=False)


def _install_fake_transformers(monkeypatch):
    import transformers

    created_models = []

    def fake_config_from_pretrained(model_name):
        return _FakeSmolVLMConfig(num_hidden_layers=4)

    def fake_from_pretrained(model_name, **kwargs):
        model = _FakeSmolVLMForConditionalGeneration(config=kwargs.get("config"))
        created_models.append(model)
        return model

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        transformers,
        "SmolVLMForConditionalGeneration",
        _FakeSmolVLMForConditionalGeneration,
    )
    return created_models


def _make_direct_vl_encoder_inputs(config, *, batch_size: int = 1):
    return {
        "images": torch.rand(batch_size, len(IMAGE_KEYS), 3, IMAGE_SIZE, IMAGE_SIZE),
        "state": torch.randn(batch_size, STATE_DIM),
        "lang_tokens": torch.tensor([[2, 3, 0]], dtype=torch.long).expand(batch_size, -1).clone(),
        "lang_masks": torch.tensor([[True, True, False]], dtype=torch.bool).expand(batch_size, -1).clone(),
    }


def test_smolvlm_embedding_text_mode_prunes_text_layers_and_uses_only_token_embeddings(monkeypatch):
    created_models = _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "embedding"

    encoder = IMFAttnResSmolVLMVLEncoder(config)
    prefix_tokens = encoder(**_make_direct_vl_encoder_inputs(config))

    fake_vlm = created_models[-1]
    assert len(encoder.vlm_model.text_model.layers) == 0
    assert encoder.vlm_model.text_model.forward_calls == 0
    assert isinstance(fake_vlm.lm_head, nn.Identity)
    assert prefix_tokens.shape == (1, 12, 6)

    language_tokens = prefix_tokens[:, 8:11]
    torch.testing.assert_close(language_tokens[0, 0], torch.full((6,), 2.0))
    torch.testing.assert_close(language_tokens[0, 1], torch.full((6,), 3.0))
    torch.testing.assert_close(language_tokens[0, 2], torch.zeros(6))


def test_smolvlm_transformer_text_mode_processes_image_language_and_state_like_smolvla(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "transformer"
    config.vlm_text_num_layers = 2

    encoder = IMFAttnResSmolVLMVLEncoder(config)
    inputs = _make_direct_vl_encoder_inputs(config)
    with torch.no_grad():
        encoder.state_projection.weight.zero_()
        encoder.state_projection.bias.fill_(1.0)
    prefix_tokens = encoder(**inputs)

    text_model = encoder.vlm_model.text_model
    assert len(text_model.layers) == 2
    assert text_model.config.num_hidden_layers == 2
    assert text_model.forward_calls == 1
    assert text_model.last_input_shape == (1, 12, 6)
    assert [layer.calls for layer in text_model.layers] == [1, 1]
    assert prefix_tokens.shape == (1, 12, 6)

    image_tokens = prefix_tokens[:, :8]
    language_tokens = prefix_tokens[:, 8:11]
    state_token = prefix_tokens[:, 11:]
    embedding_scale = torch.tensor(6.0).sqrt()
    expected_image_values = encoder._preprocess_images(inputs["images"]).mean(dim=(1, 2, 3))
    expected_image_tokens = expected_image_values.view(1, len(IMAGE_KEYS), 1, 1).expand(
        1, len(IMAGE_KEYS), 4, 6
    )
    expected_image_tokens = expected_image_tokens.reshape(1, len(IMAGE_KEYS) * 4, 6) * embedding_scale + 3.0
    torch.testing.assert_close(image_tokens, expected_image_tokens)
    torch.testing.assert_close(language_tokens[0, 0], torch.full((6,), 2.0 * embedding_scale + 3.0))
    torch.testing.assert_close(language_tokens[0, 1], torch.full((6,), 3.0 * embedding_scale + 3.0))
    torch.testing.assert_close(language_tokens[0, 2], torch.zeros(6))
    torch.testing.assert_close(state_token, torch.full((1, 1, 6), 4.0))


def test_smolvlm_transformer_text_mode_can_keep_state_outside_text_layers(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "transformer"
    config.vlm_text_num_layers = 2
    config.vlm_state_in_text_layers = False

    encoder = IMFAttnResSmolVLMVLEncoder(config)
    inputs = _make_direct_vl_encoder_inputs(config)
    with torch.no_grad():
        encoder.state_projection.weight.zero_()
        encoder.state_projection.bias.fill_(1.0)
    prefix_tokens = encoder(**inputs)

    text_model = encoder.vlm_model.text_model
    assert text_model.forward_calls == 1
    assert text_model.last_input_shape == (1, 11, 6)
    assert prefix_tokens.shape == (1, 12, 6)

    image_tokens = prefix_tokens[:, :8]
    language_tokens = prefix_tokens[:, 8:11]
    state_token = prefix_tokens[:, 11:]
    embedding_scale = torch.tensor(6.0).sqrt()
    expected_image_values = encoder._preprocess_images(inputs["images"]).mean(dim=(1, 2, 3))
    expected_image_tokens = expected_image_values.view(1, len(IMAGE_KEYS), 1, 1).expand(
        1, len(IMAGE_KEYS), 4, 6
    )
    expected_image_tokens = expected_image_tokens.reshape(1, len(IMAGE_KEYS) * 4, 6) * embedding_scale + 3.0
    torch.testing.assert_close(image_tokens, expected_image_tokens)
    torch.testing.assert_close(language_tokens[0, 0], torch.full((6,), 2.0 * embedding_scale + 3.0))
    torch.testing.assert_close(language_tokens[0, 1], torch.full((6,), 3.0 * embedding_scale + 3.0))
    torch.testing.assert_close(language_tokens[0, 2], torch.zeros(6))
    torch.testing.assert_close(state_token, torch.ones(1, 1, 6))


def test_smolvlm_layerwise_mode_returns_per_text_layer_prefix_states(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "transformer"
    config.vlm_text_num_layers = 3
    config.vlm_conditioning_mode = "layerwise"

    encoder = IMFAttnResSmolVLMVLEncoder(config)
    inputs = _make_direct_vl_encoder_inputs(config)
    with torch.no_grad():
        encoder.state_projection.weight.zero_()
        encoder.state_projection.bias.fill_(1.0)

    prefix_layers, prefix_masks = encoder.forward_layerwise_prefix(
        **inputs,
        num_layers=2,
    )

    text_model = encoder.vlm_model.text_model
    assert text_model.forward_calls == 1
    assert text_model.last_input_shape == (1, 12, 6)
    assert [layer.calls for layer in text_model.layers] == [1, 1, 1]
    assert prefix_layers.shape == (1, 2, 12, 6)
    assert prefix_masks.shape == (1, 12)
    assert prefix_masks[:, :10].all()
    assert not prefix_masks[:, 10].any()
    assert prefix_masks[:, 11:].all()

    embedding_scale = torch.tensor(6.0).sqrt()
    expected_image_values = encoder._preprocess_images(inputs["images"]).mean(dim=(1, 2, 3))
    expected_image_tokens = expected_image_values.view(1, len(IMAGE_KEYS), 1, 1).expand(
        1, len(IMAGE_KEYS), 4, 6
    )
    expected_image_tokens = expected_image_tokens.reshape(1, len(IMAGE_KEYS) * 4, 6) * embedding_scale
    torch.testing.assert_close(prefix_layers[:, 0, :8], expected_image_tokens + 1.0)
    torch.testing.assert_close(prefix_layers[:, 1, :8], expected_image_tokens + 3.0)
    torch.testing.assert_close(prefix_layers[:, 0, 11:], torch.full((1, 1, 6), 2.0))
    torch.testing.assert_close(prefix_layers[:, 1, 11:], torch.full((1, 1, 6), 4.0))


def test_smolvlm_vl_encoder_chunks_vision_forward_without_changing_token_order(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_image_forward_batch_size = 3

    encoder = IMFAttnResSmolVLMVLEncoder(config)
    inputs = _make_direct_vl_encoder_inputs(config, batch_size=3)
    inputs["images"] = torch.arange(
        3 * len(IMAGE_KEYS) * 3 * IMAGE_SIZE * IMAGE_SIZE,
        dtype=torch.float32,
    ).view(3, len(IMAGE_KEYS), 3, IMAGE_SIZE, IMAGE_SIZE)

    image_tokens = encoder._embed_images(inputs["images"])

    assert encoder.vision_model.batch_sizes == [3, 3]
    assert image_tokens.shape == (3, 8, 6)
    flat_images = encoder._preprocess_images(inputs["images"])
    expected_values = flat_images.mean(dim=(1, 2, 3))
    expected_tokens = expected_values.view(3, len(IMAGE_KEYS), 1, 1).expand(3, len(IMAGE_KEYS), 4, 6)
    expected_tokens = expected_tokens.reshape(3, len(IMAGE_KEYS) * 4, 6)
    torch.testing.assert_close(image_tokens, expected_tokens)


def test_smolvlm_text_encoder_config_rejects_invalid_mode_and_layer_count():
    with pytest.raises(ValueError, match="vlm_text_encoder_mode"):
        make_policy_config(POLICY_NAME, vlm_text_encoder_mode="full", push_to_hub=False)

    with pytest.raises(ValueError, match="vlm_text_num_layers"):
        make_policy_config(
            POLICY_NAME,
            vlm_text_encoder_mode="transformer",
            vlm_text_num_layers=0,
            push_to_hub=False,
        )

    with pytest.raises(ValueError, match="vlm_image_forward_batch_size"):
        make_policy_config(POLICY_NAME, vlm_image_forward_batch_size=-1, push_to_hub=False)

    with pytest.raises(ValueError, match="vlm_conditioning_mode"):
        make_policy_config(POLICY_NAME, vlm_conditioning_mode="late", push_to_hub=False)

    with pytest.raises(ValueError, match="vlm_text_encoder_mode='transformer'"):
        make_policy_config(
            POLICY_NAME,
            use_smolvlm_vl_encoder=True,
            vlm_conditioning_mode="layerwise",
            vlm_text_encoder_mode="embedding",
            push_to_hub=False,
        )


def test_smolvlm_layerwise_prepare_conditioning_returns_prefix_stack(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.n_layer = 2
    config.n_emb = 32
    config.n_head = 8
    config.n_kv_head = 8
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "transformer"
    config.vlm_text_num_layers = 2
    config.vlm_conditioning_mode = "layerwise"

    model = IMFAttnResModel(config)
    cond = model._prepare_conditioning(make_stacked_batch())

    assert cond.prefix_layers.shape == (
        2,
        config.n_layer,
        config.n_obs_steps * model.vl_encoder.tokens_per_step,
        model.vl_encoder.feature_dim,
    )
    assert cond.prefix_mask.shape == (
        2,
        config.n_obs_steps * model.vl_encoder.tokens_per_step,
    )
    assert cond.attention_mode == config.vlm_layerwise_attention_mode
    assert cond.self_attn_every_n_layers == config.vlm_layerwise_self_attn_every_n_layers


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_smolvlm_layerwise_policy_forward_returns_finite_scalar_loss(monkeypatch, dropout):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.n_layer = 2
    config.n_emb = 32
    config.n_head = 8
    config.n_kv_head = 8
    config.vlm_resize_shape = (8, 8)
    config.vlm_tokenizer_max_length = 3
    config.vlm_text_encoder_mode = "transformer"
    config.vlm_text_num_layers = 2
    config.vlm_conditioning_mode = "layerwise"
    config.p_drop_attn = dropout
    config.p_drop_emb = dropout

    policy = IMFAttnResPolicy(config)

    loss, diagnostics = policy.forward(make_policy_batch())

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert diagnostics is None
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.model.head.parameters())


def test_smolvlm_rectangular_resize_preserves_legacy_width_height_order(monkeypatch):
    _install_fake_transformers(monkeypatch)
    config = enable_fake_smolvlm(make_tiny_config())
    config.vlm_resize_shape = (16, 8)
    encoder = IMFAttnResSmolVLMVLEncoder(config)
    images = torch.ones(1, len(IMAGE_KEYS), 3, IMAGE_SIZE, IMAGE_SIZE)
    pixels = encoder._preprocess_images(images)
    assert pixels.shape == (len(IMAGE_KEYS), 3, 8, 16)


@pytest.mark.parametrize("backbone_type", ["attnres_full", "attnres_diff", "diff_transformer", "vanilla"])
def test_imf_transformer_layerwise_conditioning_supports_existing_backbones(backbone_type):
    torch.manual_seed(0)
    head = IMFTransformer1D(
        input_dim=ACTION_DIM,
        output_dim=ACTION_DIM,
        horizon=4,
        n_obs_steps=5,
        cond_dim=6,
        n_layer=2,
        n_head=8,
        n_emb=32,
        n_kv_head=8,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        backbone_type=backbone_type,
        time_as_cond=True,
        obs_as_cond=True,
    )
    sample = torch.randn(2, 4, ACTION_DIM)
    cond = {
        "prefix_layers": torch.randn(2, 2, 5, 6),
        "prefix_mask": torch.ones(2, 5, dtype=torch.bool),
        "attention_mode": "cross_attn",
        "self_attn_every_n_layers": 2,
    }

    output = head(sample, torch.zeros(2), torch.ones(2), cond=cond)

    assert output.shape == sample.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("backbone_type", ["attnres_full", "attnres_diff"])
def test_imf_transformer_layerwise_attnres_keeps_residual_sources_action_only(backbone_type):
    torch.manual_seed(0)
    horizon = 4
    prefix_tokens = 5
    head = IMFTransformer1D(
        input_dim=ACTION_DIM,
        output_dim=ACTION_DIM,
        horizon=horizon,
        n_obs_steps=prefix_tokens,
        cond_dim=6,
        n_layer=2,
        n_head=8,
        n_emb=32,
        n_kv_head=8,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        backbone_type=backbone_type,
        time_as_cond=True,
        obs_as_cond=True,
    )
    seen_source_lengths = []
    seen_source_depths = []

    def record_sources(_module, inputs, _output):
        (sources, *_rest) = inputs
        seen_source_depths.append(sources.shape[0])
        seen_source_lengths.append(sources.shape[2])

    handles = [layer.attn_res.register_forward_hook(record_sources) for layer in head.attnres_backbone.layers]
    try:
        sample = torch.randn(2, horizon, ACTION_DIM)
        cond = {
            "prefix_layers": torch.randn(2, 2, prefix_tokens, 6),
            "prefix_mask": torch.ones(2, prefix_tokens, dtype=torch.bool),
            "attention_mode": "cross_attn",
            "self_attn_every_n_layers": 2,
        }

        output = head(sample, torch.zeros(2), torch.ones(2), cond=cond)
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == sample.shape
    assert seen_source_lengths
    assert set(seen_source_lengths) == {horizon + 2}
    assert prefix_tokens + horizon + 2 not in seen_source_lengths
    assert seen_source_depths == list(range(1, 2 * head.n_layer + 1))
