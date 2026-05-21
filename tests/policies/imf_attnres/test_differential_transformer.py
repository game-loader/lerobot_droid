#!/usr/bin/env python

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.imf_attnres.attnres_transformer_components import (
    AttnResOperator,
    AttnResTransformerBackbone,
    DifferentialTransformerBackbone,
    MultiheadDifferentialSelfAttention,
)
from lerobot.policies.imf_attnres.modeling_imf_attnres import IMFAttnResModel
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

POLICY_NAME = "imf-attnres"
STATE_DIM = 8
ACTION_DIM = 7
IMAGE_SIZE = 16
IMAGE_KEYS = (
    f"{OBS_IMAGES}.image",
    f"{OBS_IMAGES}.image2",
)


def make_tiny_diff_transformer_config():
    config = make_policy_config(
        POLICY_NAME,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_emb=32,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        backbone_type="diff_transformer",
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


def make_tiny_attnres_diff_config():
    config = make_tiny_diff_transformer_config()
    config.backbone_type = "attnres_diff"
    return config


def make_batch(batch_size: int = 2):
    return {
        OBS_STATE: torch.randn(batch_size, 2, STATE_DIM),
        ACTION: torch.randn(batch_size, 4, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        **{
            key: torch.rand(batch_size, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
            for key in IMAGE_KEYS
        },
    }


def test_imf_attnres_default_backbone_remains_attnres_full():
    config = make_policy_config(POLICY_NAME, push_to_hub=False)

    assert config.backbone_type == "attnres_full"


def test_imf_attnres_diff_transformer_backbone_can_replace_default_transformer():
    config = make_tiny_diff_transformer_config()
    model = IMFAttnResModel(config)

    assert model.head.backbone_type == "diff_transformer"
    assert isinstance(model.head.diff_transformer_backbone, DifferentialTransformerBackbone)
    assert model.head.attnres_backbone is None


def test_imf_attnres_attnres_diff_combines_attnres_operator_with_differential_attention():
    config = make_tiny_attnres_diff_config()
    model = IMFAttnResModel(config)

    assert model.head.backbone_type == "attnres_diff"
    assert isinstance(model.head.attnres_backbone, AttnResTransformerBackbone)
    assert model.head.diff_transformer_backbone is None
    attention_layers = [layer for layer in model.head.attnres_backbone.layers if layer.is_attention]
    assert attention_layers
    assert all(isinstance(layer.attn_res, AttnResOperator) for layer in attention_layers)
    assert all(isinstance(layer.fn, MultiheadDifferentialSelfAttention) for layer in attention_layers)


def test_imf_attnres_attnres_diff_policy_forward_returns_scalar_loss():
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_attnres_diff_config()
    policy = policy_cls(config)
    policy.train()

    loss, output_dict = policy.forward(make_batch())

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert output_dict is None


def test_imf_attnres_diff_transformer_policy_forward_returns_scalar_loss():
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_diff_transformer_config()
    policy = policy_cls(config)
    policy.train()

    loss, output_dict = policy.forward(make_batch())

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert output_dict is None


def test_imf_attnres_diff_transformer_optimizer_groups_cover_all_parameters():
    config = make_tiny_diff_transformer_config()
    model = IMFAttnResModel(config)

    optim_groups = model.head.get_optim_groups(weight_decay=1e-3)
    grouped_params = {
        id(parameter)
        for group in optim_groups
        for parameter in group["params"]
    }

    assert grouped_params == {id(parameter) for parameter in model.head.parameters()}


def test_multihead_differential_self_attention_keeps_output_shape_without_flash_attn():
    module = MultiheadDifferentialSelfAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )
    module.eval()
    x = torch.randn(2, 5, 32)

    out = module(x)

    assert out.shape == x.shape


def test_differential_attention_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    causal = MultiheadDifferentialSelfAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        causal_attn=True,
    )
    causal.eval()
    noncausal = MultiheadDifferentialSelfAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        causal_attn=False,
    )
    noncausal.load_state_dict(causal.state_dict())
    noncausal.eval()

    x = torch.randn(1, 5, 32)
    x_changed = x.clone()
    x_changed[:, 3:] = x_changed[:, 3:] + 10.0

    causal_out = causal(x)
    causal_changed_out = causal(x_changed)
    noncausal_out = noncausal(x)
    noncausal_changed_out = noncausal(x_changed)

    torch.testing.assert_close(causal_out[:, 0], causal_changed_out[:, 0], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(noncausal_out[:, 0], noncausal_changed_out[:, 0], rtol=1e-5, atol=1e-5)


def test_imf_attnres_differential_transformer_does_not_import_flash_attn():
    policy_dir = Path("src/lerobot/policies/imf_attnres")
    source = "\n".join(
        [
            (policy_dir / "attnres_transformer_components.py").read_text(),
            (policy_dir / "imf_transformer1d.py").read_text(),
        ]
    )

    assert "flash_attn" not in source
