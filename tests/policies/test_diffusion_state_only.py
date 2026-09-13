#!/usr/bin/env python

"""Regression tests for state-only Diffusion Policy conditioning."""

import pytest
import torch

pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def test_diffusion_policy_trains_with_state_only():
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        n_groups=4,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )

    policy = DiffusionPolicy(config)
    batch = {
        "observation.state": torch.randn(2, 2, 3),
        "action": torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }

    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("resize_shape", [(32, 32), None])
def test_diffusion_mixed_camera_resolutions(resize_shape):
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        down_dims=(16,),
        diffusion_step_embed_dim=8,
        n_groups=4,
        num_train_timesteps=4,
        num_inference_steps=2,
        use_group_norm=True,
        resize_shape=resize_shape,
        crop_shape=None,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 32)),
            "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 48)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    if resize_shape is None:
        with pytest.raises(ValueError, match="expect all image shapes to match"):
            config.validate_features()
        return
    policy = DiffusionPolicy(config)
    batch = {
        "observation.state": torch.randn(2, 2, 3),
        "observation.images.left": torch.rand(2, 2, 3, 40, 32),
        "observation.images.right": torch.rand(2, 2, 3, 32, 48),
        "action": torch.randn(2, 4, 2),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    policy.eval()
    offline_chunk = policy.predict_action_chunk(batch)
    assert offline_chunk.shape == (2, 2, 2)
    online_batch = {key: value[:, -1] for key, value in batch.items() if key.startswith("observation.")}
    action = policy.select_action(online_batch)
    assert action.shape == (2, 2)
    assert batch["observation.images.left"].shape[-2:] == (40, 32)


def test_diffusion_requires_robot_state():
    config = DiffusionConfig(input_features={}, output_features={})
    with pytest.raises(ValueError, match="observation.state"):
        config.validate_features()
