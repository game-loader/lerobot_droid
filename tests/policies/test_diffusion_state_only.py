#!/usr/bin/env python

"""Regression tests for state-only Diffusion Policy conditioning."""

import torch

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
