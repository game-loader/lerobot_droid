# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Smoke tests for the offline diffusion-RL training orchestrator."""

from __future__ import annotations

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import CheckpointAdapter
from RL.algorithms.iql import IQL
from RL.config import TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.trainers.offline import OfflineTrainer
from RL.types import DecisionBatch, ObservationBatch


def _adapter(root: Path) -> DiffusionRLAdapter:
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        pretrained_backbone_weights=None,
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = DiffusionPolicy(config)
    policy.save_pretrained(root)
    stats = {
        "observation.state": {"min": torch.full((3,), -2.0), "max": torch.full((3,), 2.0)},
        "action": {"min": torch.tensor([-2.0, 0.0]), "max": torch.tensor([2.0, 0.0])},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return DiffusionRLAdapter(
        CheckpointAdapter.load(root, device="cpu"),
        TraceConfig(num_inference_steps=2, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )


def _batch() -> DecisionBatch:
    return DecisionBatch(
        observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.5)}),
        next_observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.75)}),
        action=torch.tensor(
            [
                [[0.25, 0.0], [0.5, 0.0]],
                [[-0.25, 0.0], [99.0, 0.0]],
            ]
        ),
        action_valid=torch.tensor([[True, True], [True, False]]),
        reward=torch.tensor([[0.0], [1.0]]),
        done=torch.tensor([[False], [True]]),
        discount=torch.full((2, 1), 0.9),
    )


def test_offline_train_step_updates_current_but_not_old_without_sync(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(
            state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4
        ),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        actor_clip_ratio=0.2,
    )
    current_before = [parameter.detach().clone() for parameter in current.policy.parameters()]
    old_before = [parameter.detach().clone() for parameter in old.policy.parameters()]

    metrics = trainer.train_step(_batch(), generator=current.make_generator(7))

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert trainer.counters.iql_updates == 1
    assert trainer.counters.actor_updates == 1
    assert trainer.counters.old_policy_syncs == 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(current_before, current.policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(old_before, old.policy.parameters(), strict=True)
    )
    normalized = trainer.normalized_batch(_batch())
    torch.testing.assert_close(
        normalized.observation.features["observation.state"],
        current.checkpoint.normalize_observation(_batch().observation).features["observation.state"],
    )
    torch.testing.assert_close(normalized.action, current.checkpoint.normalize_action(_batch().action))
