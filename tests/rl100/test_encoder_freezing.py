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

"""Offline RL freezes the observation encoder to keep the representation stable."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers", exc_type=ModuleNotFoundError)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.policies.dp3.modeling_dp3 import DP3Policy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import CheckpointAdapter
from RL.algorithms.iql import IQL
from RL.config import TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.types import DecisionBatch, ObservationBatch


def _active_mask(action_dim: int = 2) -> torch.Tensor:
    return torch.ones(action_dim, dtype=torch.bool)


def _freeze_iql() -> IQL:
    return IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=_active_mask(),
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
        freeze_encoder=True,
    )


def _batch() -> DecisionBatch:
    return DecisionBatch(
        observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.5)}),
        next_observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.75)}),
        action=torch.tensor(
            [
                [[0.25, 0.0], [0.5, 0.0]],
                [[-0.25, 0.0], [0.0, 0.0]],
            ]
        ),
        action_valid=torch.tensor([[True, True], [True, False]]),
        reward=torch.tensor([[0.0], [1.0]]),
        done=torch.tensor([[False], [True]]),
        discount=torch.full((2, 1), 0.9),
    )


def test_freeze_encoder_excludes_observation_encoder_from_q_optimizer() -> None:
    iql = _freeze_iql()
    q_ids = {id(parameter) for group in iql.q_optimizer.param_groups for parameter in group["params"]}
    encoder_ids = {id(parameter) for parameter in iql.feature_encoder.parameters()}

    assert encoder_ids.isdisjoint(q_ids)
    assert all(not parameter.requires_grad for parameter in iql.feature_encoder.parameters())
    assert not iql.feature_encoder.training


def test_freeze_encoder_keeps_encoder_weights_unchanged_across_update() -> None:
    iql = _freeze_iql()
    encoder_before = {
        name: parameter.detach().clone() for name, parameter in iql.feature_encoder.named_parameters()
    }
    q_before = [parameter.detach().clone() for parameter in iql.q1.parameters()]

    iql.update(_batch())

    for name, parameter in iql.feature_encoder.named_parameters():
        torch.testing.assert_close(parameter, encoder_before[name])
    assert any(
        not torch.equal(before, after) for before, after in zip(q_before, iql.q1.parameters(), strict=True)
    )
    # A frozen encoder must never enter the critic optimizer state.
    assert all(
        id(parameter) not in {id(g) for group in iql.q_optimizer.param_groups for g in group["params"]}
        for parameter in iql.feature_encoder.parameters()
    )


def _dp3_adapter(root: Path) -> DiffusionRLAdapter:
    config = DP3Config(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(34,)),
            "observation.point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(8, 3)),
            "observation.images.wrist_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.wrist_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        device="cpu",
        point_cloud_num_points=8,
        point_cloud_encoder_hidden_dims=(8, 16),
        point_cloud_encoder_output_dim=8,
        state_encoder_hidden_dims=(8,),
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = DP3Policy(config)
    policy.save_pretrained(root)
    stats = {
        "observation.state": {
            "min": torch.full((34,), -2.0),
            "max": torch.full((34,), 2.0),
        },
        "observation.point_cloud": {
            "min": torch.full((8, 3), -1.0),
            "max": torch.full((8, 3), 1.0),
        },
        "observation.images.wrist_left": {
            "mean": torch.zeros(3, 64, 64),
            "std": torch.ones(3, 64, 64),
        },
        "observation.images.wrist_right": {
            "mean": torch.zeros(3, 64, 64),
            "std": torch.ones(3, 64, 64),
        },
        "action": {
            "min": torch.full((20,), -2.0),
            "max": torch.full((20,), 2.0),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return DiffusionRLAdapter(
        CheckpointAdapter.load(root, device="cpu"),
        TraceConfig(num_inference_steps=2, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )


def test_dp3_adapter_freezes_policy_observation_encoder(tmp_path: Path) -> None:
    adapter = _dp3_adapter(tmp_path / "dp3")
    diffusion = adapter.policy.diffusion
    encoder = diffusion.observation_encoder
    assert encoder is not None
    assert any(parameter.requires_grad for parameter in encoder.parameters())

    adapter.freeze_observation_encoder()

    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    actor_optimizer = torch.optim.Adam(
        [parameter for parameter in adapter.policy.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    optimizer_ids = {id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]}
    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    assert encoder_ids.isdisjoint(optimizer_ids)


def test_dp3_adapter_freeze_is_idempotent(tmp_path: Path) -> None:
    adapter = _dp3_adapter(tmp_path / "dp3")
    encoder = adapter.policy.diffusion.observation_encoder

    adapter.freeze_observation_encoder()
    first = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    adapter.freeze_observation_encoder()

    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(parameter, first[name])
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_freeze_observation_encoder_is_noop_for_state_only_policy(tmp_path: Path) -> None:
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = DiffusionPolicy(config)
    policy.save_pretrained(tmp_path / "state")
    stats = {
        "observation.state": {
            "min": torch.full((3,), -2.0),
            "max": torch.full((3,), 2.0),
        },
        "action": {
            "min": torch.tensor([-2.0, 0.0]),
            "max": torch.tensor([2.0, 0.0]),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(tmp_path / "state")
    postprocessor.save_pretrained(tmp_path / "state")
    adapter = DiffusionRLAdapter(
        CheckpointAdapter.load(tmp_path / "state", device="cpu"),
        TraceConfig(num_inference_steps=2, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )
    before = [parameter.detach().clone() for parameter in adapter.policy.parameters()]

    adapter.freeze_observation_encoder()

    # A state-only policy has no learned vision encoder: freezing is a no-op.
    for snapshot, parameter in zip(before, adapter.policy.parameters(), strict=True):
        torch.testing.assert_close(parameter, snapshot)
    assert all(parameter.requires_grad for parameter in adapter.policy.parameters())
