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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[dp3])")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.policies.dp3.modeling_dp3 import DP3Policy, PointNetEncoder
from lerobot.policies.dp3.pointcloud import depth_to_point_cloud
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_POINT_CLOUD, OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame, dataset_to_policy_features


def make_tiny_config(**overrides) -> DP3Config:
    kwargs = {
        "device": "cpu",
        "push_to_hub": False,
        "n_obs_steps": 2,
        "horizon": 4,
        "n_action_steps": 2,
        "down_dims": (16,),
        "kernel_size": 3,
        "n_groups": 4,
        "diffusion_step_embed_dim": 8,
        "num_train_timesteps": 4,
        "num_inference_steps": 2,
        "point_cloud_num_points": 16,
        "point_cloud_encoder_hidden_dims": (8, 16),
        "point_cloud_encoder_output_dim": 8,
        "state_encoder_hidden_dims": (8,),
        "pretrained_backbone_weights": None,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(16, 3)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 20, 30)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 12, 18)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    }
    kwargs.update(overrides)
    return DP3Config(**kwargs)


def test_dp3_factory_registration_and_processors():
    config = make_policy_config("dp3", **make_tiny_config().__dict__)

    assert isinstance(config, DP3Config)
    assert get_policy_class("dp3") is DP3Policy
    preprocessor, postprocessor = make_pre_post_processors(config)
    assert preprocessor is not None
    assert postprocessor is not None


def test_make_policy_infers_point_cloud_from_v3_metadata():
    config = make_tiny_config(input_features={}, output_features={})
    metadata = SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (7,), "names": [f"s{i}" for i in range(7)]},
            OBS_POINT_CLOUD: {
                "dtype": "float32",
                "shape": (16, 3),
                "names": ["point", "xyz"],
            },
            "observation.images.head": {
                "dtype": "video",
                "shape": (20, 30, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (12, 18, 3),
                "names": ["height", "width", "channels"],
            },
            ACTION: {"dtype": "float32", "shape": (4,), "names": [f"a{i}" for i in range(4)]},
        },
        stats={},
    )

    policy = make_policy(config, ds_meta=metadata)

    assert isinstance(policy, DP3Policy)
    assert policy.config.point_cloud_feature == PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(16, 3))


def test_point_cloud_dataset_contract_and_inference_frame():
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (7,), "names": [f"s{i}" for i in range(7)]},
        OBS_POINT_CLOUD: {
            "dtype": "float32",
            "shape": (16, 3),
            "names": ["point", "xyz"],
        },
        ACTION: {"dtype": "float32", "shape": (4,), "names": [f"a{i}" for i in range(4)]},
    }
    policy_features = dataset_to_policy_features(features)
    points = np.arange(48, dtype=np.float32).reshape(16, 3)
    frame = build_dataset_frame(
        features,
        {**{f"s{i}": float(i) for i in range(7)}, "point_cloud": points},
        prefix="observation",
    )

    assert policy_features[OBS_POINT_CLOUD].type is FeatureType.POINT_CLOUD
    np.testing.assert_array_equal(frame[OBS_POINT_CLOUD], points)


def test_lerobot_v3_roundtrip_preserves_point_cloud(tmp_path):
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    parquet = pytest.importorskip("pyarrow.parquet", reason="pyarrow is required (install lerobot[dataset])")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        OBS_STATE: {"dtype": "float32", "shape": (7,), "names": [f"s{i}" for i in range(7)]},
        OBS_POINT_CLOUD: {
            "dtype": "float32",
            "shape": (16, 3),
            "names": ["point", "xyz"],
        },
        ACTION: {"dtype": "float32", "shape": (4,), "names": [f"a{i}" for i in range(4)]},
    }
    points = np.arange(48, dtype=np.float32).reshape(16, 3)
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "local/dp3-test",
        fps=30,
        root=root,
        features=features,
        use_videos=False,
    )
    dataset.add_frame(
        {
            OBS_STATE: np.zeros(7, dtype=np.float32),
            OBS_POINT_CLOUD: points,
            ACTION: np.zeros(4, dtype=np.float32),
            "task": "point-cloud smoke test",
        }
    )
    dataset.save_episode()
    dataset.finalize()

    parquet_files = list(root.glob("data/**/*.parquet"))
    assert len(parquet_files) == 1
    table = parquet.read_table(parquet_files[0], columns=[OBS_POINT_CLOUD])
    np.testing.assert_array_equal(np.asarray(table[OBS_POINT_CLOUD][0].as_py()), points)


def test_dp3_rejects_invalid_point_cloud_contract():
    config = make_tiny_config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.STATE, shape=(16, 3)),
        }
    )

    with pytest.raises(ValueError, match="POINT_CLOUD"):
        config.validate_features()


def test_pointnet_is_permutation_invariant_without_subsampling():
    config = make_tiny_config(point_cloud_num_points=None)
    encoder = PointNetEncoder(config).eval()
    points = torch.randn(2, 16, 3)
    permutation = torch.randperm(points.shape[1])

    torch.testing.assert_close(encoder(points), encoder(points[:, permutation]))


def test_depth_to_point_cloud_uses_intrinsics_extrinsics_and_rgb():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    intrinsics = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, 3] = (1.0, 2.0, 3.0)

    points = depth_to_point_cloud(
        depth,
        intrinsics,
        rgb=rgb,
        extrinsics=extrinsics,
        num_points=None,
    )

    np.testing.assert_allclose(
        points[:, :3],
        [[1.0, 2.0, 4.0], [2.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
    )
    np.testing.assert_allclose(points[:, 3:], rgb.reshape(-1, 3) / 255.0)


def test_depth_to_point_cloud_fixed_sampling_is_reproducible():
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    first = depth_to_point_cloud(depth, intrinsics, num_points=8, seed=7)
    second = depth_to_point_cloud(depth, intrinsics, num_points=8, seed=7)

    assert first.shape == (8, 3)
    np.testing.assert_array_equal(first, second)


def test_dp3_forward_and_select_action_ignore_rgb_modalities():
    torch.manual_seed(0)
    config = make_tiny_config()
    policy = DP3Policy(config)
    train_batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 7),
        OBS_POINT_CLOUD: torch.randn(2, config.n_obs_steps, 16, 3),
        ACTION: torch.randn(2, config.horizon, 4),
        "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
        "observation.images.head": torch.randn(2, config.n_obs_steps, 3, 20, 30),
        "observation.images.wrist": torch.randn(2, config.n_obs_steps, 3, 12, 18),
    }

    loss, output = policy(train_batch)
    loss.backward()

    assert output is None
    assert loss.isfinite()
    assert any(parameter.grad is not None for parameter in policy.parameters())

    policy.eval()
    policy.reset()
    action = policy.select_action(
        {
            OBS_STATE: torch.randn(2, 7),
            OBS_POINT_CLOUD: torch.randn(2, 16, 3),
            "observation.images.head": torch.randn(2, 3, 20, 30),
            "observation.images.wrist": torch.randn(2, 3, 12, 18),
        },
        noise=torch.randn(2, config.horizon, 4),
    )
    assert action.shape == (2, 4)


def test_dp3_checkpoint_roundtrip(tmp_path):
    config = make_tiny_config()
    policy = DP3Policy(config).eval()
    policy.save_pretrained(tmp_path)

    restored = DP3Policy.from_pretrained(tmp_path, strict=True)

    assert isinstance(restored.config, DP3Config)
    assert restored.config.type == "dp3"
    assert restored.config.point_cloud_feature == config.point_cloud_feature
    for expected, actual in zip(policy.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
