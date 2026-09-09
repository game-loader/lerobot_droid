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

WRIST_LEFT = "observation.images.wrist_left"
WRIST_RIGHT = "observation.images.wrist_right"


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
        "spatial_softmax_num_keypoints": 4,
        "use_group_norm": True,
        "pretrained_backbone_weights": None,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(34,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(16, 3)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 20, 30)),
            WRIST_LEFT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            WRIST_RIGHT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
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


@pytest.mark.parametrize("encoder", ["ptv3", "sonata"])
def test_experimental_encoders_fail_closed(encoder):
    config = make_tiny_config(**{f"use_{encoder}_encoder": True})
    with pytest.raises(ValueError, match="MLP placeholders"):
        config.validate_features()
    config.experimental_allow_encoder_placeholders = True
    setattr(config, f"{encoder}_model_path", "unvalidated-pretrained.pth")
    with pytest.raises(ValueError, match="cannot be loaded"):
        DP3Policy(config)


def test_dp3_mixed_wrist_resolutions():
    config = make_tiny_config(resize_shape=(32, 32), crop_shape=None)
    config.input_features[WRIST_LEFT] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 32))
    config.input_features[WRIST_RIGHT] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 48))
    policy = DP3Policy(config)
    batch = {
        OBS_STATE: torch.randn(2, 2, 34),
        OBS_POINT_CLOUD: torch.randn(2, 2, 16, 3),
        WRIST_LEFT: torch.rand(2, 2, 3, 40, 32),
        WRIST_RIGHT: torch.rand(2, 2, 3, 32, 48),
        ACTION: torch.randn(2, 4, 20),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    policy.eval()
    assert policy.predict_action_chunk(batch).shape == (2, 2, 20)
    observation = {key: value[:, -1] for key, value in batch.items() if key.startswith("observation.")}
    assert policy.select_action(observation).shape == (2, 20)


def test_make_policy_infers_point_cloud_from_v3_metadata():
    config = make_tiny_config(input_features={}, output_features={})
    metadata = SimpleNamespace(
        features={
            OBS_STATE: {"dtype": "float32", "shape": (34,), "names": [f"s{i}" for i in range(34)]},
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
            WRIST_LEFT: {
                "dtype": "video",
                "shape": (32, 32, 3),
                "names": ["height", "width", "channels"],
            },
            WRIST_RIGHT: {
                "dtype": "video",
                "shape": (32, 32, 3),
                "names": ["height", "width", "channels"],
            },
            ACTION: {"dtype": "float32", "shape": (20,), "names": [f"a{i}" for i in range(20)]},
        },
        stats={},
    )

    policy = make_policy(config, ds_meta=metadata)

    assert isinstance(policy, DP3Policy)
    assert policy.config.point_cloud_feature == PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(16, 3))


def test_point_cloud_dataset_contract_and_inference_frame():
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (34,), "names": [f"s{i}" for i in range(34)]},
        OBS_POINT_CLOUD: {
            "dtype": "float32",
            "shape": (16, 3),
            "names": ["point", "xyz"],
        },
        ACTION: {"dtype": "float32", "shape": (20,), "names": [f"a{i}" for i in range(20)]},
    }
    policy_features = dataset_to_policy_features(features)
    points = np.arange(48, dtype=np.float32).reshape(16, 3)
    frame = build_dataset_frame(
        features,
        {**{f"s{i}": float(i) for i in range(34)}, "point_cloud": points},
        prefix="observation",
    )

    assert policy_features[OBS_POINT_CLOUD].type is FeatureType.POINT_CLOUD
    np.testing.assert_array_equal(frame[OBS_POINT_CLOUD], points)


def test_lerobot_v3_roundtrip_preserves_point_cloud(tmp_path):
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    parquet = pytest.importorskip("pyarrow.parquet", reason="pyarrow is required (install lerobot[dataset])")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        OBS_STATE: {"dtype": "float32", "shape": (34,), "names": [f"s{i}" for i in range(34)]},
        OBS_POINT_CLOUD: {
            "dtype": "float32",
            "shape": (16, 3),
            "names": ["point", "xyz"],
        },
        ACTION: {"dtype": "float32", "shape": (20,), "names": [f"a{i}" for i in range(20)]},
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
            OBS_STATE: np.zeros(34, dtype=np.float32),
            OBS_POINT_CLOUD: points,
            ACTION: np.zeros(20, dtype=np.float32),
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
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(34,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.STATE, shape=(16, 3)),
            WRIST_LEFT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            WRIST_RIGHT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
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


def test_dp3_defaults_match_rl100_pointnet_contract():
    config = DP3Config(device="cpu", push_to_hub=False)

    assert config.expected_state_dim == 34
    assert config.expected_action_dim == 20
    assert config.point_cloud_num_points == 2048
    assert config.point_cloud_encoder_hidden_dims == (64, 128, 256)
    assert config.point_cloud_encoder_output_dim == 64
    assert config.point_cloud_use_layer_norm is True
    assert config.point_cloud_random_subsample is False
    assert config.crop_is_random is False


def test_dp3_rejects_stochastic_visual_or_point_augmentation():
    with pytest.raises(ValueError, match="random image crops"):
        make_tiny_config(crop_is_random=True).validate_features()
    with pytest.raises(ValueError, match="random point-cloud subsampling"):
        make_tiny_config(point_cloud_random_subsample=True).validate_features()


def test_dp3_xyz_pointnet_parameter_count_is_independent_of_point_count():
    config = DP3Config(
        device="cpu",
        push_to_hub=False,
        point_cloud_num_points=2048,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(34,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(2048, 3)),
        },
    )

    point_net = PointNetEncoder(config)

    assert sum(parameter.numel() for parameter in point_net.parameters()) == 59_072


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


def test_depth_to_point_cloud_zero_pads_like_rl100():
    depth = np.ones((1, 1), dtype=np.float32)

    points = depth_to_point_cloud(depth, np.eye(3, dtype=np.float32), num_points=4, seed=7)

    np.testing.assert_array_equal(points[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(points[1:], np.zeros((3, 3), dtype=np.float32))


def test_depth_to_point_cloud_all_invalid_returns_rl100_zero_cloud():
    depth = np.zeros((2, 2), dtype=np.float32)

    points = depth_to_point_cloud(depth, np.eye(3, dtype=np.float32), num_points=4)

    assert points.shape == (4, 3)
    np.testing.assert_array_equal(points, np.zeros((4, 3), dtype=np.float32))


def test_dp3_requires_dual_wrist_rgb_and_fixed_state_action_dimensions():
    missing_wrist = make_tiny_config()
    del missing_wrist.input_features[WRIST_RIGHT]
    with pytest.raises(ValueError, match="wrist_right"):
        missing_wrist.validate_features()

    wrong_state = make_tiny_config()
    wrong_state.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(33,))
    with pytest.raises(ValueError, match="34-dimensional"):
        wrong_state.validate_features()

    wrong_action = make_tiny_config(
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(17,))}
    )
    with pytest.raises(ValueError, match="20-dimensional"):
        wrong_action.validate_features()

    wrong_contract = make_tiny_config(expected_state_dim=40)
    with pytest.raises(ValueError, match="fixed at 34"):
        wrong_contract.validate_features()


def test_dp3_requires_randomly_initialized_independent_resnet18_encoders():
    config = make_tiny_config()
    policy = DP3Policy(config)
    rgb_encoders = policy.diffusion.observation_encoder.rgb_encoders

    assert config.vision_backbone == "resnet18"
    assert config.pretrained_backbone_weights is None
    assert len(rgb_encoders) == 2
    assert rgb_encoders[0] is not rgb_encoders[1]
    assert rgb_encoders[0].backbone is not rgb_encoders[1].backbone
    assert tuple(config.image_features) == (WRIST_LEFT, WRIST_RIGHT)

    pretrained = make_tiny_config(pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1")
    with pytest.raises(ValueError, match="without pretrained weights"):
        pretrained.validate_features()


def test_dp3_forward_and_select_action_use_point_cloud_dual_wrist_rgb_and_state():
    torch.manual_seed(0)
    config = make_tiny_config()
    policy = DP3Policy(config)
    train_batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 34),
        OBS_POINT_CLOUD: torch.randn(2, config.n_obs_steps, 16, 3),
        ACTION: torch.randn(2, config.horizon, 20),
        "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
        "observation.images.head": torch.randn(2, config.n_obs_steps, 3, 20, 30),
        WRIST_LEFT: torch.randn(2, config.n_obs_steps, 3, 32, 32),
        WRIST_RIGHT: torch.randn(2, config.n_obs_steps, 3, 32, 32),
    }

    loss, output = policy(train_batch)
    loss.backward()

    assert output is None
    assert loss.isfinite()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(
        any(parameter.grad is not None for parameter in encoder.parameters())
        for encoder in policy.diffusion.observation_encoder.rgb_encoders
    )

    policy.eval()
    policy.reset()
    action = policy.select_action(
        {
            OBS_STATE: torch.randn(2, 34),
            OBS_POINT_CLOUD: torch.randn(2, 16, 3),
            "observation.images.head": torch.randn(2, 3, 20, 30),
            WRIST_LEFT: torch.randn(2, 3, 32, 32),
            WRIST_RIGHT: torch.randn(2, 3, 32, 32),
        },
        noise=torch.randn(2, config.horizon, 20),
    )
    assert action.shape == (2, 20)


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


def _transformer_backbone_config(**overrides) -> DP3Config:
    kwargs = {
        "wrist_image_keys": [],
        "diffusion_backbone": "transformer",
        "transformer_hidden_dim": 32,
        "transformer_num_layers": 2,
        "transformer_num_heads": 4,
        "transformer_timestep_embed_dim": 16,
        "point_cloud_use_projection": False,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(34,)),
            OBS_POINT_CLOUD: PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(16, 3)),
        },
    }
    kwargs.update(overrides)
    return make_tiny_config(**kwargs)


def test_dp3_unet_backbone_is_default():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

    policy = DP3Policy(make_tiny_config())

    assert policy.config.diffusion_backbone == "unet"
    assert isinstance(policy.diffusion.unet, DiffusionConditionalUnet1d)


def test_dp3_transformer_backbone_forward_and_select_action():
    from lerobot.policies.dp3.modeling_dp3_transformer import DP3DiffusionTransformer

    torch.manual_seed(0)
    config = _transformer_backbone_config()
    policy = DP3Policy(config)
    denoiser = policy.diffusion.unet

    assert isinstance(denoiser, DP3DiffusionTransformer)
    # obs_dim = pooled point width (16, no projection) + state width (8).
    assert denoiser.obs_dim == 16 + 8
    assert denoiser.prefix_len == config.n_obs_steps + 1
    assert denoiser.seq_len == config.n_obs_steps + 1 + config.horizon

    train_batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 34),
        OBS_POINT_CLOUD: torch.randn(2, config.n_obs_steps, 16, 3),
        ACTION: torch.randn(2, config.horizon, 20),
        "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
    }
    loss, output = policy(train_batch)
    loss.backward()

    assert output is None
    assert loss.isfinite()
    assert any(parameter.grad is not None for parameter in denoiser.parameters())

    policy.eval()
    policy.reset()
    action = policy.select_action(
        {OBS_STATE: torch.randn(2, 34), OBS_POINT_CLOUD: torch.randn(2, 16, 3)},
        noise=torch.randn(2, config.horizon, 20),
    )
    assert action.shape == (2, 20)


def test_dp3_transformer_backbone_prefix_attention_mask():
    from lerobot.policies.dp3.modeling_dp3_transformer import build_prefix_attention_mask

    prefix_len, action_len = 3, 4
    mask = build_prefix_attention_mask(prefix_len, action_len, torch.device("cpu"))

    assert mask.shape == (prefix_len + action_len, prefix_len + action_len)
    assert mask.dtype == torch.bool
    # Prefix (observation + timestep) tokens attend only to the prefix.
    assert mask[:prefix_len, :prefix_len].all()
    assert not mask[:prefix_len, prefix_len:].any()
    # Action tokens attend to the prefix and to every other action token.
    assert mask[prefix_len:, :].all()


def test_dp3_transformer_backbone_checkpoint_roundtrip(tmp_path):
    config = _transformer_backbone_config()
    policy = DP3Policy(config).eval()
    policy.save_pretrained(tmp_path)

    restored = DP3Policy.from_pretrained(tmp_path, strict=True)

    assert restored.config.diffusion_backbone == "transformer"
    assert restored.config.point_cloud_use_projection is False
    for expected, actual in zip(policy.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_dp3_transformer_backbone_rejects_invalid_config():
    with pytest.raises(ValueError, match="diffusion_backbone"):
        make_tiny_config(diffusion_backbone="mlp").validate_features()
    with pytest.raises(ValueError, match="divisible"):
        _transformer_backbone_config(transformer_hidden_dim=30, transformer_num_heads=4).validate_features()
