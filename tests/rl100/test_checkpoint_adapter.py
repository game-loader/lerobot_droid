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

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs import NormalizationMode
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from RL.adapters import checkpoint as checkpoint_module
from RL.adapters.checkpoint import (
    CheckpointAdapter,
    _active_action_mask_from_ranges,
    _processor_artifact_fingerprint,
    _validate_normalizer_features,
)
from RL.types import ObservationBatch

REAL_CHECKPOINT = Path(
    "outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model"
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny_diffusion_checkpoint")
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
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
        "observation.state": {
            "min": torch.tensor([-1.0, -2.0, -3.0]),
            "max": torch.tensor([1.0, 2.0, 3.0]),
        },
        "action": {
            "min": torch.tensor([-1.0, 0.0]),
            "max": torch.tensor([1.0, 0.0]),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return root


@pytest.fixture(scope="module")
def checkpoint_adapter(tiny_checkpoint: Path) -> CheckpointAdapter:
    return CheckpointAdapter.load(tiny_checkpoint, device="cpu")


@pytest.fixture(scope="module")
def real_checkpoint_adapter() -> CheckpointAdapter:
    if not REAL_CHECKPOINT.is_dir():
        pytest.skip(f"real checkpoint is unavailable: {REAL_CHECKPOINT}")
    return CheckpointAdapter.load(REAL_CHECKPOINT, device="cpu")


def test_checkpoint_adapter_loads_policy_and_processors(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    assert checkpoint_adapter.policy.config.device == "cpu"
    assert not checkpoint_adapter.policy.training
    assert checkpoint_adapter.active_action_mask.dtype == torch.bool
    assert checkpoint_adapter.active_action_mask.tolist() == [True, False]


def test_checkpoint_adapter_normalization_round_trip(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    raw = torch.tensor([[0.25, 0.0], [-0.5, 0.0]])

    normalized = checkpoint_adapter.normalize_action(raw)
    restored = checkpoint_adapter.unnormalize_action(normalized)

    torch.testing.assert_close(restored, raw, rtol=1e-5, atol=1e-6)


def test_checkpoint_adapter_promotes_half_actions_to_finite_float32(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    normalized = checkpoint_adapter.normalize_action(
        torch.tensor([[-1.0, 0.0]], dtype=torch.float16)
    )

    assert normalized.dtype == torch.float32
    assert torch.isfinite(normalized).all()


def test_checkpoint_adapter_normalizes_batched_state_history(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    observation = ObservationBatch(
        {
            "observation.state": torch.zeros(2, 2, 3, dtype=torch.float16),
            "observation.images.front": torch.zeros(2, 2, 3, 4, 4, dtype=torch.uint8),
        }
    )

    normalized = checkpoint_adapter.normalize_observation(observation)

    assert normalized.features["observation.state"].shape == (2, 2, 3)
    assert normalized.features["observation.state"].dtype == torch.float32
    assert torch.isfinite(normalized.features["observation.state"]).all()
    assert normalized.features["observation.images.front"].dtype == torch.uint8
    torch.testing.assert_close(
        normalized.features["observation.images.front"],
        observation.features["observation.images.front"],
    )


def test_active_action_mask_uses_real_saved_ranges(
    real_checkpoint_adapter: CheckpointAdapter,
) -> None:
    assert real_checkpoint_adapter.active_action_indices.tolist() == [0, 1, 2, 12, 13]


def test_active_action_mask_rejects_inverted_ranges() -> None:
    with pytest.raises(ValueError, match="inverted"):
        _active_action_mask_from_ranges(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            tolerance=1e-8,
        )


def test_processor_fingerprint_is_stable_sha256(checkpoint_adapter: CheckpointAdapter) -> None:
    fingerprint = checkpoint_adapter.processor_fingerprint()
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")
    assert checkpoint_adapter.processor_fingerprint() == fingerprint


def test_processor_fingerprint_tracks_referenced_state_file(tmp_path: Path) -> None:
    state_file = tmp_path / "custom_stats.safetensors"
    state_file.write_bytes(b"first")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": state_file.name}]}),
        encoding="utf-8",
    )
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    before = _processor_artifact_fingerprint(tmp_path)

    state_file.write_bytes(b"second")

    assert _processor_artifact_fingerprint(tmp_path) != before


def test_processor_fingerprint_accepts_relative_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    fingerprint = _processor_artifact_fingerprint(Path("checkpoint"))

    assert len(fingerprint) == 64


def test_checkpoint_load_is_strict_about_missing_weights(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "damaged"
    shutil.copytree(tiny_checkpoint, damaged)
    weights_path = damaged / "model.safetensors"
    weights = load_file(weights_path)
    weights.pop(next(iter(weights)))
    save_file(weights, weights_path)

    with pytest.raises(RuntimeError):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_rejects_mismatched_processor_stats(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "processor_mismatch"
    shutil.copytree(tiny_checkpoint, damaged)
    config = json.loads((damaged / "policy_postprocessor.json").read_text(encoding="utf-8"))
    state_name = config["steps"][0]["state_file"]
    state_path = damaged / state_name
    stats = load_file(state_path)
    stats["action.max"] = torch.tensor([2.0, 0.0])
    save_file(stats, state_path)

    with pytest.raises(ValueError, match="processor action statistics disagree"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_ignores_auxiliary_stat_shape_mismatch(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "auxiliary_stat_shape"
    shutil.copytree(tiny_checkpoint, checkpoint)
    pre_config = json.loads(
        (checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    post_config = json.loads(
        (checkpoint / "policy_postprocessor.json").read_text(encoding="utf-8")
    )
    pre_step = next(
        step for step in pre_config["steps"] if step["registry_name"] == "normalizer_processor"
    )
    post_step = next(
        step
        for step in post_config["steps"]
        if step["registry_name"] == "unnormalizer_processor"
    )
    pre_path = checkpoint / pre_step["state_file"]
    post_path = checkpoint / post_step["state_file"]
    pre_stats = load_file(pre_path)
    post_stats = load_file(post_path)
    pre_stats["action.count"] = torch.tensor(100.0)
    post_stats["action.count"] = torch.tensor([100.0])
    save_file(pre_stats, pre_path)
    save_file(post_stats, post_path)

    adapter = CheckpointAdapter.load(checkpoint, device="cpu")

    assert adapter.active_action_mask.tolist() == [True, False]


def test_checkpoint_rejects_mismatched_state_feature_descriptor(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "state_feature_mismatch"
    shutil.copytree(tiny_checkpoint, damaged)
    config_path = damaged / "policy_preprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalizer = next(
        step for step in config["steps"] if step["registry_name"] == "normalizer_processor"
    )
    normalizer["config"]["features"]["observation.state"]["shape"] = [4]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"observation\.state.*feature"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_rejects_mismatched_state_normalization_mode(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "state_normalization_mismatch"
    shutil.copytree(tiny_checkpoint, damaged)
    config_path = damaged / "policy_preprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalizer = next(
        step for step in config["steps"] if step["registry_name"] == "normalizer_processor"
    )
    normalizer["config"]["norm_map"]["STATE"] = "MEAN_STD"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"STATE.*normalization mode"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_rejects_missing_state_normalization_stats(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "missing_state_stats"
    shutil.copytree(tiny_checkpoint, damaged)
    config = json.loads((damaged / "policy_preprocessor.json").read_text(encoding="utf-8"))
    normalizer = next(
        step for step in config["steps"] if step["registry_name"] == "normalizer_processor"
    )
    state_path = damaged / normalizer["state_file"]
    stats = load_file(state_path)
    for key in tuple(stats):
        if key.startswith("observation.state."):
            stats.pop(key)
    save_file(stats, state_path)

    with pytest.raises(ValueError, match=r"observation\.state.*statistics"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_rejects_mismatched_postprocessor_action_type(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "postprocessor_action_type_mismatch"
    shutil.copytree(tiny_checkpoint, damaged)
    config_path = damaged / "policy_postprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unnormalizer = next(
        step for step in config["steps"] if step["registry_name"] == "unnormalizer_processor"
    )
    unnormalizer["config"]["features"]["action"]["type"] = "STATE"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="postprocessor feature"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_checkpoint_rejects_extra_postprocessor_feature(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "postprocessor_extra_feature"
    shutil.copytree(tiny_checkpoint, damaged)
    config_path = damaged / "policy_postprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unnormalizer = next(
        step for step in config["steps"] if step["registry_name"] == "unnormalizer_processor"
    )
    unnormalizer["config"]["features"]["extra"] = {"type": "STATE", "shape": [1]}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="postprocessor feature keys"):
        CheckpointAdapter.load(damaged, device="cpu")


def test_visual_stats_broadcast_for_observation_image_dot_key() -> None:
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.image.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 8, 8)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        pretrained_backbone_weights=None,
        down_dims=(8,),
    )
    features = {**config.input_features, **config.output_features}
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map={
            FeatureType.STATE: NormalizationMode.MIN_MAX,
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MIN_MAX,
        },
        stats={
            "observation.state": {
                "min": torch.zeros(3),
                "max": torch.ones(3),
            },
            "observation.image.front": {
                "mean": torch.zeros(3),
                "std": torch.ones(3),
            },
            "action": {
                "min": torch.zeros(2),
                "max": torch.ones(2),
            },
        },
    )

    _validate_normalizer_features(config, normalizer)


def test_checkpoint_rejects_external_processor_state_before_loading(
    tiny_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    damaged = tmp_path / "external_processor_state"
    shutil.copytree(tiny_checkpoint, damaged)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"must not be opened")
    config_path = damaged / "policy_preprocessor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalizer = next(
        step for step in config["steps"] if step["registry_name"] == "normalizer_processor"
    )
    normalizer["state_file"] = f"../{outside.name}"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    processor_loaded = False

    def unexpected_processor_load(*_args: object, **_kwargs: object) -> None:
        nonlocal processor_loaded
        processor_loaded = True
        raise AssertionError("processor loader must not run for an external state_file")

    monkeypatch.setattr(
        checkpoint_module, "make_pre_post_processors", unexpected_processor_load
    )

    with pytest.raises(ValueError, match="inside the checkpoint"):
        CheckpointAdapter.load(damaged, device="cpu")
    assert not processor_loaded
