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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import (
    CheckpointAdapter,
    _active_action_mask_from_ranges,
    _processor_artifact_fingerprint,
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
