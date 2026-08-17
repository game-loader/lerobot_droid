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

"""Load a LeRobot Diffusion checkpoint without replaying inference-only processors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs import NormalizationMode
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.pipeline import PolicyProcessorPipeline
from RL.types import ObservationBatch


def _single_step(pipeline: PolicyProcessorPipeline, step_type: type, *, label: str):
    matches = [step for step in pipeline.steps if isinstance(step, step_type)]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one {step_type.__name__}, got {len(matches)}")
    return matches[0]


def _processor_artifact_fingerprint(checkpoint: str | Path) -> str:
    path = Path(checkpoint)
    if not path.is_dir():
        raise ValueError(
            "processor_fingerprint currently requires a local checkpoint directory, "
            f"got {checkpoint!r}"
        )
    root = path.resolve()
    path = root
    files: set[Path] = set()
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config_path = path / config_name
        if not config_path.is_file():
            raise ValueError(f"checkpoint is missing processor config {config_name!r}")
        files.add(config_path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid processor config {config_path}: {exc}") from exc
        steps = payload.get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            raise ValueError(f"processor config {config_path} must contain a steps list")
        for step in steps:
            if not isinstance(step, dict) or "state_file" not in step:
                continue
            state_name = step["state_file"]
            if not isinstance(state_name, str) or not state_name:
                raise ValueError(
                    f"processor config {config_path} has invalid state_file={state_name!r}"
                )
            state_path = (path / state_name).resolve()
            if not state_path.is_relative_to(root):
                raise ValueError(
                    f"processor state_file must remain inside the checkpoint: {state_name!r}"
                )
            if not state_path.is_file():
                raise ValueError(f"processor state file is missing: {state_path}")
            files.add(state_path)
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative_name = file.relative_to(root).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _active_action_mask_from_ranges(
    action_min: Tensor, action_max: Tensor, *, tolerance: float
) -> Tensor:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"action range tolerance must be finite and nonnegative, got {tolerance!r}")
    if action_min.shape != action_max.shape or action_min.ndim != 1:
        raise ValueError(
            "action min/max statistics must be matching vectors, "
            f"got min={tuple(action_min.shape)} max={tuple(action_max.shape)}"
        )
    if not torch.isfinite(action_min).all().item() or not torch.isfinite(action_max).all().item():
        raise ValueError("action min/max statistics must contain only finite values")
    if torch.any(action_max < action_min).item():
        raise ValueError("action min/max statistics contain inverted ranges")
    return (action_max - action_min) > tolerance


_REQUIRED_STATS = {
    NormalizationMode.IDENTITY: (),
    NormalizationMode.MEAN_STD: ("mean", "std"),
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


def _expected_normalization_mode(
    config: DiffusionConfig, feature_type: FeatureType
) -> NormalizationMode:
    return config.normalization_mapping.get(feature_type.value, NormalizationMode.IDENTITY)


def _validate_stat_shape(key: str, feature_shape: tuple[int, ...], stat_name: str, value: Tensor) -> None:
    if key.startswith("observation.images.") or key == "observation.image":
        try:
            broadcast_shape = torch.broadcast_shapes(tuple(value.shape), feature_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{key} normalization statistic {stat_name!r} with shape "
                f"{tuple(value.shape)} cannot broadcast to feature shape {feature_shape}"
            ) from exc
        if broadcast_shape != feature_shape:
            raise ValueError(
                f"{key} normalization statistic {stat_name!r} must broadcast exactly to "
                f"feature shape {feature_shape}, got {tuple(value.shape)}"
            )
        return
    if tuple(value.shape) != feature_shape:
        raise ValueError(
            f"{key} normalization statistic {stat_name!r} must have shape {feature_shape}, "
            f"got {tuple(value.shape)}"
        )


def _validate_normalizer_features(
    config: DiffusionConfig, normalizer: NormalizerProcessorStep
) -> None:
    expected_features = {**config.input_features, **config.output_features}
    actual_keys = set(normalizer.features)
    expected_keys = set(expected_features)
    if actual_keys != expected_keys:
        raise ValueError(
            "preprocessor feature keys disagree with the policy config, "
            f"expected={sorted(expected_keys)} actual={sorted(actual_keys)}"
        )

    for key, expected_feature in expected_features.items():
        processor_feature = normalizer.features[key]
        if processor_feature != expected_feature:
            raise ValueError(
                f"{key} processor feature must match the policy config, "
                f"expected={expected_feature!r} actual={processor_feature!r}"
            )

        expected_mode = _expected_normalization_mode(config, expected_feature.type)
        actual_mode = normalizer.norm_map.get(expected_feature.type, NormalizationMode.IDENTITY)
        if actual_mode != expected_mode:
            raise ValueError(
                f"{expected_feature.type.value} normalization mode disagrees with the policy config, "
                f"expected={expected_mode.value} actual={actual_mode.value}"
            )
        try:
            required_stats = _REQUIRED_STATS[expected_mode]
        except KeyError as exc:
            raise ValueError(
                f"unsupported normalization mode for {key}: {expected_mode.value}"
            ) from exc
        if not required_stats:
            continue

        stats = normalizer._tensor_stats.get(key)
        if not stats or any(stat_name not in stats for stat_name in required_stats):
            raise ValueError(
                f"{key} normalization statistics must contain {required_stats} "
                f"for {expected_mode.value}"
            )
        for stat_name in required_stats:
            value = stats[stat_name].detach().cpu()
            if not torch.isfinite(value).all().item():
                raise ValueError(
                    f"{key} normalization statistic {stat_name!r} contains non-finite values"
                )
            _validate_stat_shape(key, tuple(expected_feature.shape), stat_name, value)


def _validate_unnormalizer_features(
    config: DiffusionConfig, unnormalizer: UnnormalizerProcessorStep
) -> None:
    expected_features = config.output_features
    actual_keys = set(unnormalizer.features)
    expected_keys = set(expected_features)
    if actual_keys != expected_keys:
        raise ValueError(
            "postprocessor feature keys disagree with the policy config, "
            f"expected={sorted(expected_keys)} actual={sorted(actual_keys)}"
        )
    for key, expected_feature in expected_features.items():
        processor_feature = unnormalizer.features[key]
        if processor_feature != expected_feature:
            raise ValueError(
                f"{key} postprocessor feature must match the policy config, "
                f"expected={expected_feature!r} actual={processor_feature!r}"
            )


def _validate_processor_compatibility(
    config: DiffusionConfig,
    normalizer: NormalizerProcessorStep,
    unnormalizer: UnnormalizerProcessorStep,
) -> tuple[Tensor, Tensor]:
    _validate_normalizer_features(config, normalizer)
    _validate_unnormalizer_features(config, unnormalizer)
    action_feature = config.action_feature
    if action_feature is None or len(action_feature.shape) != 1:
        raise ValueError(f"policy must define a vector action feature, got {action_feature!r}")
    expected_action_dim = action_feature.shape[0]
    for label, step in (("preprocessor", normalizer), ("postprocessor", unnormalizer)):
        processor_feature = step.features.get("action")
        if processor_feature is None or tuple(processor_feature.shape) != (expected_action_dim,):
            raise ValueError(
                f"{label} action feature must have shape {(expected_action_dim,)}, "
                f"got {processor_feature!r}"
            )
    expected_action_mode = _expected_normalization_mode(config, FeatureType.ACTION)
    normalizer_action_mode = normalizer.norm_map.get(
        FeatureType.ACTION, NormalizationMode.IDENTITY
    )
    unnormalizer_action_mode = unnormalizer.norm_map.get(
        FeatureType.ACTION, NormalizationMode.IDENTITY
    )
    if normalizer_action_mode != expected_action_mode:
        raise ValueError(
            "ACTION normalization mode disagrees with the policy config, "
            f"expected={expected_action_mode.value} actual={normalizer_action_mode.value}"
        )
    if normalizer_action_mode != unnormalizer_action_mode:
        raise ValueError("preprocessor and postprocessor action normalization modes disagree")

    pre_stats = normalizer._tensor_stats.get("action")
    post_stats = unnormalizer._tensor_stats.get("action")
    if not pre_stats or not post_stats:
        raise ValueError("processors are missing action normalization statistics")
    if set(pre_stats) != set(post_stats):
        raise ValueError("processor action statistic fields disagree")
    for key in sorted(pre_stats):
        pre_value = pre_stats[key].detach().cpu()
        post_value = post_stats[key].detach().cpu()
        if pre_value.shape != post_value.shape or not torch.equal(pre_value, post_value):
            raise ValueError(f"processor action statistics disagree for {key!r}")
        if not torch.isfinite(pre_value).all().item():
            raise ValueError(f"processor action statistic {key!r} contains non-finite values")
    if "min" not in pre_stats or "max" not in pre_stats:
        raise ValueError("processors are missing action min/max statistics")
    action_min = pre_stats["min"].detach().to(device="cpu", dtype=torch.float32)
    action_max = pre_stats["max"].detach().to(device="cpu", dtype=torch.float32)
    if action_min.shape != (expected_action_dim,) or action_max.shape != (expected_action_dim,):
        raise ValueError(
            "action min/max statistics must match the policy action dimension, "
            f"expected={(expected_action_dim,)}, got min={tuple(action_min.shape)} "
            f"max={tuple(action_max.shape)}"
        )
    return action_min, action_max


@dataclass
class CheckpointAdapter:
    """Policy plus the exact saved normalization contract used to train it."""

    policy: DiffusionPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    active_action_mask: Tensor
    _normalizer: NormalizerProcessorStep = field(repr=False)
    _unnormalizer: UnnormalizerProcessorStep = field(repr=False)
    _processor_fingerprint: str = field(repr=False)

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | str = "cpu",
        action_range_tolerance: float = 1e-8,
    ) -> CheckpointAdapter:
        if not math.isfinite(action_range_tolerance) or action_range_tolerance < 0:
            raise ValueError(
                "action_range_tolerance must be nonnegative, "
                f"got {action_range_tolerance!r}"
            )
        device = torch.device(device)
        device_name = str(device)
        config = PreTrainedConfig.from_pretrained(
            checkpoint,
            cli_overrides=[f"--device={device_name}"],
        )
        if not isinstance(config, DiffusionConfig):
            raise ValueError(
                f"checkpoint must contain a DiffusionConfig, got {type(config).__name__}"
            )
        config.device = device_name
        policy = DiffusionPolicy.from_pretrained(checkpoint, config=config, strict=True)
        policy.to(device)
        policy.eval()
        device_override = {"device_processor": {"device": device_name}}
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=device_override,
            postprocessor_overrides=device_override,
        )
        normalizer = _single_step(
            preprocessor, NormalizerProcessorStep, label="preprocessor"
        )
        unnormalizer = _single_step(
            postprocessor, UnnormalizerProcessorStep, label="postprocessor"
        )
        action_min, action_max = _validate_processor_compatibility(
            config, normalizer, unnormalizer
        )
        active_action_mask = _active_action_mask_from_ranges(
            action_min, action_max, tolerance=action_range_tolerance
        )
        if not active_action_mask.any().item():
            raise ValueError(
                "saved action statistics contain no active dimensions at "
                f"tolerance={action_range_tolerance}"
            )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            active_action_mask=active_action_mask,
            _normalizer=normalizer,
            _unnormalizer=unnormalizer,
            _processor_fingerprint=_processor_artifact_fingerprint(checkpoint),
        )

    @property
    def active_action_indices(self) -> Tensor:
        return torch.nonzero(self.active_action_mask, as_tuple=False).flatten()

    def normalize_observation(self, observation: ObservationBatch) -> ObservationBatch:
        if not isinstance(observation, ObservationBatch):
            raise ValueError(
                f"observation must be an ObservationBatch, got {type(observation).__name__}"
            )
        prepared = {
            key: value.float() if value.is_floating_point() else value
            for key, value in observation.features.items()
        }
        normalized = self._normalizer._normalize_observation(prepared, inverse=False)
        for key, value in normalized.items():
            if not torch.isfinite(value).all().item():
                raise ValueError(f"normalized observation {key!r} contains non-finite values")
        return ObservationBatch(normalized)

    def normalize_action(self, action: Tensor) -> Tensor:
        self._validate_action(action)
        normalized = self._normalizer._normalize_action(action.float(), inverse=False)
        if not torch.isfinite(normalized).all().item():
            raise ValueError("normalized action contains non-finite values")
        return normalized

    def unnormalize_action(self, action: Tensor) -> Tensor:
        self._validate_action(action)
        unnormalized = self._unnormalizer._normalize_action(action.float(), inverse=True)
        if not torch.isfinite(unnormalized).all().item():
            raise ValueError("unnormalized action contains non-finite values")
        return unnormalized

    def processor_fingerprint(self) -> str:
        return self._processor_fingerprint

    def _validate_action(self, action: Tensor) -> None:
        if not isinstance(action, Tensor):
            raise ValueError(f"action must be a torch.Tensor, got {type(action).__name__}")
        if not action.is_floating_point():
            raise ValueError(f"action must have a floating-point dtype, got {action.dtype}")
        if action.ndim == 0 or action.shape[-1] != self.active_action_mask.numel():
            raise ValueError(
                f"action must end in dimension {self.active_action_mask.numel()}, "
                f"got shape {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all().item():
            raise ValueError("action must contain only finite values")
