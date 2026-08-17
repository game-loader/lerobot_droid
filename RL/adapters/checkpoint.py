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
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
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
    files = sorted(
        {
            *path.glob("policy_preprocessor*"),
            *path.glob("policy_postprocessor*"),
        },
        key=lambda item: item.name,
    )
    if not files:
        raise ValueError(f"checkpoint contains no processor artifacts: {path}")
    digest = hashlib.sha256()
    for file in files:
        if not file.is_file():
            continue
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


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
        if action_range_tolerance < 0:
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
        policy = DiffusionPolicy.from_pretrained(checkpoint, config=config)
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
        action_stats = normalizer._tensor_stats.get("action")
        if not action_stats or "min" not in action_stats or "max" not in action_stats:
            raise ValueError("normalizer is missing action min/max statistics")
        action_min = action_stats["min"].detach().to(device="cpu", dtype=torch.float32)
        action_max = action_stats["max"].detach().to(device="cpu", dtype=torch.float32)
        if action_min.shape != action_max.shape or action_min.ndim != 1:
            raise ValueError(
                "action min/max statistics must be matching vectors, "
                f"got min={tuple(action_min.shape)} max={tuple(action_max.shape)}"
            )
        active_action_mask = (action_max - action_min).abs() > action_range_tolerance
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
        normalized = self._normalizer._normalize_observation(
            dict(observation.features), inverse=False
        )
        return ObservationBatch(normalized)

    def normalize_action(self, action: Tensor) -> Tensor:
        self._validate_action(action)
        return self._normalizer._normalize_action(action, inverse=False)

    def unnormalize_action(self, action: Tensor) -> Tensor:
        self._validate_action(action)
        return self._unnormalizer._normalize_action(action, inverse=True)

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
