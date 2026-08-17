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

"""Adapt LeRobot v3 episodes into sparse-reward decision transitions."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from RL.config import RLConfig
from RL.types import DecisionBatch, ObservationBatch

_SUCCESS_FIELDS = {
    "true_grasp_ever",
    "clear_table_ever",
    "final_lift_height_m",
    "final_table_contacts",
    "final_hand_contacts",
}
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _required_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool, got {value!r}")
    return value


def _required_integer(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return int(value)


def _required_real(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a real number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return converted


def terminal_success(
    metadata: Mapping[str, Any], *, min_final_lift_height_m: float = 0.015
) -> bool:
    """Apply the collection acceptance criterion to one episode summary."""

    if not isinstance(metadata, Mapping):
        raise ValueError(f"episode metadata must be a mapping, got {type(metadata).__name__}")
    if (
        isinstance(min_final_lift_height_m, bool)
        or not isinstance(min_final_lift_height_m, Real)
        or not math.isfinite(min_final_lift_height_m)
        or min_final_lift_height_m < 0
    ):
        raise ValueError(
            "min_final_lift_height_m must be finite and nonnegative, "
            f"got {min_final_lift_height_m!r}"
        )
    missing = sorted(_SUCCESS_FIELDS.difference(metadata))
    if missing:
        raise ValueError(f"episode metadata is missing {missing}")
    true_grasp_ever = _required_bool(metadata, "true_grasp_ever")
    clear_table_ever = _required_bool(metadata, "clear_table_ever")
    final_lift_height_m = _required_real(metadata, "final_lift_height_m")
    final_table_contacts = _required_integer(metadata, "final_table_contacts")
    final_hand_contacts = _required_integer(metadata, "final_hand_contacts")
    return bool(
        true_grasp_ever
        and clear_table_ever
        and final_lift_height_m >= min_final_lift_height_m
        and final_table_contacts == 0
        and final_hand_contacts > 0
    )


def load_episode_labels(
    summary_path: Path | str, *, expected_episode_count: int
) -> dict[int, bool]:
    """Load and validate terminal success labels from a collection summary."""

    if isinstance(expected_episode_count, bool) or not isinstance(expected_episode_count, Integral):
        raise ValueError(
            "expected_episode_count must be a nonnegative integer, "
            f"got {expected_episode_count!r}"
        )
    expected_episode_count = int(expected_episode_count)
    if expected_episode_count < 0:
        raise ValueError(
            "expected_episode_count must be a nonnegative integer, "
            f"got {expected_episode_count!r}"
        )
    path = Path(summary_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read collection summary {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"collection summary must contain an object, got {payload!r}")
    records = payload.get("episodes")
    if not isinstance(records, list):
        raise ValueError(f"collection summary episodes must be a list, got {records!r}")

    labels: dict[int, bool] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"episode summary record must be a mapping, got {record!r}")
        if "episode_index" not in record:
            raise ValueError(f"episode summary record is missing episode_index: {record!r}")
        episode_index = _required_integer(record, "episode_index")
        if episode_index in labels:
            raise ValueError(f"collection summary contains duplicate episode_index={episode_index}")
        labels[episode_index] = terminal_success(record)

    expected_indices = list(range(expected_episode_count))
    observed_indices = sorted(labels)
    if observed_indices != expected_indices:
        raise ValueError(
            "collection summary episode indices do not match the dataset: "
            f"expected={expected_indices}, got={observed_indices}"
        )
    return labels


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _finite_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty, got shape {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _index_vector(name: str, value: Tensor, *, length: int) -> Tensor:
    _finite_tensor(name, value)
    if value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must have an integer dtype, got {value.dtype}")
    if value.numel() != length:
        raise ValueError(f"{name} must contain {length} entries, got shape {tuple(value.shape)}")
    return value.reshape(length)


def _validate_episode(
    episode: Mapping[str, Tensor], *, state_key: str
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if not isinstance(episode, Mapping):
        raise ValueError(f"episode must be a mapping, got {type(episode).__name__}")
    required = {state_key, "action"}
    missing = sorted(required.difference(episode))
    if missing:
        raise ValueError(f"episode is missing required tensors {missing}")

    states = episode[state_key]
    actions = episode["action"]
    _finite_tensor(state_key, states)
    _finite_tensor("action", actions)
    if states.ndim != 2:
        raise ValueError(f"{state_key} must have shape [frames, state_dim], got {tuple(states.shape)}")
    if actions.ndim != 2:
        raise ValueError(f"action must have shape [frames, action_dim], got {tuple(actions.shape)}")
    if not states.is_floating_point():
        raise ValueError(f"{state_key} must have a floating-point dtype, got {states.dtype}")
    if not actions.is_floating_point():
        raise ValueError(f"action must have a floating-point dtype, got {actions.dtype}")
    length = states.shape[0]
    if length == 0:
        raise ValueError(f"{state_key} must contain at least one frame, got shape {tuple(states.shape)}")
    if actions.shape[0] != length:
        raise ValueError(
            f"action length must match {state_key}: expected={length}, got={actions.shape[0]}"
        )

    observation_keys = {state_key}
    observation_keys.update(key for key in episode if key.startswith("observation."))
    observation_tensors: dict[str, Tensor] = {}
    for key in sorted(observation_keys):
        tensor = episode[key]
        _finite_tensor(key, tensor)
        if tensor.ndim == 0 or tensor.shape[0] != length:
            raise ValueError(
                f"{key} leading dimension must equal episode length {length}, "
                f"got shape {tuple(tensor.shape)}"
            )
        observation_tensors[key] = tensor

    if "frame_index" in episode:
        frame_index = _index_vector("frame_index", episode["frame_index"], length=length)
        if length > 1 and not torch.all(frame_index[1:] > frame_index[:-1]).item():
            raise ValueError(f"frame_index must be strictly increasing, got {frame_index.tolist()}")
    if "episode_index" in episode:
        episode_index = _index_vector("episode_index", episode["episode_index"], length=length)
        if not torch.all(episode_index == episode_index[0]).item():
            raise ValueError(f"episode_index must be constant, got {episode_index.tolist()}")
    return states, actions, observation_tensors


def _history_indices(anchor: int, *, n_obs_steps: int) -> list[int]:
    first = anchor - n_obs_steps + 1
    return [max(0, first + offset) for offset in range(n_obs_steps)]


def _select_history(tensor: Tensor, indices: Sequence[int]) -> Tensor:
    index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index).unsqueeze(0)


def build_episode_decisions(
    episode: Mapping[str, Tensor],
    *,
    success: bool,
    n_obs_steps: int,
    chunk_size: int,
    gamma: float,
    state_key: str = "observation.state",
) -> list[DecisionBatch]:
    """Convert one frame-level episode into fixed-size decision transitions."""

    _positive_int("n_obs_steps", n_obs_steps)
    _positive_int("chunk_size", chunk_size)
    if not isinstance(state_key, str) or not state_key:
        raise ValueError(f"state_key must be a nonempty string, got {state_key!r}")
    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise ValueError(f"gamma must be a finite number in [0, 1], got {gamma!r}")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be a finite number in [0, 1], got {gamma!r}")
    if not isinstance(success, bool):
        raise ValueError(f"success must be a bool, got {success!r}")

    states, actions, observation_tensors = _validate_episode(episode, state_key=state_key)
    length = states.shape[0]
    decisions: list[DecisionBatch] = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        valid_steps = stop - start
        current_indices = _history_indices(start, n_obs_steps=n_obs_steps)
        next_anchor = min(stop, length - 1)
        next_indices = _history_indices(next_anchor, n_obs_steps=n_obs_steps)
        observation = ObservationBatch(
            {
                key: _select_history(tensor, current_indices)
                for key, tensor in observation_tensors.items()
            }
        )
        next_observation = ObservationBatch(
            {
                key: _select_history(tensor, next_indices)
                for key, tensor in observation_tensors.items()
            }
        )

        action_chunk = actions[start:stop]
        if valid_steps < chunk_size:
            padding = action_chunk[-1:].expand(chunk_size - valid_steps, -1)
            action_chunk = torch.cat((action_chunk, padding), dim=0)
        action_valid = torch.arange(chunk_size, device=actions.device) < valid_steps
        terminal = stop == length
        decision = DecisionBatch(
            observation=observation,
            next_observation=next_observation,
            action=action_chunk.unsqueeze(0),
            action_valid=action_valid.unsqueeze(0),
            reward=torch.tensor(
                [[float(success and terminal)]], dtype=torch.float32, device=actions.device
            ),
            done=torch.tensor([[terminal]], dtype=torch.bool, device=actions.device),
            discount=torch.tensor(
                [[gamma**valid_steps]], dtype=torch.float32, device=actions.device
            ),
        )
        decision.validate(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            chunk_size=chunk_size,
            n_obs_steps=n_obs_steps,
            state_key=state_key,
        )
        decisions.append(decision)
    return decisions


def _feature_shape(features: Mapping[str, Any], key: str) -> list[int]:
    if key not in features:
        raise ValueError(f"dataset features are missing {key!r}")
    feature = features[key]
    if not isinstance(feature, Mapping) or "shape" not in feature:
        raise ValueError(f"dataset feature {key!r} has no shape: {feature!r}")
    shape = feature["shape"]
    if not isinstance(shape, (list, tuple)) or not all(isinstance(size, int) for size in shape):
        raise ValueError(f"dataset feature {key!r} has invalid shape {shape!r}")
    return list(shape)


def _stack_values(values: Sequence[Any], *, key: str) -> Tensor:
    tensors: list[Tensor] = []
    for value in values:
        try:
            tensors.append(value if isinstance(value, Tensor) else torch.as_tensor(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dataset value for {key!r} is not tensor-compatible: {value!r}") from exc
    try:
        return torch.stack(tensors)
    except RuntimeError as exc:
        raise ValueError(f"dataset values for {key!r} have inconsistent shapes") from exc


def _load_episode_tensors(
    dataset: LeRobotDataset,
    *,
    start: int,
    stop: int,
    observation_keys: Sequence[str],
) -> dict[str, Tensor]:
    keys = [*observation_keys, "action", "frame_index", "episode_index"]
    raw = dataset.hf_dataset[start:stop]
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset slice [{start}:{stop}] must be a mapping, got {type(raw).__name__}")
    episode = {}
    for key in keys:
        if key not in raw:
            raise ValueError(f"dataset slice [{start}:{stop}] is missing {key!r}")
        value = raw[key]
        if isinstance(value, Tensor):
            episode[key] = value
        elif isinstance(value, Sequence):
            episode[key] = _stack_values(value, key=key)
        else:
            raise ValueError(f"dataset slice value for {key!r} is invalid: {value!r}")
    return episode


def _camera_keys(dataset: LeRobotDataset) -> list[str]:
    configured = getattr(dataset.meta, "camera_keys", None)
    if configured is None:
        configured = [
            key
            for key, feature in dataset.features.items()
            if isinstance(feature, Mapping) and feature.get("dtype") in {"image", "video"}
        ]
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        raise ValueError(f"dataset camera_keys must be a sequence, got {configured!r}")
    camera_keys = sorted(set(configured))
    missing = [key for key in camera_keys if key not in dataset.features]
    if missing:
        raise ValueError(f"dataset camera_keys are missing from features: {missing}")
    return camera_keys


def _metadata_integer(row: Mapping[str, Any], key: str) -> int:
    if key not in row:
        raise ValueError(f"episode metadata row is missing {key!r}: {row!r}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"episode metadata {key!r} must be an integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class _DecisionLocation:
    episode_start: int
    episode_length: int
    decision_start: int


def _load_camera_histories(
    dataset: LeRobotDataset,
    *,
    camera_keys: Sequence[str],
    current_indices: Sequence[int],
    next_indices: Sequence[int],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    rows: dict[int, Mapping[str, Any]] = {}
    for index in dict.fromkeys([*current_indices, *next_indices]):
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise ValueError(f"dataset row {index} must be a mapping, got {type(row).__name__}")
        rows[index] = row

    def stack(indices: Sequence[int]) -> dict[str, Tensor]:
        features: dict[str, Tensor] = {}
        for key in camera_keys:
            missing = next((index for index in indices if key not in rows[index]), None)
            if missing is not None:
                raise ValueError(f"dataset row {missing} is missing camera feature {key!r}")
            tensor = _stack_values([rows[index][key] for index in indices], key=key).unsqueeze(0)
            _finite_tensor(key, tensor)
            features[key] = tensor
        return features

    return stack(current_indices), stack(next_indices)


def collate_decision_batches(batch: Sequence[DecisionBatch]) -> DecisionBatch:
    """Concatenate already-batched decision records along their batch dimension."""

    if not batch:
        raise ValueError("decision batch sequence must be nonempty")
    observation_keys = set(batch[0].observation.features)
    next_observation_keys = set(batch[0].next_observation.features)
    for index, decision in enumerate(batch[1:], start=1):
        if set(decision.observation.features) != observation_keys:
            raise ValueError(f"observation feature keys disagree at batch index {index}")
        if set(decision.next_observation.features) != next_observation_keys:
            raise ValueError(f"next_observation feature keys disagree at batch index {index}")
    observation = ObservationBatch(
        {
            key: torch.cat([decision.observation.features[key] for decision in batch], dim=0)
            for key in sorted(observation_keys)
        }
    )
    next_observation = ObservationBatch(
        {
            key: torch.cat(
                [decision.next_observation.features[key] for decision in batch], dim=0
            )
            for key in sorted(next_observation_keys)
        }
    )
    return DecisionBatch(
        observation=observation,
        next_observation=next_observation,
        action=torch.cat([decision.action for decision in batch], dim=0),
        action_valid=torch.cat([decision.action_valid for decision in batch], dim=0),
        reward=torch.cat([decision.reward for decision in batch], dim=0),
        done=torch.cat([decision.done for decision in batch], dim=0),
        discount=torch.cat([decision.discount for decision in batch], dim=0),
    )


class LeRobotV3DecisionDataset(Dataset[DecisionBatch]):
    """Decision view with eager state tensors and lazy camera histories."""

    def __init__(
        self,
        records: Sequence[DecisionBatch],
        summary: Mapping[str, Any],
        *,
        source_dataset: LeRobotDataset | None = None,
        camera_keys: Sequence[str] = (),
        locations: Sequence[_DecisionLocation] = (),
        n_obs_steps: int = 2,
        chunk_size: int = 32,
    ) -> None:
        if not records:
            raise ValueError("decision dataset records must be nonempty")
        self._records = tuple(records)
        self._summary = copy.deepcopy(dict(summary))
        self._source_dataset = source_dataset
        self._camera_keys = tuple(camera_keys)
        self._locations = tuple(locations)
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        if self._camera_keys:
            if self._source_dataset is None:
                raise ValueError("source_dataset is required when camera_keys are present")
            if len(self._locations) != len(self._records):
                raise ValueError(
                    "camera decision locations must match records: "
                    f"expected={len(self._records)}, got={len(self._locations)}"
                )

    @classmethod
    def from_root(
        cls,
        *,
        dataset_root: str | Path,
        repo_id: str,
        summary_path: str | Path,
        config: RLConfig | None = None,
    ) -> LeRobotV3DecisionDataset:
        config = RLConfig() if config is None else config
        if not isinstance(config, RLConfig):
            raise ValueError(f"config must be an RLConfig, got {type(config).__name__}")
        root = Path(dataset_root)
        dataset = LeRobotDataset(repo_id, root=root)
        state_shape = _feature_shape(dataset.features, config.state_key)
        action_shape = _feature_shape(dataset.features, "action")
        if state_shape != [config.state_dim]:
            raise ValueError(
                f"{config.state_key} shape must be {[config.state_dim]}, got {state_shape}"
            )
        if action_shape != [config.action_dim]:
            raise ValueError(f"action shape must be {[config.action_dim]}, got {action_shape}")

        labels = load_episode_labels(summary_path, expected_episode_count=dataset.num_episodes)
        camera_keys = _camera_keys(dataset)
        observation_keys = {config.state_key}
        observation_keys.update(
            key for key in dataset.features if key.startswith("observation.")
        )
        observation_keys.update(camera_keys)
        observation_keys = sorted(observation_keys)
        raw_observation_keys = [key for key in observation_keys if key not in camera_keys]
        parsed_episode_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
        for raw_row in dataset.meta.episodes:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"episode metadata row must be a mapping, got {raw_row!r}")
            row = dict(raw_row)
            parsed_episode_rows.append(
                (
                    _metadata_integer(row, "episode_index"),
                    _metadata_integer(row, "dataset_from_index"),
                    _metadata_integer(row, "dataset_to_index"),
                    _metadata_integer(row, "length"),
                    row,
                )
            )
        episode_rows = sorted(parsed_episode_rows, key=lambda item: item[0])
        if len(episode_rows) != dataset.num_episodes:
            raise ValueError(
                "episode metadata count must match the dataset: "
                f"expected={dataset.num_episodes}, got={len(episode_rows)}"
            )

        records: list[DecisionBatch] = []
        locations: list[_DecisionLocation] = []
        frame_count = 0
        for expected_episode_index, (episode_index, start, stop, length, _row) in enumerate(
            episode_rows
        ):
            if episode_index != expected_episode_index:
                raise ValueError(
                    "episode metadata indices must be contiguous: "
                    f"expected={expected_episode_index}, got={episode_index}"
                )
            if start < 0 or stop <= start or stop > dataset.num_frames:
                raise ValueError(f"invalid frame bounds for episode {episode_index}: [{start}, {stop})")
            if stop - start != length:
                raise ValueError(
                    f"episode {episode_index} length mismatch: metadata={length}, bounds={stop - start}"
                )
            episode = _load_episode_tensors(
                dataset,
                start=start,
                stop=stop,
                observation_keys=raw_observation_keys,
            )
            observed_episode_index = _index_vector(
                "episode_index", episode["episode_index"], length=length
            )
            if not torch.all(observed_episode_index == episode_index).item():
                raise ValueError(
                    f"episode_index values do not match metadata {episode_index}: "
                    f"got={observed_episode_index.unique().tolist()}"
                )
            episode_decisions = build_episode_decisions(
                episode,
                success=labels[episode_index],
                n_obs_steps=config.n_obs_steps,
                chunk_size=config.chunk_size,
                gamma=config.gamma,
                state_key=config.state_key,
            )
            for decision in episode_decisions:
                decision.validate(
                    state_dim=config.state_dim,
                    action_dim=config.action_dim,
                    chunk_size=config.chunk_size,
                    n_obs_steps=config.n_obs_steps,
                    state_key=config.state_key,
                )
            records.extend(episode_decisions)
            locations.extend(
                _DecisionLocation(
                    episode_start=start,
                    episode_length=length,
                    decision_start=decision_start,
                )
                for decision_start in range(0, length, config.chunk_size)
            )
            frame_count += length

        if frame_count != dataset.num_frames:
            raise ValueError(
                f"episode metadata frame count must be {dataset.num_frames}, got {frame_count}"
            )
        partial_chunk_count = sum(
            not bool(decision.action_valid.all().item()) for decision in records
        )
        summary = {
            "dataset_path": str(root),
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
            "decisions": len(records),
            "positive_labels": sum(labels.values()),
            "negative_labels": len(labels) - sum(labels.values()),
            "state_shape": state_shape,
            "action_shape": action_shape,
            "image_keys": camera_keys,
            "partial_chunk_count": partial_chunk_count,
        }
        return cls(
            records,
            summary,
            source_dataset=dataset if camera_keys else None,
            camera_keys=camera_keys,
            locations=locations,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> DecisionBatch:
        record = self._records[index]
        if not self._camera_keys:
            return record
        location = self._locations[index]
        current_local = _history_indices(
            location.decision_start, n_obs_steps=self._n_obs_steps
        )
        next_anchor = min(
            location.decision_start + self._chunk_size, location.episode_length - 1
        )
        next_local = _history_indices(next_anchor, n_obs_steps=self._n_obs_steps)
        current_indices = [location.episode_start + item for item in current_local]
        next_indices = [location.episode_start + item for item in next_local]
        if self._source_dataset is None:
            raise RuntimeError("camera source dataset is unavailable")
        current_camera, next_camera = _load_camera_histories(
            self._source_dataset,
            camera_keys=self._camera_keys,
            current_indices=current_indices,
            next_indices=next_indices,
        )
        observation_features = dict(record.observation.features)
        observation_features.update(current_camera)
        next_observation_features = dict(record.next_observation.features)
        next_observation_features.update(next_camera)
        return dataclasses.replace(
            record,
            observation=ObservationBatch(observation_features),
            next_observation=ObservationBatch(next_observation_features),
        )

    @staticmethod
    def collate_fn(batch: Sequence[DecisionBatch]) -> DecisionBatch:
        return collate_decision_batches(batch)

    def inspection_summary(self) -> dict[str, Any]:
        return copy.deepcopy(self._summary)
