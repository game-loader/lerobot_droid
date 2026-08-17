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
import json
import math
from collections.abc import Mapping, Sequence
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


def terminal_success(
    metadata: Mapping[str, Any], *, min_final_lift_height_m: float = 0.015
) -> bool:
    """Apply the collection acceptance criterion to one episode summary."""

    if not isinstance(metadata, Mapping):
        raise ValueError(f"episode metadata must be a mapping, got {type(metadata).__name__}")
    if not math.isfinite(min_final_lift_height_m) or min_final_lift_height_m < 0:
        raise ValueError(
            "min_final_lift_height_m must be finite and nonnegative, "
            f"got {min_final_lift_height_m!r}"
        )
    missing = sorted(_SUCCESS_FIELDS.difference(metadata))
    if missing:
        raise ValueError(f"episode metadata is missing {missing}")
    try:
        final_lift_height_m = float(metadata["final_lift_height_m"])
        final_table_contacts = int(metadata["final_table_contacts"])
        final_hand_contacts = int(metadata["final_hand_contacts"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode metadata has invalid acceptance values: {metadata!r}") from exc
    if not math.isfinite(final_lift_height_m):
        raise ValueError(f"final_lift_height_m must be finite, got {final_lift_height_m!r}")
    return bool(
        metadata["true_grasp_ever"]
        and metadata["clear_table_ever"]
        and final_lift_height_m >= min_final_lift_height_m
        and final_table_contacts == 0
        and final_hand_contacts > 0
    )


def load_episode_labels(
    summary_path: Path | str, *, expected_episode_count: int
) -> dict[int, bool]:
    """Load and validate terminal success labels from a collection summary."""

    if isinstance(expected_episode_count, bool) or not isinstance(expected_episode_count, int):
        raise ValueError(
            "expected_episode_count must be a nonnegative integer, "
            f"got {expected_episode_count!r}"
        )
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
        try:
            episode_index = int(record["episode_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode_index must be an integer, got {record['episode_index']!r}") from exc
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
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError(f"gamma must be a finite number in (0, 1], got {gamma!r}")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError(f"gamma must be a finite number in (0, 1], got {gamma!r}")
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
    image_keys: Sequence[str],
) -> dict[str, Tensor]:
    keys = [*observation_keys, "action", "frame_index", "episode_index"]
    if image_keys:
        rows = [dataset[index] for index in range(start, stop)]
        episode: dict[str, Tensor] = {}
        for key in keys:
            missing_at = next((index for index, row in enumerate(rows) if key not in row), None)
            if missing_at is not None:
                raise ValueError(f"dataset row {start + missing_at} is missing {key!r}")
            episode[key] = _stack_values([row[key] for row in rows], key=key)
        return episode

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


class LeRobotV3DecisionDataset(Dataset[DecisionBatch]):
    """In-memory decision view over a validated LeRobot v3 dataset."""

    def __init__(self, records: Sequence[DecisionBatch], summary: Mapping[str, Any]) -> None:
        if not records:
            raise ValueError("decision dataset records must be nonempty")
        self._records = tuple(records)
        self._summary = copy.deepcopy(dict(summary))

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

        labels = load_episode_labels(
            summary_path, expected_episode_count=int(dataset.num_episodes)
        )
        observation_keys = {config.state_key}
        observation_keys.update(
            key for key in dataset.features if key.startswith("observation.")
        )
        observation_keys = sorted(observation_keys)
        image_keys = sorted(
            key for key in observation_keys if key.startswith("observation.images.")
        )
        episode_rows = sorted(
            (dict(row) for row in dataset.meta.episodes), key=lambda row: int(row["episode_index"])
        )
        if len(episode_rows) != dataset.num_episodes:
            raise ValueError(
                "episode metadata count must match the dataset: "
                f"expected={dataset.num_episodes}, got={len(episode_rows)}"
            )

        records: list[DecisionBatch] = []
        frame_count = 0
        for expected_episode_index, row in enumerate(episode_rows):
            try:
                episode_index = int(row["episode_index"])
                start = int(row["dataset_from_index"])
                stop = int(row["dataset_to_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid episode metadata row: {row!r}") from exc
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
                observation_keys=observation_keys,
                image_keys=image_keys,
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
            "image_keys": image_keys,
            "partial_chunk_count": partial_chunk_count,
        }
        return cls(records, summary)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> DecisionBatch:
        return self._records[index]

    def inspection_summary(self) -> dict[str, Any]:
        return copy.deepcopy(self._summary)
