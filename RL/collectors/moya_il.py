# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure sparse-reward episode bookkeeping for Moya IL rollouts.

The collector deliberately has no policy, Newton, or LeRobot writer imports.
It receives the raw pre-action state, the action that will be executed, and
diagnostics for that same environment step.  This makes the terminal and
SAME_STEP semantics testable without a simulator.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

import numpy as np

STATE_DIM = 39
ACTION_DIM = 14
# Keep the threshold in the same precision as the simulator state.  The
# resulting Python value is 0.014999999664... and therefore survives an audit
# round trip through JSON without turning an exact 15 mm terminal into a
# failure when the summary is reloaded by the v3 adapter.
SUCCESS_MIN_FINAL_LIFT_HEIGHT = float(np.float32(0.015))

_REWARD_COMPONENTS_KEY = "reward_components"
_TRUE_GRASP_KEY = "true_grasp"
_CLEAR_TABLE_KEY = "clear_table"
_LIFT_KEY = "charger_lift_height"
_TABLE_CONTACTS_KEY = "charger_table_contacts"
_HAND_CONTACTS_KEY = "right_hand_charger_contacts"
_FINAL_INFO_KEYS = ("_final_info", "final_info_mask")


def _array(value: Any, *, name: str) -> np.ndarray:
    """Convert tensors/array-like values without accepting ragged objects."""

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover - backend-specific tensor errors
            raise ValueError(f"{name} could not be converted to NumPy") from exc
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like") from exc


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _vector(
    value: Any,
    *,
    name: str,
    num_envs: int,
    dtype: np.dtype[Any] | type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
    allow_bool: bool = False,
) -> np.ndarray:
    """Validate a dense per-world vector.

    A scalar is accepted only for a one-world environment.  Accepting it in
    larger batches would silently broadcast reset leakage across worlds.
    """

    raw = _array(value, name=name)
    if raw.ndim == 0 and num_envs == 1:
        raw = raw.reshape(1)
    if raw.shape != (num_envs,):
        raise ValueError(f"{name} must have shape {(num_envs,)}, got {raw.shape}")
    if allow_bool:
        if raw.dtype == np.bool_:
            return raw.astype(np.bool_, copy=False)
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain boolean values") from exc
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            raise ValueError(f"{name} must contain only boolean/0/1 values")
        return numeric.astype(np.bool_)
    try:
        converted = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.issubdtype(converted.dtype, np.number) or not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return converted


def _info_vector(info: Mapping[str, Any], key: str, *, num_envs: int) -> np.ndarray:
    if key not in info:
        raise ValueError(f"info is missing required component {key!r}")
    return _vector(info[key], name=f"info[{key!r}]", num_envs=num_envs)


def _component_vector(info: Mapping[str, Any], key: str, *, num_envs: int) -> np.ndarray:
    components = info.get(_REWARD_COMPONENTS_KEY)
    # Moya's dense reward payload intentionally omits components that did not
    # fire on a step.  Missing ``true_grasp``/``clear_table`` means zero; a
    # present component still goes through strict shape and finite checks.
    if components is None:
        return np.zeros(num_envs, dtype=np.float32)
    if not isinstance(components, Mapping):
        raise ValueError("info['reward_components'] must be a mapping")
    if key not in components:
        return np.zeros(num_envs, dtype=np.float32)
    return _vector(
        components[key],
        name=f"info[reward_components][{key!r}]",
        num_envs=num_envs,
    )


def _mask_from_info(info: Mapping[str, Any], *, num_envs: int) -> np.ndarray | None:
    present = [key for key in _FINAL_INFO_KEYS if key in info]
    if not present:
        return None
    masks = [
        _vector(info[key], name=f"info[{key!r}]", num_envs=num_envs, allow_bool=True)
        for key in present
    ]
    first = masks[0]
    if any(not np.array_equal(first, mask) for mask in masks[1:]):
        raise ValueError("final info mask aliases disagree")
    return first


def _final_row_value(raw: Any, index: int, *, path: str) -> Any:
    """Read one value from a collated mapping or object-array final_info."""

    if isinstance(raw, Mapping):
        if path not in raw:
            raise ValueError(f"final_info is missing required component {path!r}")
        return raw[path]
    values = _array(raw, name="info[final_info]")
    if values.ndim != 1:
        expected = (values.shape[0],) if values.ndim else "[num_envs]"
        raise ValueError(f"info[final_info] must have shape {expected}, got {values.shape}")
    if index >= values.shape[0]:
        raise ValueError("final_info has too few environment rows")
    item = values[index]
    if item is None:
        raise ValueError(f"final_info[{index}] is missing")
    if not isinstance(item, Mapping):
        raise ValueError(f"final_info[{index}] is missing required component {path!r}")
    current: Any = item
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"final_info[{index}] is missing required component {path!r}")
        current = current[part]
    return current


def _final_component_vector(
    raw_final_info: Any,
    *,
    path: str,
    done_mask: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Extract a final-info vector and retain ordinary rows from ``fallback``."""

    num_envs = done_mask.shape[0]
    if isinstance(raw_final_info, Mapping):
        value: Any
        if path == f"{_REWARD_COMPONENTS_KEY}.{_TRUE_GRASP_KEY}":
            components = raw_final_info.get(_REWARD_COMPONENTS_KEY)
            if components is None:
                value = np.zeros(num_envs, dtype=np.float32)
            elif not isinstance(components, Mapping):
                raise ValueError("final_info['reward_components'] must be a mapping")
            else:
                value = components.get(_TRUE_GRASP_KEY, np.zeros(num_envs, dtype=np.float32))
        elif path == f"{_REWARD_COMPONENTS_KEY}.{_CLEAR_TABLE_KEY}":
            components = raw_final_info.get(_REWARD_COMPONENTS_KEY)
            if components is None:
                value = np.zeros(num_envs, dtype=np.float32)
            elif not isinstance(components, Mapping):
                raise ValueError("final_info['reward_components'] must be a mapping")
            else:
                value = components.get(_CLEAR_TABLE_KEY, np.zeros(num_envs, dtype=np.float32))
        else:
            if path not in raw_final_info:
                raise ValueError(f"final_info is missing required component {path!r}")
            value = raw_final_info[path]
        final_values = _vector(value, name=f"final_info[{path!r}]", num_envs=num_envs)
        result = fallback.copy()
        result[done_mask] = final_values[done_mask]
        return result

    # Per-environment object-array final_info.  Only terminal rows need a
    # payload; nonterminal rows retain their ordinary top-level diagnostics.
    result = fallback.copy()
    for index in np.flatnonzero(done_mask):
        if path in (
            f"{_REWARD_COMPONENTS_KEY}.{_TRUE_GRASP_KEY}",
            f"{_REWARD_COMPONENTS_KEY}.{_CLEAR_TABLE_KEY}",
        ):
            values = _array(raw_final_info, name="info[final_info]")
            if values.ndim != 1 or int(index) >= values.shape[0]:
                raise ValueError("final_info has too few environment rows")
            row = values[int(index)]
            if row is None:
                item = 0.0
            elif not isinstance(row, Mapping):
                raise ValueError(f"final_info[{int(index)}] must be a mapping")
            else:
                components = row.get(_REWARD_COMPONENTS_KEY)
                component_key = path.split(".", 1)[1]
                if components is None:
                    item = 0.0
                elif not isinstance(components, Mapping):
                    raise ValueError(
                        f"final_info[{int(index)}]['reward_components'] must be a mapping"
                    )
                elif component_key not in components:
                    item = 0.0
                else:
                    item = components[component_key]
        else:
            item = _final_row_value(raw_final_info, int(index), path=path)
        parsed = _vector(item, name=f"final_info[{index}][{path!r}]", num_envs=1)
        result[index] = parsed[0]
    return result


def _validate_contact_values(values: np.ndarray, *, name: str) -> np.ndarray:
    rounded = np.rint(values)
    if not np.all(values == rounded) or np.any(rounded < 0):
        raise ValueError(f"{name} must contain nonnegative integer contact counts")
    return rounded.astype(np.float32)


_SUCCESS_METADATA_FIELDS = {
    "true_grasp_ever",
    "clear_table_ever",
    "final_lift_height_m",
    "final_table_contacts",
    "final_hand_contacts",
}


def _metadata_success(metadata: Mapping[str, Any]) -> bool:
    """Evaluate the five acceptance fields without importing the v3 adapter."""

    missing = sorted(_SUCCESS_METADATA_FIELDS.difference(metadata))
    if missing:
        raise ValueError(f"metadata is missing acceptance fields {missing}")
    for key in ("true_grasp_ever", "clear_table_ever"):
        value = metadata[key]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"metadata.{key} must be a bool, got {value!r}")
    lift = metadata["final_lift_height_m"]
    if isinstance(lift, bool) or not isinstance(lift, Real) or not np.isfinite(float(lift)):
        raise ValueError(f"metadata.final_lift_height_m must be finite, got {lift!r}")
    contacts: dict[str, int] = {}
    for key in ("final_table_contacts", "final_hand_contacts"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"metadata.{key} must be a nonnegative integer, got {value!r}")
        contacts[key] = int(value)
    return bool(
        bool(metadata["true_grasp_ever"])
        and bool(metadata["clear_table_ever"])
        and np.float32(float(lift)) >= np.float32(SUCCESS_MIN_FINAL_LIFT_HEIGHT)
        and contacts["final_table_contacts"] == 0
        and contacts["final_hand_contacts"] > 0
    )


@dataclass(frozen=True)
class StepDiagnostics:
    """Per-world diagnostics associated with one executed environment step."""

    true_grasp: np.ndarray
    clear_table: np.ndarray
    lift_height: np.ndarray
    table_contacts: np.ndarray
    hand_contacts: np.ndarray
    native_done: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "true_grasp": self.true_grasp,
            "clear_table": self.clear_table,
            "lift_height": self.lift_height,
            "table_contacts": self.table_contacts,
            "hand_contacts": self.hand_contacts,
        }
        converted: dict[str, np.ndarray] = {}
        sizes: set[int] = set()
        for name, value in arrays.items():
            array = _array(value, name=name)
            if array.ndim != 1 or array.shape[0] == 0:
                raise ValueError(f"{name} must have shape [num_envs], got {array.shape}")
            try:
                array = np.asarray(array, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            converted[name] = array.copy()
            sizes.add(array.shape[0])
        native_done = _vector(
            self.native_done,
            name="native_done",
            num_envs=next(iter(sizes)),
            allow_bool=True,
        )
        sizes.add(native_done.shape[0])
        if len(sizes) != 1:
            raise ValueError(f"diagnostic vector lengths disagree: {sorted(sizes)}")
        converted["native_done"] = native_done.copy()
        converted["table_contacts"] = _validate_contact_values(
            converted["table_contacts"], name="table_contacts"
        )
        converted["hand_contacts"] = _validate_contact_values(
            converted["hand_contacts"], name="hand_contacts"
        )
        for name, array in converted.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def num_envs(self) -> int:
        return int(self.true_grasp.shape[0])


@dataclass(frozen=True)
class CollectedEpisode:
    """One finalized sparse-reward episode in raw NumPy form."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        states = _array(self.states, name="states")
        actions = _array(self.actions, name="actions")
        rewards = _array(self.rewards, name="rewards")
        dones = _array(self.dones, name="dones")
        truncated = _array(self.truncated, name="truncated")
        if states.ndim != 2 or states.shape[1:] != (STATE_DIM,) or states.shape[0] == 0:
            raise ValueError(f"states must have shape [T, {STATE_DIM}], got {states.shape}")
        frames = states.shape[0]
        if actions.shape != (frames, ACTION_DIM):
            raise ValueError(f"actions must have shape {(frames, ACTION_DIM)}, got {actions.shape}")
        for name, value in (("states", states), ("actions", actions)):
            if value.dtype != np.float32:
                raise ValueError(f"{name} must have dtype float32, got {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        if rewards.shape != (frames, 1) or rewards.dtype != np.float32:
            raise ValueError(f"rewards must have shape {(frames, 1)} and dtype float32")
        if not np.all(np.isfinite(rewards)) or not np.all((rewards == 0.0) | (rewards == 1.0)):
            raise ValueError("rewards must be finite binary values")
        for name, value in (("dones", dones), ("truncated", truncated)):
            if value.shape != (frames, 1) or value.dtype != np.bool_:
                raise ValueError(f"{name} must have shape {(frames, 1)} and dtype bool")
        if not bool(dones[-1, 0]) or bool(dones[:-1].any()):
            raise ValueError("exactly the final episode frame must be done")
        if bool(np.any(truncated & ~dones)):
            raise ValueError("truncated may only be true on a done frame")
        if np.any(rewards[:-1] != 0.0):
            raise ValueError("non-terminal rewards must be zero")
        terminal_reward = float(rewards[-1, 0])
        terminal_truncated = bool(truncated[-1, 0])
        if terminal_reward == 1.0 and terminal_truncated:
            raise ValueError("successful terminal transition cannot be truncated")
        if terminal_reward == 0.0 and not terminal_truncated:
            raise ValueError("failed terminal transition must be truncated")
        if terminal_reward not in (0.0, 1.0):  # defensive; checked above too
            raise ValueError("terminal reward must be either 0 or 1")
        if "success" not in self.metadata:
            raise ValueError("metadata must include success")
        success = self.metadata["success"]
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("metadata.success must be a bool")
        derived_success = _metadata_success(self.metadata)
        if bool(success) != derived_success:
            raise ValueError("metadata.success disagrees with the five acceptance conditions")
        if bool(success) != (terminal_reward == 1.0):
            raise ValueError("metadata.success disagrees with terminal reward")
        if "frames" in self.metadata:
            frames_metadata = self.metadata["frames"]
            if (
                isinstance(frames_metadata, bool)
                or not isinstance(frames_metadata, Integral)
                or int(frames_metadata) != frames
            ):
                raise ValueError(
                    f"metadata.frames disagrees with episode length: expected={frames}, "
                    f"got={frames_metadata!r}"
                )
        for name, value in (("states", states), ("actions", actions), ("rewards", rewards)):
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        for name, value in (("dones", dones), ("truncated", truncated)):
            copy = value.astype(np.bool_, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "metadata", dict(self.metadata))


class EpisodeBatch:
    """State machine for a fused batch of independently recorded worlds."""

    def __init__(self, recording_mask_value: np.ndarray) -> None:
        mask = _array(recording_mask_value, name="recording_mask")
        if mask.ndim != 1 or mask.shape[0] == 0 or mask.dtype != np.bool_:
            raise ValueError(
                f"recording_mask must be a nonempty bool vector, got {mask.shape} {mask.dtype}"
            )
        if not np.any(mask):
            raise ValueError("recording_mask must select at least one environment")
        self.recording_mask = mask.astype(np.bool_, copy=True)
        self.num_envs = int(mask.shape[0])
        self._true_grasp_ever = np.zeros(self.num_envs, dtype=np.bool_)
        self._clear_table_ever = np.zeros(self.num_envs, dtype=np.bool_)
        self._finished = np.zeros(self.num_envs, dtype=np.bool_)
        self._states: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._actions: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._rewards: list[list[float]] = [[] for _ in range(self.num_envs)]
        self._dones: list[list[bool]] = [[] for _ in range(self.num_envs)]
        self._truncated: list[list[bool]] = [[] for _ in range(self.num_envs)]
        self._metadata: list[dict[str, Any] | None] = [None] * self.num_envs
        self._step_count = 0

    @classmethod
    def create(cls, recording_mask: np.ndarray) -> EpisodeBatch:
        return cls(recording_mask)

    def append_step(
        self,
        states_value: np.ndarray,
        actions_value: np.ndarray,
        diagnostics: StepDiagnostics,
    ) -> None:
        states = _array(states_value, name="states")
        actions = _array(actions_value, name="actions")
        if states.shape != (self.num_envs, STATE_DIM) or states.dtype != np.float32:
            raise ValueError(
                f"states must have shape {(self.num_envs, STATE_DIM)} and dtype float32, "
                f"got {states.shape} {states.dtype}"
            )
        if actions.shape != (self.num_envs, ACTION_DIM) or actions.dtype != np.float32:
            raise ValueError(
                f"actions must have shape {(self.num_envs, ACTION_DIM)} and dtype float32, "
                f"got {actions.shape} {actions.dtype}"
            )
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError("states and actions must contain only finite values")
        if not isinstance(diagnostics, StepDiagnostics) or diagnostics.num_envs != self.num_envs:
            raise ValueError(
                f"diagnostics must describe {self.num_envs} environments"
            )

        active = self.recording_mask & ~self._finished
        if not np.any(active):
            raise RuntimeError("all recorded worlds are already finalized")

        # Fold this step's components before checking success.  This is
        # important when the first true-grasp/clear-table signal is terminal.
        self._true_grasp_ever[active] |= diagnostics.true_grasp[active] > 0.0
        self._clear_table_ever[active] |= diagnostics.clear_table[active] > 0.0
        lift = diagnostics.lift_height
        success = (
            self._true_grasp_ever
            & self._clear_table_ever
            & (lift >= np.float32(SUCCESS_MIN_FINAL_LIFT_HEIGHT))
            & (diagnostics.table_contacts == 0.0)
            & (diagnostics.hand_contacts > 0.0)
        )
        success &= active
        native_done = diagnostics.native_done & active
        terminal = success | native_done

        for index in np.flatnonzero(active):
            index = int(index)
            self._states[index].append(states[index].copy())
            self._actions[index].append(actions[index].copy())
            is_success = bool(success[index])
            is_terminal = bool(terminal[index])
            self._rewards[index].append(1.0 if is_success else 0.0)
            self._dones[index].append(is_terminal)
            self._truncated[index].append(bool(native_done[index] and not is_success))
            if is_terminal:
                # Store raw terminal diagnostics for the audit summary.  The
                # reward itself remains the canonical sparse binary label.
                self._finished[index] = True
                self._metadata[index] = {
                    "world_index": index,
                    "frames": len(self._states[index]),
                    "true_grasp_ever": bool(self._true_grasp_ever[index]),
                    "clear_table_ever": bool(self._clear_table_ever[index]),
                    "final_lift_height_m": float(lift[index]),
                    "final_table_contacts": int(diagnostics.table_contacts[index]),
                    "final_hand_contacts": int(diagnostics.hand_contacts[index]),
                    "success": is_success,
                    "terminal_reason": "success" if is_success else "horizon",
                }
        self._step_count += 1

    def complete(self) -> bool:
        """Return whether every world selected by the recording mask is done."""

        return bool(np.all(self._finished[self.recording_mask]))

    @property
    def is_complete(self) -> bool:
        """Property alias useful to callers that prefer attribute-style checks."""

        return self.complete()

    def finalized(self) -> tuple[CollectedEpisode, ...]:
        """Return completed episodes in ascending fused-world order."""

        result: list[CollectedEpisode] = []
        for index in np.flatnonzero(self.recording_mask & self._finished):
            index = int(index)
            metadata = self._metadata[index]
            if metadata is None:  # pragma: no cover - guarded by _finished
                continue
            result.append(
                CollectedEpisode(
                    states=np.stack(self._states[index], axis=0).astype(np.float32),
                    actions=np.stack(self._actions[index], axis=0).astype(np.float32),
                    rewards=np.asarray(self._rewards[index], dtype=np.float32).reshape(-1, 1),
                    dones=np.asarray(self._dones[index], dtype=np.bool_).reshape(-1, 1),
                    truncated=np.asarray(self._truncated[index], dtype=np.bool_).reshape(-1, 1),
                    metadata=metadata,
                )
            )
        return tuple(result)


def extract_step_diagnostics(
    info: Mapping[str, Any],
    *,
    terminated: np.ndarray,
    truncated: np.ndarray,
    num_envs: int,
) -> StepDiagnostics:
    """Extract diagnostics while correcting Moya SAME_STEP terminal rows.

    For a native terminal row the ordinary top-level info belongs to the
    autoreset episode.  ``final_info`` is therefore mandatory and its mask
    must exactly equal ``terminated | truncated``.
    """

    if not isinstance(info, Mapping):
        raise ValueError(f"info must be a mapping, got {type(info).__name__}")
    num_envs = _positive_int("num_envs", num_envs)
    terminated_values = _vector(
        terminated, name="terminated", num_envs=num_envs, allow_bool=True
    )
    truncated_values = _vector(
        truncated, name="truncated", num_envs=num_envs, allow_bool=True
    )
    native_done = np.logical_or(terminated_values, truncated_values)

    # Parse all ordinary values first.  They are required even on terminal
    # rows so a malformed vector cannot be silently hidden by final_info.
    true_grasp = _component_vector(info, _TRUE_GRASP_KEY, num_envs=num_envs)
    clear_table = _component_vector(info, _CLEAR_TABLE_KEY, num_envs=num_envs)
    lift_height = _info_vector(info, _LIFT_KEY, num_envs=num_envs)
    table_contacts = _validate_contact_values(
        _info_vector(info, _TABLE_CONTACTS_KEY, num_envs=num_envs),
        name=_TABLE_CONTACTS_KEY,
    )
    hand_contacts = _validate_contact_values(
        _info_vector(info, _HAND_CONTACTS_KEY, num_envs=num_envs),
        name=_HAND_CONTACTS_KEY,
    )

    final_mask = _mask_from_info(info, num_envs=num_envs)
    raw_final_info = info.get("final_info")
    if np.any(native_done):
        if final_mask is None or not np.array_equal(final_mask, native_done):
            raise ValueError("final info mask does not match native done")
        if raw_final_info is None:
            raise ValueError("native done rows are missing final_info")
        true_grasp = _final_component_vector(
            raw_final_info,
            path=f"{_REWARD_COMPONENTS_KEY}.{_TRUE_GRASP_KEY}",
            done_mask=native_done,
            fallback=true_grasp,
        )
        clear_table = _final_component_vector(
            raw_final_info,
            path=f"{_REWARD_COMPONENTS_KEY}.{_CLEAR_TABLE_KEY}",
            done_mask=native_done,
            fallback=clear_table,
        )
        lift_height = _final_component_vector(
            raw_final_info,
            path=_LIFT_KEY,
            done_mask=native_done,
            fallback=lift_height,
        )
        table_contacts = _validate_contact_values(
            _final_component_vector(
                raw_final_info,
                path=_TABLE_CONTACTS_KEY,
                done_mask=native_done,
                fallback=table_contacts,
            ),
            name=f"final_info[{_TABLE_CONTACTS_KEY}]",
        )
        hand_contacts = _validate_contact_values(
            _final_component_vector(
                raw_final_info,
                path=_HAND_CONTACTS_KEY,
                done_mask=native_done,
                fallback=hand_contacts,
            ),
            name=f"final_info[{_HAND_CONTACTS_KEY}]",
        )
    elif final_mask is not None and np.any(final_mask):
        raise ValueError("final info mask contains rows without native done")

    for name, value in (
        ("true_grasp", true_grasp),
        ("clear_table", clear_table),
        ("lift_height", lift_height),
        ("table_contacts", table_contacts),
        ("hand_contacts", hand_contacts),
    ):
        if value.shape != (num_envs,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} diagnostics must be finite vectors")

    return StepDiagnostics(
        true_grasp=true_grasp,
        clear_table=clear_table,
        lift_height=lift_height,
        table_contacts=table_contacts,
        hand_contacts=hand_contacts,
        native_done=native_done,
    )


def recording_mask(remaining: int, num_envs: int) -> np.ndarray:
    """Select the lowest world indices needed for the final fused batch."""

    remaining = _positive_int("remaining", remaining)
    num_envs = _positive_int("num_envs", num_envs)
    result = np.zeros(num_envs, dtype=np.bool_)
    result[: min(remaining, num_envs)] = True
    return result


def batch_seeds(base_seed: int, batch_index: int, num_envs: int) -> tuple[list[int], int]:
    """Return full-world environment seeds and an independent policy seed."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}")
    if isinstance(batch_index, bool) or not isinstance(batch_index, Integral) or int(batch_index) < 0:
        raise ValueError(f"batch_index must be a nonnegative integer, got {batch_index!r}")
    num_envs = _positive_int("num_envs", num_envs)
    batch_index = int(batch_index)
    base_seed = int(base_seed)
    env_seeds = [base_seed + batch_index * num_envs + index for index in range(num_envs)]
    return env_seeds, base_seed + 1_000_000 + batch_index


class PolicyRunner(Protocol):
    """Minimal policy contract used by the simulator-independent collector."""

    def reset(self) -> None:
        """Reset temporal policy/processor state at the start of a fused batch."""
        raise NotImplementedError

    def select_action(self, raw_states: np.ndarray) -> np.ndarray:
        """Return the postprocessed action that will be passed to ``env.step``."""
        raise NotImplementedError


@dataclass(frozen=True)
class CollectionResult:
    """Collected episodes and deterministic batch audit records."""

    episodes: tuple[CollectedEpisode, ...]
    batches: tuple[dict[str, Any], ...] = ()

    @property
    def episodes_saved(self) -> int:
        return len(self.episodes)


def _bool_batch(value: Any, *, name: str, num_envs: int) -> np.ndarray:
    array = _array(value, name=name)
    if array.ndim == 0 and num_envs == 1:
        array = array.reshape(1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape {(num_envs,)}, got {array.shape}")
    if array.dtype == np.bool_:
        return array.astype(np.bool_, copy=False)
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain boolean values") from exc
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return numeric.astype(np.bool_)


def _split_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        if not isinstance(info, Mapping):
            raise ValueError("environment reset info must be a mapping")
        return observation, info
    return result, {}


def _split_step(result: Any, *, num_envs: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("environment step must return (observation, reward, terminated, truncated, info)")
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, Mapping):
        raise ValueError("environment step info must be a mapping")
    reward_array = _array(reward, name="reward")
    if reward_array.ndim == 0 and num_envs == 1:
        reward_array = reward_array.reshape(1)
    if reward_array.shape != (num_envs,):
        raise ValueError(f"reward must have shape {(num_envs,)}, got {reward_array.shape}")
    reward_array = np.asarray(reward_array, dtype=np.float32)
    if not np.all(np.isfinite(reward_array)):
        raise ValueError("environment reward must contain only finite values")
    return (
        observation,
        reward_array,
        _bool_batch(terminated, name="terminated", num_envs=num_envs),
        _bool_batch(truncated, name="truncated", num_envs=num_envs),
        info,
    )


def _policy_seed(seed: int) -> None:
    """Seed common policy RNGs without making torch a hard import dependency."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coerce_action(action: Any, *, num_envs: int) -> np.ndarray:
    array = _array(action, name="policy action")
    if array.shape != (num_envs, ACTION_DIM):
        raise ValueError(
            f"policy action must have shape {(num_envs, ACTION_DIM)}, got {array.shape}"
        )
    try:
        result = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy action must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("policy action must contain only finite values")
    return result


def collect_rollouts(
    env: Any,
    policy: PolicyRunner,
    *,
    target_episodes: int,
    episode_length: int,
    base_seed: int,
) -> CollectionResult:
    """Collect exactly ``target_episodes`` candidates from a fused env.

    A full fused batch is always inferred and stepped, including the tail
    batch.  Only the lowest world indices selected by ``recording_mask`` are
    retained, which keeps episode ordering and RNG streams independent of the
    requested count.
    """

    if env is None or not hasattr(env, "reset") or not hasattr(env, "step"):
        raise ValueError("env must provide reset() and step()")
    if not hasattr(policy, "reset") or not hasattr(policy, "select_action"):
        raise ValueError("policy must implement reset() and select_action()")
    target_episodes = _positive_int("target_episodes", target_episodes)
    episode_length = _positive_int("episode_length", episode_length)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}")
    num_envs = getattr(env, "num_envs", None)
    num_envs = _positive_int("env.num_envs", num_envs)

    from RL.adapters.moya_newton import extract_state_observation

    episodes: list[CollectedEpisode] = []
    batch_records: list[dict[str, Any]] = []
    batch_index = 0
    while len(episodes) < target_episodes:
        remaining = target_episodes - len(episodes)
        mask = recording_mask(remaining, num_envs)
        env_seeds, policy_seed = batch_seeds(int(base_seed), batch_index, num_envs)
        _policy_seed(policy_seed)
        policy.reset()
        reset_result = env.reset(seed=env_seeds)
        observation, _reset_info = _split_reset(reset_result)
        states = extract_state_observation(observation, num_envs=num_envs)
        batch = EpisodeBatch.create(mask)
        steps = 0
        while not batch.complete() and steps < episode_length:
            pre_action_states = states.copy()
            action = _coerce_action(
                policy.select_action(pre_action_states), num_envs=num_envs
            )
            executed_action = action.copy()
            next_observation, _native_reward, terminated, truncated, info = _split_step(
                env.step(action), num_envs=num_envs
            )
            diagnostics = extract_step_diagnostics(
                info,
                terminated=terminated,
                truncated=truncated,
                num_envs=num_envs,
            )
            steps += 1
            if steps >= episode_length:
                forced_horizon = np.ones(num_envs, dtype=np.bool_)
                # ``extract_step_diagnostics`` must see the backend's actual
                # terminal mask so SAME_STEP final_info remains mandatory only
                # where the backend really autoreset.  The explicit horizon is
                # folded in after extraction for ordinary rows.
                effective_done = diagnostics.native_done | forced_horizon
                diagnostics = StepDiagnostics(
                    true_grasp=diagnostics.true_grasp,
                    clear_table=diagnostics.clear_table,
                    lift_height=diagnostics.lift_height,
                    table_contacts=diagnostics.table_contacts,
                    hand_contacts=diagnostics.hand_contacts,
                    native_done=effective_done,
                )
            batch.append_step(pre_action_states, executed_action, diagnostics)
            states = extract_state_observation(next_observation, num_envs=num_envs)

        if not batch.complete():
            raise RuntimeError(
                "fused rollout did not finalize all recorded worlds within episode_length"
            )
        finalized = batch.finalized()
        if len(finalized) != int(mask.sum()):
            raise RuntimeError("fused rollout finalized an unexpected number of episodes")
        batch_record = {
            "batch_index": batch_index,
            "environment_seeds": list(env_seeds),
            "policy_seed": policy_seed,
            "recording_world_indices": np.flatnonzero(mask).astype(int).tolist(),
            "steps": steps,
        }
        batch_records.append(batch_record)
        for world_index, episode in zip(np.flatnonzero(mask), finalized, strict=True):
            metadata = dict(episode.metadata)
            metadata.update(
                {
                    "episode_index": len(episodes),
                    "episode_length": int(episode.states.shape[0]),
                    "batch_index": batch_index,
                    "world_index": int(world_index),
                    "environment_seed": int(env_seeds[int(world_index)]),
                    "seed": int(env_seeds[int(world_index)]),
                    "policy_seed": int(policy_seed),
                }
            )
            episodes.append(
                CollectedEpisode(
                    states=episode.states,
                    actions=episode.actions,
                    rewards=episode.rewards,
                    dones=episode.dones,
                    truncated=episode.truncated,
                    metadata=metadata,
                )
            )
        batch_index += 1

    return CollectionResult(tuple(episodes[:target_episodes]), tuple(batch_records))


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalar/container values into strict JSON values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, Integral)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("summary contains a non-finite float")
        return converted
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"summary value is not JSON serializable: {type(value).__name__}")


def _write_json(path: Any, payload: Mapping[str, Any]) -> None:
    path = os.fspath(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_published_dataset(dataset_root: Any, *, repo_id: str, episodes: Sequence[CollectedEpisode], fps: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    if getattr(loaded.meta.info, "codebase_version", None) != "v3.0":
        raise ValueError("published dataset is not LeRobot v3.0")
    if loaded.fps != fps or loaded.num_episodes != len(episodes):
        raise ValueError(
            "published dataset counters disagree: "
            f"fps={loaded.fps}, episodes={loaded.num_episodes}, expected={fps},{len(episodes)}"
        )
    expected_features = {
        "observation.state": ((39,), "float32"),
        "action": ((14,), "float32"),
        "next.reward": ((1,), "float32"),
        "next.done": ((1,), "bool"),
        "next.truncated": ((1,), "bool"),
    }
    for key, (shape, dtype) in expected_features.items():
        feature = loaded.features.get(key)
        if feature is None or tuple(feature.get("shape", ())) != shape or feature.get("dtype") != dtype:
            raise ValueError(f"published feature {key!r} disagrees with the canonical schema")
    if any(key.startswith("observation.images") for key in loaded.features):
        raise ValueError("published dataset unexpectedly contains image features")
    root = Path(dataset_root)
    if (root / "videos").exists() or (root / "images").exists():
        raise ValueError("published dataset unexpectedly contains video/image files")
    # Read the raw Arrow columns once.  Indexing ``get_raw_item`` repeatedly
    # can re-enter the dataset formatter for every frame, which is needlessly
    # expensive for the formal 100 x 930-frame collection.
    raw = loaded.hf_dataset[: loaded.num_frames]
    if not isinstance(raw, Mapping):
        raise ValueError(f"published raw dataset slice must be a mapping, got {type(raw).__name__}")
    columns: dict[str, Any] = {}
    for key in (
        "episode_index",
        "observation.state",
        "action",
        "next.reward",
        "next.done",
        "next.truncated",
    ):
        if key not in raw:
            raise ValueError(f"published raw dataset slice is missing {key!r}")
        column = raw[key]
        try:
            column_length = len(column)
        except TypeError as exc:
            raise ValueError(f"published raw column {key!r} is not indexable") from exc
        if column_length != loaded.num_frames:
            raise ValueError(
                f"published raw column {key!r} has {column_length} rows, "
                f"expected {loaded.num_frames}"
            )
        columns[key] = column

    offset = 0
    for index, episode in enumerate(episodes):
        frame_count = int(episode.states.shape[0])
        for row_index in range(frame_count):
            row_offset = offset + row_index
            episode_index = int(np.asarray(columns["episode_index"][row_offset]).reshape(-1)[0])
            if episode_index != index:
                raise ValueError(
                    f"published row {row_offset} has episode_index={episode_index}, expected={index}"
                )
            state = np.asarray(columns["observation.state"][row_offset], dtype=np.float32)
            action = np.asarray(columns["action"][row_offset], dtype=np.float32)
            reward = np.asarray(columns["next.reward"][row_offset], dtype=np.float32).reshape(1)
            done = np.asarray(columns["next.done"][row_offset], dtype=np.bool_).reshape(1)
            trunc = np.asarray(columns["next.truncated"][row_offset], dtype=np.bool_).reshape(1)
            if not np.array_equal(state, episode.states[row_index]) or not np.array_equal(
                action, episode.actions[row_index]
            ):
                raise ValueError(f"episode {index} state/action changed during v3 reload")
            if not np.array_equal(reward, episode.rewards[row_index]) or not np.array_equal(
                done, episode.dones[row_index]
            ) or not np.array_equal(trunc, episode.truncated[row_index]):
                raise ValueError(f"episode {index} sparse fields changed during v3 reload")
        offset += frame_count
    if offset != loaded.num_frames:
        raise ValueError(
            f"published frame count disagrees after reload: expected={loaded.num_frames}, got={offset}"
        )


def publish_collection(
    output_dir: Path,
    *,
    repo_id: str,
    episodes: Sequence[CollectedEpisode],
    summary: Mapping[str, Any],
    fps: int = 60,
 ) -> Path:
    """Write a native v3 dataset through a sibling staging directory.

    The final path is created only after writer finalization, reload
    validation, and a complete summary have all succeeded.  A failure leaves
    the ``.incomplete-*`` directory for post-mortem inspection.
    """

    from pathlib import Path

    output_path = Path(output_dir)
    if ".incomplete" in output_path.name:
        raise ValueError("output_dir must be a final path, not a staging path")
    if output_path.exists():
        raise FileExistsError(f"final collection path already exists: {output_path}")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a nonempty string")
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if isinstance(fps, bool) or not isinstance(fps, Integral) or int(fps) <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps!r}")
    if int(fps) != 60:
        raise ValueError(f"Moya collection fps is fixed at 60, got {fps!r}")
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("episodes must be nonempty")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, CollectedEpisode):
            raise ValueError(f"episodes[{index}] must be a CollectedEpisode")
        metadata_index = episode.metadata.get("episode_index", index)
        if metadata_index != index:
            raise ValueError(f"episode indices must be contiguous from zero, got {metadata_index!r} at {index}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f"{output_path.name}.incomplete-{uuid.uuid4().hex}"
    dataset_root = staging / "dataset"
    summary_path = staging / "collection_summary.json"
    task_value = summary.get("task", "moya_charger_grasp")
    if not isinstance(task_value, str) or not task_value.strip():
        raise ValueError("summary task must be a nonempty string")
    task = task_value
    feature_schema = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
    }
    records: list[dict[str, Any]] = []
    from RL.adapters.lerobot_v3 import terminal_success

    for index, episode in enumerate(episodes):
        record = dict(episode.metadata)
        record["episode_index"] = index
        record["frames"] = int(episode.states.shape[0])
        derived_success = terminal_success(record)
        recorded_success = record.get("success")
        if not isinstance(recorded_success, (bool, np.bool_)):
            raise ValueError(f"episode {index} metadata.success must be a bool")
        if bool(recorded_success) != derived_success:
            raise ValueError(
                f"episode {index} success disagrees with the five acceptance conditions"
            )
        if derived_success != bool(episode.rewards[-1, 0] == 1.0):
            raise ValueError(
                f"episode {index} terminal reward disagrees with its summary conditions"
            )
        records.append(record)
    initial_summary = dict(summary)
    initial_summary.update(
        {
            "complete": False,
            "episodes_saved": len(episodes),
            "fps": int(fps),
            "repo_id": repo_id,
            "episodes": records,
            "success_count": sum(bool(record["success"]) for record in records),
            "failure_count": sum(not bool(record["success"]) for record in records),
        }
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _write_json(summary_path, initial_summary)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            features=feature_schema,
            root=dataset_root,
            use_videos=False,
        )
        for episode in episodes:
            for state, action, reward, done, truncated in zip(
                episode.states,
                episode.actions,
                episode.rewards,
                episode.dones,
                episode.truncated,
                strict=True,
            ):
                dataset.add_frame(
                    {
                        "task": task,
                        "observation.state": state,
                        "action": action,
                        "next.reward": reward,
                        "next.done": done,
                        "next.truncated": truncated,
                    }
                )
            dataset.save_episode()
        dataset.finalize()
        _validate_published_dataset(dataset_root, repo_id=repo_id, episodes=episodes, fps=int(fps))
        complete_summary = dict(initial_summary)
        complete_summary["complete"] = True
        _write_json(summary_path, complete_summary)
        os.replace(staging, output_path)
        try:
            directory_fd = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        # Deliberately retain ``staging``.  It is the only artifact that should
        # remain after an interrupted/failed collection.
        raise
    return output_path


__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "SUCCESS_MIN_FINAL_LIFT_HEIGHT",
    "CollectedEpisode",
    "CollectionResult",
    "EpisodeBatch",
    "PolicyRunner",
    "StepDiagnostics",
    "batch_seeds",
    "extract_step_diagnostics",
    "collect_rollouts",
    "publish_collection",
    "recording_mask",
]
