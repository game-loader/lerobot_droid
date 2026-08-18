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

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

STATE_DIM = 39
ACTION_DIM = 14
SUCCESS_MIN_FINAL_LIFT_HEIGHT = 0.015

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
    if not isinstance(components, Mapping) or key not in components:
        raise ValueError(f"info is missing reward_components[{key!r}]")
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
            if not isinstance(components, Mapping) or _TRUE_GRASP_KEY not in components:
                raise ValueError("final_info is missing reward_components['true_grasp']")
            value = components[_TRUE_GRASP_KEY]
        elif path == f"{_REWARD_COMPONENTS_KEY}.{_CLEAR_TABLE_KEY}":
            components = raw_final_info.get(_REWARD_COMPONENTS_KEY)
            if not isinstance(components, Mapping) or _CLEAR_TABLE_KEY not in components:
                raise ValueError("final_info is missing reward_components['clear_table']")
            value = components[_CLEAR_TABLE_KEY]
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
        item = _final_row_value(raw_final_info, int(index), path=path)
        parsed = _vector(item, name=f"final_info[{index}][{path!r}]", num_envs=1)
        result[index] = parsed[0]
    return result


def _validate_contact_values(values: np.ndarray, *, name: str) -> np.ndarray:
    rounded = np.rint(values)
    if not np.all(values == rounded) or np.any(rounded < 0):
        raise ValueError(f"{name} must contain nonnegative integer contact counts")
    return rounded.astype(np.float32)


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
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
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


__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "SUCCESS_MIN_FINAL_LIFT_HEIGHT",
    "CollectedEpisode",
    "EpisodeBatch",
    "StepDiagnostics",
    "batch_seeds",
    "extract_step_diagnostics",
    "recording_mask",
]
