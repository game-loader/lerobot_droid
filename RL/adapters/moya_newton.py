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

"""Small adapter for the fused Moya Newton vector environment.

The Newton backend uses Gymnasium's ``SAME_STEP`` autoreset mode.  Therefore
the observation returned by ``step`` for a completed world belongs to the next
episode.  ``select_transition_next_state`` restores the terminal observation
from ``final_obs``/``final_observation`` before an online transition is stored.

The module deliberately imports no Newton backend modules at import time.  A
normal LeRobot command can consequently import :mod:`RL` without installing
Warp/Newton, while ``create_moya_env`` still goes through the public
``MoyaNewtonEnvConfig`` and ``make_env`` entry points.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

STATE_DIM = 39
SUCCESS_MIN_FINAL_LIFT_HEIGHT = 0.015

_STATE_KEYS = ("agent_pos", "observation.state", "state", "environment_state")
_FINAL_OBSERVATION_KEYS = ("final_obs", "final_observation")
_FINAL_MASK_KEYS = (
    "_final_obs",
    "_final_observation",
    "final_obs_mask",
    "final_observation_mask",
)
_FINAL_INFO_MASK_KEYS = ("_final_info", "final_info_mask")
_MISSING = object()


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    """Convert an array-like value without silently accepting ragged objects."""

    # This keeps the adapter usable with CPU tensors in unit tests without
    # importing torch in the normal online path.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover - backend-specific tensor errors
            raise ValueError(f"{name} could not be converted to a NumPy array") from exc
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like, got {type(value).__name__}") from exc


def _validate_num_envs(num_envs: Any) -> int:
    if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or int(num_envs) <= 0:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    return int(num_envs)


def _coerce_bool_vector(value: Any, *, name: str, num_envs: int) -> np.ndarray:
    array = _to_numpy(value, name=name)
    if array.ndim == 0 and num_envs == 1:
        array = array.reshape(1)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape {(num_envs,)}, got {array.shape}")
    if array.dtype == np.bool_:
        return array.astype(np.bool_, copy=False)
    # Gym emits bool arrays, but accepting integer 0/1 makes the helper useful
    # with serialized rollout fixtures without accepting arbitrary truthiness.
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain boolean values") from exc
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return numeric.astype(np.bool_)


def _coerce_state_matrix(value: Any, *, name: str, num_envs: int) -> np.ndarray:
    """Validate one dense ``[num_envs, 39]`` state matrix."""

    array = _to_numpy(value, name=name)
    if array.ndim == 1 and num_envs == 1 and array.shape == (STATE_DIM,):
        array = array.reshape(1, STATE_DIM)
    if array.shape != (num_envs, STATE_DIM):
        raise ValueError(f"{name} must have shape {(num_envs, STATE_DIM)}, got {array.shape}")
    try:
        result = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric state values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _mapping_state_value(value: Mapping[str, Any], *, name: str) -> Any:
    present = [key for key in _STATE_KEYS if key in value]
    if len(present) > 1:
        # Multiple aliases are usually an accidental observation contract
        # mismatch.  Require them to be explicitly disambiguated by the caller.
        raise ValueError(f"{name} contains multiple state keys: {present}")
    if present:
        return value[present[0]]
    if len(value) == 1:
        return next(iter(value.values()))
    raise ValueError(
        f"{name} mapping must contain one of {_STATE_KEYS} or exactly one value; "
        f"got keys={sorted(str(key) for key in value)}"
    )


def _single_state_row(value: Any, *, name: str) -> tuple[np.ndarray | None, bool]:
    """Extract one optional row from an object-array/per-env representation."""

    if value is None:
        return None, False
    if isinstance(value, Mapping):
        return _single_state_row(_mapping_state_value(value, name=name), name=name)
    array = _to_numpy(value, name=name)
    if array.dtype == object and array.ndim == 0:
        return _single_state_row(array.item(), name=name)
    if array.shape != (STATE_DIM,):
        raise ValueError(f"{name} row must have shape {(STATE_DIM,)}, got {array.shape}")
    try:
        row = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} row must contain numeric state values") from exc
    if not np.all(np.isfinite(row)):
        raise ValueError(f"{name} must contain only finite values")
    return row, True


def _extract_state_rows(value: Any, *, name: str, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract rows and a presence mask from dense, object, or dict forms."""

    num_envs = _validate_num_envs(num_envs)
    if isinstance(value, Mapping):
        return _extract_state_rows(_mapping_state_value(value, name=name), name=name, num_envs=num_envs)

    array = _to_numpy(value, name=name)
    # Dense output is the common path.  Object matrices can still be dense if
    # all values are numeric, so try this before treating them as per-env rows.
    if array.shape == (num_envs, STATE_DIM):
        try:
            dense = _coerce_state_matrix(array, name=name, num_envs=num_envs)
        except ValueError:
            if array.dtype != object:
                raise
        else:
            return dense, np.ones(num_envs, dtype=np.bool_)

    if array.ndim == 1 and num_envs == 1 and array.shape == (STATE_DIM,):
        return _coerce_state_matrix(array, name=name, num_envs=1), np.ones(1, dtype=np.bool_)

    # Gym's final observation convention is an object array with one element
    # per world.  A Python list/tuple of per-world dictionaries has the same
    # semantics, so normalize both through the row parser.
    if array.ndim != 1 or array.shape[0] != num_envs:
        raise ValueError(
            f"{name} must have shape {(num_envs, STATE_DIM)} or {(num_envs,)}, got {array.shape}"
        )
    rows = np.zeros((num_envs, STATE_DIM), dtype=np.float32)
    present = np.zeros(num_envs, dtype=np.bool_)
    for index, item in enumerate(array.tolist()):
        row, is_present = _single_state_row(item, name=f"{name}[{index}]")
        if is_present:
            assert row is not None
            rows[index] = row
            present[index] = True
    return rows, present


def _find_aliases(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[tuple[str, Any]]:
    return [(key, mapping[key]) for key in keys if key in mapping]


def _compare_alias_values(
    first: tuple[str, Any], second: tuple[str, Any], *, num_envs: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    first_rows, first_present = _extract_state_rows(first[1], name=f"{name}[{first[0]}]", num_envs=num_envs)
    second_rows, second_present = _extract_state_rows(
        second[1], name=f"{name}[{second[0]}]", num_envs=num_envs
    )
    if not np.array_equal(first_present, second_present) or not np.allclose(
        first_rows[first_present], second_rows[second_present], rtol=0.0, atol=0.0
    ):
        raise ValueError(f"{name} aliases {first[0]!r} and {second[0]!r} disagree")
    return first_rows, first_present


def select_transition_next_state(
    next_observation: Any,
    info: Mapping[str, Any],
    done: Any,
) -> np.ndarray:
    """Return terminal-aware next states for a SAME_STEP vector transition.

    Args:
        next_observation: Observation returned by ``env.step``.  It may be a
            dense ``[N,39]`` array, a LeRobot-style mapping (``agent_pos``), or
            another single-key mapping.
        info: Gymnasium vector info.  ``final_obs`` and
            ``final_observation`` (plus their underscore masks) are accepted.
        done: Boolean ``[N]`` vector, normally ``terminated | truncated``.

    Only rows marked done are replaced.  Non-done rows retain the ordinary
    observation even when an info payload happens to contain a final state for
    them.
    """

    if not isinstance(info, Mapping):
        raise ValueError(f"info must be a mapping, got {type(info).__name__}")
    done_array = _to_numpy(done, name="done")
    if done_array.ndim == 0:
        raise ValueError("done must contain one value per environment")
    if done_array.ndim == 1:
        num_envs = int(done_array.shape[0])
    else:
        raise ValueError(f"done must have shape [num_envs], got {done_array.shape}")
    num_envs = _validate_num_envs(num_envs)
    done_mask = _coerce_bool_vector(done_array, name="done", num_envs=num_envs)

    base, base_present = _extract_state_rows(next_observation, name="next_observation", num_envs=num_envs)
    if not np.all(base_present):
        missing = np.flatnonzero(~base_present).tolist()
        raise ValueError(f"next_observation is missing state rows for environments {missing}")

    final_aliases = _find_aliases(info, _FINAL_OBSERVATION_KEYS)
    mask_aliases = _find_aliases(info, _FINAL_MASK_KEYS)

    if not final_aliases:
        if mask_aliases:
            parsed_masks = [
                _coerce_bool_vector(value, name=f"info[{key}]", num_envs=num_envs)
                for key, value in mask_aliases
            ]
            if any(not np.array_equal(mask, done_mask) for mask in parsed_masks):
                raise ValueError("final observation mask does not match done")
            if np.any(done_mask):
                raise ValueError("done rows are missing final_obs/final_observation")
        elif np.any(done_mask):
            raise ValueError("done rows are missing final_obs/final_observation")
        return base

    if len(final_aliases) == 2:
        final_rows, final_present = _compare_alias_values(
            final_aliases[0], final_aliases[1], num_envs=num_envs, name="final observation"
        )
    else:
        final_key, raw_final = final_aliases[0]
        final_rows, final_present = _extract_state_rows(
            raw_final, name=f"info[{final_key}]", num_envs=num_envs
        )

    if not mask_aliases:
        final_mask = done_mask
    else:
        parsed_masks = [
            _coerce_bool_vector(value, name=f"info[{key}]", num_envs=num_envs) for key, value in mask_aliases
        ]
        final_mask = parsed_masks[0]
        if any(not np.array_equal(mask, final_mask) for mask in parsed_masks[1:]):
            raise ValueError("final observation mask aliases disagree")
        if not np.array_equal(final_mask, done_mask):
            raise ValueError("final observation mask does not match done")

    missing = np.flatnonzero(done_mask & ~final_present)
    if missing.size:
        raise ValueError(f"done rows are missing final observation values: {missing.tolist()}")

    selected = base.copy()
    selected[done_mask] = final_rows[done_mask]
    if not np.all(np.isfinite(selected)):
        raise ValueError("selected next state must contain only finite values")
    return selected


def extract_terminal_success(info: Mapping[str, Any], done: Any) -> np.ndarray:
    """Read terminal success from ``final_info`` for completed worlds only.

    Moya's ordinary top-level info fields describe the post-autoreset state for
    completed worlds.  The accepted success bit is therefore read exclusively
    from each terminal ``final_info[...]["is_success"]`` payload.
    """

    if not isinstance(info, Mapping):
        raise ValueError(f"info must be a mapping, got {type(info).__name__}")
    done_array = _to_numpy(done, name="done")
    if done_array.ndim != 1:
        raise ValueError(f"done must have shape [num_envs], got {done_array.shape}")
    num_envs = _validate_num_envs(int(done_array.shape[0]))
    done_mask = _coerce_bool_vector(done_array, name="done", num_envs=num_envs)

    mask_aliases = _find_aliases(info, _FINAL_INFO_MASK_KEYS)
    if not mask_aliases:
        final_mask = done_mask
    else:
        parsed_masks = [
            _coerce_bool_vector(value, name=f"info[{key}]", num_envs=num_envs) for key, value in mask_aliases
        ]
        final_mask = parsed_masks[0]
        if any(not np.array_equal(mask, final_mask) for mask in parsed_masks[1:]):
            raise ValueError("final info mask aliases disagree")
        if not np.array_equal(final_mask, done_mask):
            raise ValueError("final info mask does not match done")

    raw_final_info = info.get("final_info", _MISSING)
    if raw_final_info is _MISSING:
        if np.any(done_mask):
            raise ValueError("done rows are missing final_info")
        return np.zeros(num_envs, dtype=np.bool_)

    present = np.zeros(num_envs, dtype=np.bool_)
    success = np.zeros(num_envs, dtype=np.bool_)
    if isinstance(raw_final_info, Mapping):
        if "is_success" not in raw_final_info:
            if np.any(done_mask):
                raise ValueError("final_info is missing is_success")
        else:
            success = _coerce_bool_vector(
                raw_final_info["is_success"],
                name="info[final_info][is_success]",
                num_envs=num_envs,
            )
            present[:] = True
    else:
        array = _to_numpy(raw_final_info, name="info[final_info]")
        if array.shape != (num_envs,):
            raise ValueError(f"info[final_info] must have shape {(num_envs,)}, got {array.shape}")
        for index, item in enumerate(array.tolist()):
            if item is None:
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"info[final_info][{index}] must be a mapping or None")
            if "is_success" not in item:
                raise ValueError(f"info[final_info][{index}] is missing is_success")
            value = item["is_success"]
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"info[final_info][{index}][is_success] must be a bool, got {value!r}")
            present[index] = True
            success[index] = bool(value)

    missing = np.flatnonzero(done_mask & ~present)
    if missing.size:
        raise ValueError(f"done rows are missing final_info values: {missing.tolist()}")
    success[~done_mask] = False
    return success


def extract_state_observation(observation: Any, *, num_envs: int | None = None) -> np.ndarray:
    """Extract and validate the 39-D state from a Moya observation."""

    if num_envs is None:
        array = _to_numpy(observation, name="observation")
        if array.ndim == 2:
            num_envs = int(array.shape[0])
        elif array.ndim == 1 and array.shape == (STATE_DIM,):
            num_envs = 1
        elif isinstance(observation, Mapping):
            value = _mapping_state_value(observation, name="observation")
            nested = _to_numpy(value, name="observation state")
            if nested.ndim == 1 and nested.shape == (STATE_DIM,):
                num_envs = 1
            elif nested.ndim >= 1:
                num_envs = int(nested.shape[0])
        if num_envs is None:
            raise ValueError("could not infer num_envs from observation")
    rows, present = _extract_state_rows(observation, name="observation", num_envs=num_envs)
    if not np.all(present):
        raise ValueError("observation is missing one or more state rows")
    return rows


def create_moya_env(
    config: Any | None = None,
    *,
    num_envs: int = 1,
    device: str | None = None,
    episode_length: int | None = None,
    headless: bool | None = None,
    sim_substeps: int | None = None,
    preset: str | None = None,
) -> Any:
    """Create the validated fused Moya vector environment.

    Construction intentionally goes through :class:`MoyaNewtonEnvConfig` and
    :func:`lerobot.envs.make_env`; in particular, ``use_async_envs`` is always
    ``False`` because Moya owns batching inside one fused backend.
    """

    num_envs = _validate_num_envs(num_envs)
    # Lazy import keeps Newton/Warp optional for dataset and offline commands.
    from lerobot.envs import MoyaNewtonEnvConfig, make_env

    if config is None:
        config = MoyaNewtonEnvConfig()
    elif not isinstance(config, MoyaNewtonEnvConfig):
        raise TypeError(f"config must be a MoyaNewtonEnvConfig, got {type(config).__name__}")

    updates: dict[str, Any] = {}
    for name, value in (
        ("device", device),
        ("episode_length", episode_length),
        ("headless", headless),
        ("sim_substeps", sim_substeps),
        ("preset", preset),
    ):
        if value is not None:
            updates[name] = value
    if updates:
        config = dataclasses.replace(config, **updates)
    success_threshold = float(getattr(config, "success_min_final_lift_height", SUCCESS_MIN_FINAL_LIFT_HEIGHT))
    if not np.isfinite(success_threshold) or success_threshold != SUCCESS_MIN_FINAL_LIFT_HEIGHT:
        raise ValueError("Moya RL success_min_final_lift_height must remain fixed at 0.015 meters")

    environments = make_env(config, n_envs=num_envs, use_async_envs=False)
    if not isinstance(environments, Mapping) or not environments:
        raise RuntimeError("Moya make_env must return a non-empty environment mapping")
    suite = environments.get(config.type)
    if suite is None:
        if len(environments) != 1:
            raise RuntimeError(f"Moya make_env returned unexpected suites: {sorted(environments)}")
        suite = next(iter(environments.values()))
    if not isinstance(suite, Mapping) or not suite:
        raise RuntimeError("Moya make_env suite must contain one vector environment")
    if 0 in suite:
        env = suite[0]
    elif len(suite) == 1:
        env = next(iter(suite.values()))
    else:
        raise RuntimeError(f"Moya make_env returned unexpected task ids: {sorted(suite)}")
    if not hasattr(env, "reset") or not hasattr(env, "step"):
        raise TypeError(f"Moya make_env returned a non-vector environment: {type(env).__name__}")
    return env


# Descriptive aliases keep the adapter convenient for callers while retaining
# one implementation and one environment-construction path.
make_moya_newton_env = create_moya_env
build_moya_newton_env = create_moya_env
create_moya_newton_env = create_moya_env
observation_to_state = extract_state_observation


__all__ = [
    "STATE_DIM",
    "SUCCESS_MIN_FINAL_LIFT_HEIGHT",
    "build_moya_newton_env",
    "create_moya_env",
    "create_moya_newton_env",
    "extract_terminal_success",
    "extract_state_observation",
    "make_moya_newton_env",
    "observation_to_state",
    "select_transition_next_state",
]
