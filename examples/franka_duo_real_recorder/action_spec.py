#!/usr/bin/env python3
"""The explicit Franka Duo evaluation action contract.

The recorder's training action is joint-space, but the real-robot evaluation
tool consumes a policy exported for Cartesian control.  Keeping this contract
in a small dependency-free module makes it possible to validate an export
before ROS is started.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ACTION_DIM = 20
EE_DIM = 9
GRIPPER_DIM = 2
EE_ROTATION_REPRESENTATION = "rot6d_rows"


def _as_vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def rot6d_to_matrix(rotation6d: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert RL-100's first-two-rows 6D rotation representation to a matrix."""

    value = np.asarray(rotation6d, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rotation6d must end in 6 values, got {value.shape}")
    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("rotation6d first basis vector is degenerate")
    basis1 = first / first_norm
    second_orthogonal = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("rotation6d vectors are collinear")
    basis2 = second_orthogonal / second_norm
    basis3 = np.cross(basis1, basis2, axis=-1)
    return np.stack((basis1, basis2, basis3), axis=-2).astype(np.float32)


def matrix_to_rot6d(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Flatten the first two matrix rows, matching RL-100 ``mat_to_rot6d``."""

    value = np.asarray(matrix, dtype=np.float32)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {value.shape}")
    return np.ascontiguousarray(value[..., :2, :].reshape(value.shape[:-2] + (6,)))


@dataclasses.dataclass(frozen=True)
class FrankaDuoActionSpec:
    """Shape, representation, and safety limits for a 20D Cartesian action."""

    dimension: int = ACTION_DIM
    ee_dimension: int = EE_DIM
    ee_rotation: str = EE_ROTATION_REPRESENTATION
    gripper_range: tuple[float, float] = (0.0, 1.0)
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    clip_workspace: bool = False

    def __post_init__(self) -> None:
        if self.dimension != ACTION_DIM:
            raise ValueError(f"Franka Duo evaluation requires dimension={ACTION_DIM}")
        if self.ee_dimension != EE_DIM:
            raise ValueError(f"Each Franka Duo EE action must have dimension={EE_DIM}")
        if self.ee_rotation != EE_ROTATION_REPRESENTATION:
            raise ValueError(
                f"Unsupported EE rotation {self.ee_rotation!r}; export with {EE_ROTATION_REPRESENTATION!r}"
            )
        low, high = map(float, self.gripper_range)
        if not math.isfinite(low) or not math.isfinite(high) or not low < high:
            raise ValueError("gripper_range must contain finite low < high")
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must be supplied together")
        if self.workspace_min is not None:
            minimum = _as_vector(self.workspace_min, 3, "workspace_min")
            maximum = _as_vector(self.workspace_max or (), 3, "workspace_max")
            if not np.all(minimum < maximum):
                raise ValueError("workspace_min must be strictly less than workspace_max")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> FrankaDuoActionSpec:
        raw = manifest.get("action_spec")
        if not isinstance(raw, Mapping):
            raise ValueError("manifest.action_spec is required for real-robot evaluation")
        dimension = int(raw.get("dimension", raw.get("action_dim", -1)))
        ee_dimension = int(raw.get("ee_dimension", -1))
        gripper_range_raw = raw.get("gripper_range", (0.0, 1.0))
        if not isinstance(gripper_range_raw, Sequence) or len(gripper_range_raw) != 2:
            raise ValueError("manifest.action_spec.gripper_range must contain two values")
        workspace_min = raw.get("workspace_min")
        workspace_max = raw.get("workspace_max")
        return cls(
            dimension=dimension,
            ee_dimension=ee_dimension,
            ee_rotation=str(raw.get("ee_rotation", "")),
            gripper_range=(float(gripper_range_raw[0]), float(gripper_range_raw[1])),
            workspace_min=tuple(float(item) for item in workspace_min) if workspace_min is not None else None,
            workspace_max=tuple(float(item) for item in workspace_max) if workspace_max is not None else None,
            clip_workspace=bool(raw.get("clip_workspace", False)),
        )

    def validate(self, action: Sequence[float] | np.ndarray, *, clip: bool | None = None) -> np.ndarray:
        """Validate and sanitize one action; returned array is contiguous float32."""

        value = _as_vector(action, self.dimension, "action").copy()
        for offset, side in ((0, "left"), (9, "right")):
            # Checking conversion catches NaNs and collinear vectors before a command reaches the robot.
            rot6d_to_matrix(value[offset + 3 : offset + 9])
            if self.workspace_min is not None:
                minimum = np.asarray(self.workspace_min, dtype=np.float32)
                maximum = np.asarray(self.workspace_max, dtype=np.float32)
                position = value[offset : offset + 3]
                outside = (position < minimum) | (position > maximum)
                should_clip = self.clip_workspace if clip is None else bool(clip)
                if np.any(outside):
                    if not should_clip:
                        raise ValueError(
                            f"{side} EE position {position.tolist()} is outside workspace limits"
                        )
                    value[offset : offset + 3] = np.clip(position, minimum, maximum)
        low, high = map(float, self.gripper_range)
        value[18:20] = np.clip(value[18:20], low, high)
        return np.ascontiguousarray(value, dtype=np.float32)


DEFAULT_ACTION_SPEC = FrankaDuoActionSpec()


def action_spec_manifest(spec: FrankaDuoActionSpec = DEFAULT_ACTION_SPEC) -> dict[str, Any]:
    """Return a JSON-compatible action contract for an export manifest."""

    return {
        "dimension": spec.dimension,
        "ee_dimension": spec.ee_dimension,
        "ee_rotation": spec.ee_rotation,
        "layout": {
            "left_ee": [0, 9],
            "right_ee": [9, 18],
            "left_gripper": 18,
            "right_gripper": 19,
        },
        "ee_format": "xyz + rotation-matrix first two rows flattened row-major",
        "gripper_range": list(spec.gripper_range),
        "workspace_min": list(spec.workspace_min) if spec.workspace_min is not None else None,
        "workspace_max": list(spec.workspace_max) if spec.workspace_max is not None else None,
        "clip_workspace": spec.clip_workspace,
    }
