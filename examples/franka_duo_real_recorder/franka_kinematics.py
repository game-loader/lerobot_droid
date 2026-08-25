"""Offline Franka FK helpers.

The real-robot recorder intentionally does not run FK in its capture loop.  This
module is used by ``enrich_fk.py`` after recording, where a calibrated URDF and
the arm mounting transforms are available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _require_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:  # pragma: no cover - depends on the robot host
        raise RuntimeError(
            "Offline FK requires pinocchio. Install the ROS/robot environment "
            "package (often python3-pinocchio) before running enrich_fk.py."
        ) from exc
    return pin


def _as_mount_transform(values: Sequence[float] | None) -> np.ndarray:
    """Return a 4x4 base transform from xyz + quaternion xyzw values."""

    if values is None:
        return np.eye(4, dtype=np.float64)
    values = tuple(float(value) for value in values)
    if len(values) not in (3, 7):
        raise ValueError("A mount transform must contain xyz or xyz+qx,qy,qz,qw")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = values[:3]
    if len(values) == 7:
        qx, qy, qz, qw = values[3:]
        norm = float(np.linalg.norm((qx, qy, qz, qw)))
        if norm <= 1e-12:
            raise ValueError("Mount quaternion must be non-zero")
        qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
        transform[:3, :3] = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
    return transform


def _pose_vector(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to xyz + quaternion xyzw."""

    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        major = int(np.argmax(diagonal))
        if major == 0:
            s = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12))
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif major == 1:
            s = 2.0 * np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12))
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12))
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return np.concatenate((transform[:3, 3], quaternion)).astype(np.float32)


@dataclass
class FrankaUrdfKinematics:
    """FK for one arm in a possibly multi-arm URDF."""

    model: object
    data: object
    pin: object
    joint_names: tuple[str, ...]
    frame_name: str
    mount_transform: np.ndarray

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        joint_names: Iterable[str],
        frame_name: str,
        mount_transform: Sequence[float] | None = None,
    ) -> FrankaUrdfKinematics:
        pin = _require_pinocchio()
        model = pin.buildModelFromUrdf(str(urdf_path))
        frame_id = model.getFrameId(frame_name)
        if frame_id >= model.nframes:
            raise ValueError(f"Frame {frame_name!r} is not present in {urdf_path}")
        names = tuple(joint_names)
        missing = [name for name in names if model.getJointId(name) >= model.njoints]
        if missing:
            raise ValueError(f"Joints missing from {urdf_path}: {missing}")
        return cls(model, model.createData(), pin, names, frame_name, _as_mount_transform(mount_transform))

    def compute(self, joint_positions_rad: Sequence[float]) -> np.ndarray:
        values = np.asarray(joint_positions_rad, dtype=np.float64)
        if values.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {len(self.joint_names)} joint values, got {values.shape}")
        q = np.zeros(self.model.nq, dtype=np.float64)
        for name, value in zip(self.joint_names, values, strict=True):
            joint_id = self.model.getJointId(name)
            q[self.model.idx_qs[joint_id]] = value
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self.mount_transform @ self.data.oMf[self.model.getFrameId(self.frame_name)].homogeneous
        return _pose_vector(transform)


def compute_dual_fk(
    left: FrankaUrdfKinematics,
    right: FrankaUrdfKinematics,
    left_joints: Sequence[float],
    right_joints: Sequence[float],
) -> np.ndarray:
    """Return ``[left_xyzquat, right_xyzquat]`` as float32."""

    return np.concatenate((left.compute(left_joints), right.compute(right_joints))).astype(np.float32)
