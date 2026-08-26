#!/usr/bin/env python3
"""ROS camera synchronization and ZED point-cloud input for Franka eval.

ROS is imported only by the executable node in ``eval_franka_duo.py``.  The
cache, decoding, synchronizer, and point sampler in this module are therefore
unit-testable on a workstation without a ROS installation.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lerobot.policies.dp3.pointcloud import depth_to_point_cloud

from .record_franka_duo import (
    _joint_map,
    _stamp_ns,
    build_state,
    depth_msg_to_meters,
    image_msg_to_rgb,
)


@dataclasses.dataclass(frozen=True)
class EvalCameraConfig:
    key: str
    image_topic: str
    width: int
    height: int
    fps: int = 30
    depth_topic: str | None = None
    camera_info_topic: str | None = None
    # ROS ``16UC1`` camera depth is conventionally millimetres; ``32FC1`` is
    # already metres and the decoder ignores this value for that encoding.
    depth_scale: float = 0.001
    depth_registered: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError(f"Invalid camera dimensions/fps for {self.key}")
        if self.depth_scale <= 0 or not math.isfinite(float(self.depth_scale)):
            raise ValueError("depth_scale must be positive and finite")
        if self.depth_topic is not None and not self.depth_registered:
            raise ValueError(
                f"{self.key} depth must be RGB-registered; provide a registered ZED depth topic before eval"
            )


@dataclasses.dataclass(frozen=True)
class PointCloudConfig:
    num_points: int = 512
    channels: int = 3
    sampling: str = "random"
    seed: int = 0
    min_depth: float = 0.05
    max_depth: float = 5.0
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    extrinsics: tuple[float, ...] | None = None
    fps_candidate_limit: int = 4096

    def __post_init__(self) -> None:
        if self.num_points <= 0:
            raise ValueError("pointcloud.num_points must be positive")
        if self.channels not in (3, 6):
            raise ValueError("pointcloud.channels must be 3 (XYZ) or 6 (XYZRGB)")
        if self.sampling not in {"random", "fps"}:
            raise ValueError("pointcloud.sampling must be 'random' or 'fps'")
        if not 0 <= self.min_depth < self.max_depth:
            raise ValueError("pointcloud requires 0 <= min_depth < max_depth")
        if self.workspace_min is not None or self.workspace_max is not None:
            if self.workspace_min is None or self.workspace_max is None:
                raise ValueError("workspace_min and workspace_max must be supplied together")
            lower = np.asarray(self.workspace_min, dtype=np.float32)
            upper = np.asarray(self.workspace_max, dtype=np.float32)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or not np.all(lower < upper)
            ):
                raise ValueError("pointcloud workspace bounds must be finite vec3 lower < upper")
        if self.extrinsics is not None:
            matrix = np.asarray(self.extrinsics, dtype=np.float32)
            if matrix.shape != (16,) or not np.isfinite(matrix).all():
                raise ValueError("pointcloud.extrinsics must contain a finite row-major 4x4 matrix")
            if not np.allclose(matrix.reshape(4, 4)[3], (0, 0, 0, 1)):
                raise ValueError("pointcloud.extrinsics must be homogeneous")
        if self.fps_candidate_limit < self.num_points:
            raise ValueError("fps_candidate_limit must be >= num_points")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> PointCloudConfig:
        raw = manifest.get("pointcloud")
        if not isinstance(raw, Mapping):
            raise ValueError("manifest.pointcloud is required for real-robot evaluation")
        def _vec3(name: str) -> tuple[float, float, float] | None:
            value = raw.get(name)
            if value is None:
                return None
            if not isinstance(value, Sequence) or len(value) != 3:
                raise ValueError(f"manifest.pointcloud.{name} must contain three values")
            return tuple(float(item) for item in value)  # type: ignore[return-value]

        extrinsics = raw.get("extrinsics")
        if extrinsics is not None:
            if not isinstance(extrinsics, Sequence) or len(extrinsics) != 16:
                raise ValueError("manifest.pointcloud.extrinsics must contain 16 values")
            extrinsics = tuple(float(item) for item in extrinsics)
        return cls(
            num_points=int(raw.get("num_points", 512)),
            channels=int(raw.get("channels", 3)),
            sampling=str(raw.get("sampling", "random")),
            seed=int(raw.get("seed", 0)),
            min_depth=float(raw.get("min_depth", 0.05)),
            max_depth=float(raw.get("max_depth", 5.0)),
            workspace_min=_vec3("workspace_min"),
            workspace_max=_vec3("workspace_max"),
            extrinsics=extrinsics,
            fps_candidate_limit=int(raw.get("fps_candidate_limit", 4096)),
        )


@dataclasses.dataclass(frozen=True)
class TimedValue:
    message: Any
    arrival_ns: int
    stamp_ns: int | None


@dataclasses.dataclass(frozen=True)
class SynchronizedObservation:
    stamp_ns: int
    head_rgb: np.ndarray
    head_depth_m: np.ndarray
    wrist_left_rgb: np.ndarray
    wrist_right_rgb: np.ndarray
    point_cloud: np.ndarray
    state: np.ndarray | None
    source_stamps_ns: dict[str, int]


class EvalObservationCache:
    """Bounded ROS callback cache with a condition variable for the sampler."""

    def __init__(self, *, history_size: int = 30):
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._condition = threading.Condition()
        self._revision = 0
        self.images: dict[str, deque[TimedValue]] = {
            key: deque(maxlen=history_size) for key in ("head", "wrist_left", "wrist_right")
        }
        self.depth = deque(maxlen=history_size * 2)
        self.camera_info: TimedValue | None = None
        self.joint_states = deque(maxlen=history_size)

    def _store(self, target: deque[TimedValue], message: Any) -> None:
        with self._condition:
            target.append(TimedValue(message, time.monotonic_ns(), _stamp_ns(message)))
            self._revision += 1
            self._condition.notify_all()

    def store_image(self, key: str, message: Any) -> None:
        if key not in self.images:
            raise KeyError(key)
        self._store(self.images[key], message)

    def store_depth(self, message: Any) -> None:
        self._store(self.depth, message)

    def store_joint_states(self, message: Any) -> None:
        self._store(self.joint_states, message)

    def store_camera_info(self, message: Any) -> None:
        with self._condition:
            self.camera_info = TimedValue(message, time.monotonic_ns(), _stamp_ns(message))
            self._revision += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "revision": self._revision,
                "images": {key: tuple(items) for key, items in self.images.items()},
                "depth": tuple(self.depth),
                "joint_states": tuple(self.joint_states),
                "camera_info": self.camera_info,
            }

    def wait_for_head_after(self, stamp_ns: int, timeout_s: float) -> TimedValue | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while True:
                candidates = [
                    item
                    for item in self.images["head"]
                    if item.stamp_ns is not None and item.stamp_ns > stamp_ns
                ]
                if candidates:
                    return min(candidates, key=lambda item: int(item.stamp_ns))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def wait_for_update(self, revision: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._revision <= revision:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


def _nearest(values: Sequence[TimedValue], target_ns: int, tolerance_ns: int) -> TimedValue | None:
    candidates = [item for item in values if item.stamp_ns is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(int(item.stamp_ns) - target_ns))
    if abs(int(best.stamp_ns) - target_ns) > tolerance_ns:
        return None
    return best


def _farthest_point_sample(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Deterministic CPU FPS with a bounded candidate set for 30 Hz live use."""

    if points.shape[0] <= num_points:
        if points.shape[0] == num_points:
            return points
        rng = np.random.default_rng(seed)
        padding = rng.choice(points.shape[0], num_points - points.shape[0], replace=True)
        return np.concatenate((points, points[padding]), axis=0)
    rng = np.random.default_rng(seed)
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = int(rng.integers(points.shape[0]))
    geometry = points[:, :3]
    distances = np.full(points.shape[0], np.inf, dtype=np.float32)
    for index in range(1, num_points):
        current = geometry[selected[index - 1]]
        distances = np.minimum(distances, np.sum((geometry - current) ** 2, axis=1))
        selected[index] = int(np.argmax(distances))
    return np.ascontiguousarray(points[selected])


def make_point_cloud(
    depth_m: np.ndarray,
    head_rgb: np.ndarray,
    camera_info: Any,
    config: PointCloudConfig,
    *,
    frame_index: int = 0,
) -> np.ndarray:
    """Create XYZ or XYZRGB points from one synchronized ZED pair."""

    depth = np.asarray(depth_m)
    rgb = np.asarray(head_rgb)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"ZED RGB/depth shapes differ: RGB {rgb.shape}, depth {depth.shape}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"head_rgb must be HWC RGB, got {rgb.shape}")
    info_width = int(getattr(camera_info, "width", depth.shape[1]))
    info_height = int(getattr(camera_info, "height", depth.shape[0]))
    if (info_height, info_width) != depth.shape:
        raise ValueError(
            f"ZED CameraInfo dimensions {(info_height, info_width)} do not match depth {depth.shape}"
        )
    info_k = getattr(camera_info, "k", None)
    if info_k is None and isinstance(camera_info, Mapping):
        info_k = camera_info.get("k")
    if info_k is None:
        raise ValueError("ZED CameraInfo.k is required for point-cloud projection")
    if np.asarray(info_k, dtype=np.float32).size != 9:
        raise ValueError("ZED CameraInfo.k must contain nine values")
    extrinsics = None if config.extrinsics is None else np.asarray(config.extrinsics, dtype=np.float32).reshape(4, 4)
    color = rgb if config.channels == 6 else None
    seed = int(config.seed) + int(frame_index)
    if config.sampling == "random":
        return depth_to_point_cloud(
            depth,
            info_k,
            depth_scale=1.0,
            rgb=color,
            extrinsics=extrinsics,
            workspace_min=config.workspace_min,
            workspace_max=config.workspace_max,
            min_depth=config.min_depth,
            max_depth=config.max_depth,
            num_points=config.num_points,
            seed=seed,
        )
    all_points = depth_to_point_cloud(
        depth,
        info_k,
        depth_scale=1.0,
        rgb=color,
        extrinsics=extrinsics,
        workspace_min=config.workspace_min,
        workspace_max=config.workspace_max,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        num_points=None,
    )
    if all_points.shape[0] > config.fps_candidate_limit:
        candidate_indices = np.linspace(
            0, all_points.shape[0] - 1, config.fps_candidate_limit, dtype=np.int64
        )
        all_points = all_points[candidate_indices]
    return _farthest_point_sample(all_points, config.num_points, seed)


class SynchronizedObservationReader:
    """Match one head RGB frame with depth, wrists, and optional joint state."""

    def __init__(
        self,
        cache: EvalObservationCache,
        cameras: Mapping[str, EvalCameraConfig],
        pointcloud: PointCloudConfig,
        *,
        rgb_tolerance_ms: float = 45.0,
        depth_tolerance_ms: float = 16.0,
        state_tolerance_ms: float = 50.0,
        max_message_age_ms: float = 150.0,
        sync_wait_timeout_ms: float = 75.0,
        state_gripper_closed: float | None = None,
        state_gripper_open: float | None = None,
    ):
        required = {"head", "wrist_left", "wrist_right"}
        if set(cameras) != required:
            raise ValueError(f"eval requires cameras {sorted(required)}, got {sorted(cameras)}")
        self.cache = cache
        self.cameras = dict(cameras)
        self.pointcloud = pointcloud
        self.rgb_tolerance_ns = int(float(rgb_tolerance_ms) * 1_000_000)
        self.depth_tolerance_ns = int(float(depth_tolerance_ms) * 1_000_000)
        self.state_tolerance_ns = int(float(state_tolerance_ms) * 1_000_000)
        self.max_message_age_ns = int(float(max_message_age_ms) * 1_000_000)
        self.sync_wait_timeout_s = float(sync_wait_timeout_ms) / 1000.0
        self.state_gripper_closed = state_gripper_closed
        self.state_gripper_open = state_gripper_open
        self._last_head_stamp = -1
        self._frame_index = 0

    @property
    def last_head_stamp(self) -> int:
        return self._last_head_stamp

    def _select(self, target_ns: int, *, require_state: bool) -> dict[str, TimedValue]:
        deadline = time.monotonic() + self.sync_wait_timeout_s
        last_error = "required ROS messages are not available"
        while time.monotonic() < deadline:
            snapshot = self.cache.snapshot()
            selected: dict[str, TimedValue] = {}
            selected["head"] = _nearest(snapshot["images"]["head"], target_ns, self.rgb_tolerance_ns) or None  # type: ignore[assignment]
            selected["wrist_left"] = _nearest(snapshot["images"]["wrist_left"], target_ns, self.rgb_tolerance_ns) or None  # type: ignore[assignment]
            selected["wrist_right"] = _nearest(snapshot["images"]["wrist_right"], target_ns, self.rgb_tolerance_ns) or None  # type: ignore[assignment]
            selected["depth"] = _nearest(snapshot["depth"], target_ns, self.depth_tolerance_ns) or None  # type: ignore[assignment]
            if any(value is None for value in selected.values()):
                last_error = "head/depth/wrist timestamp skew exceeds configured tolerance"
            else:
                now = time.monotonic_ns()
                if any(now - value.arrival_ns > self.max_message_age_ns for value in selected.values()):
                    last_error = "one or more synchronized ROS messages are stale"
                elif snapshot["camera_info"] is None:
                    last_error = "ZED CameraInfo has not arrived"
                else:
                    state = _nearest(snapshot["joint_states"], target_ns, self.state_tolerance_ns)
                    if state is not None:
                        selected["state"] = state
                    if require_state and state is None:
                        last_error = "state timestamp skew exceeds configured tolerance"
                    else:
                        return selected
            self.cache.wait_for_update(int(snapshot["revision"]), max(0.0, deadline - time.monotonic()))
        raise TimeoutError(last_error)

    def next(self, timeout_s: float = 1.0, *, require_state: bool = False) -> SynchronizedObservation:
        head = self.cache.wait_for_head_after(self._last_head_stamp, timeout_s)
        if head is None or head.stamp_ns is None:
            raise TimeoutError("timed out waiting for a new stamped ZED RGB frame")
        target_ns = int(head.stamp_ns)
        selected = self._select(target_ns, require_state=require_state)
        if require_state and "state" not in selected:
            raise TimeoutError("model requires observation state, but no synchronized JointState arrived")
        images = {
            key: image_msg_to_rgb(
                selected[key].message,
                expected_shape=(self.cameras[key].height, self.cameras[key].width, 3),
            )
            for key in ("head", "wrist_left", "wrist_right")
        }
        depth = depth_msg_to_meters(selected["depth"].message, self.cameras["head"].depth_scale)
        if depth.shape != (self.cameras["head"].height, self.cameras["head"].width):
            raise ValueError(f"ZED depth shape {depth.shape} does not match configured RGB dimensions")
        camera_info = self.cache.snapshot()["camera_info"]
        assert camera_info is not None
        points = make_point_cloud(
            depth,
            images["head"],
            camera_info.message,
            self.pointcloud,
            frame_index=self._frame_index,
        )
        state = None
        if "state" in selected:
            if self.state_gripper_closed is None or self.state_gripper_open is None:
                raise ValueError("state gripper calibration is required when a model consumes JointState")
            state = build_state(
                _joint_map(selected["state"].message),
                closed_rad=self.state_gripper_closed,
                open_rad=self.state_gripper_open,
            )
        self._last_head_stamp = target_ns
        self._frame_index += 1
        source_stamps = {key: int(value.stamp_ns) for key, value in selected.items() if value.stamp_ns is not None}
        return SynchronizedObservation(
            stamp_ns=target_ns,
            head_rgb=images["head"],
            head_depth_m=depth,
            wrist_left_rgb=images["wrist_left"],
            wrist_right_rgb=images["wrist_right"],
            point_cloud=points,
            state=state,
            source_stamps_ns=source_stamps,
        )
