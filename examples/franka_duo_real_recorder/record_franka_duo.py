#!/usr/bin/env python3
"""Record a real Franka Duo demonstration into LeRobot v3.

The recorder deliberately stores only measurements and controller targets that
are available on the real robot.  It does not invent a world frame, base
odometry, or end-effector pose.  FK enrichment is an offline operation (see
``enrich_fk.py``).

The ROS topic names mirror the benchmark recorder by default.  A real robot
controller should publish the same semantic topics, or use a ROS relay and a
YAML config to map its native names.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import queue
import re
import select
import shutil
import sys
import termios
import threading
import time
import tty
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("franka_duo_recorder")

FPS = 30
ACTION_DIM = 17
STATE_DIM = 17
ACTION_NAMES = tuple(
    [f"left_franka_joint{i}.target" for i in range(1, 8)]
    + [f"right_franka_joint{i}.target" for i in range(1, 8)]
    + ["left_gripper.open_fraction.target", "right_gripper.open_fraction.target", "spine.height.target"]
)
STATE_NAMES = tuple(
    [f"left_franka_joint{i}.pos" for i in range(1, 8)]
    + [f"right_franka_joint{i}.pos" for i in range(1, 8)]
    + ["left_gripper.open_fraction", "right_gripper.open_fraction", "spine.height"]
)

DEFAULT_TOPICS = {
    "joint_states": "/isaac/joint_states_full",
    "applied_commands": "/isaac/applied_joint_commands",
}
DEFAULT_CAMERAS = {
    "head": {
        "image": "/isaac/head_camera/image_raw",
        "depth": "/isaac/head_camera/depth",
        "camera_info": "/isaac/head_camera/camera_info",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "record_depth": True,
    },
    "wrist_left": {
        "image": "/isaac/left_wrist_camera/image_raw",
        "camera_info": "/isaac/left_wrist_camera/camera_info",
        "width": 480,
        "height": 270,
        "fps": 30,
        "record_depth": False,
    },
    "wrist_right": {
        "image": "/isaac/right_wrist_camera/image_raw",
        "camera_info": "/isaac/right_wrist_camera/camera_info",
        "width": 480,
        "height": 270,
        "fps": 30,
        "record_depth": False,
    },
}

LEFT_JOINTS = tuple(f"left_fr3v2_joint{i}" for i in range(1, 8))
RIGHT_JOINTS = tuple(f"right_fr3v2_joint{i}" for i in range(1, 8))
SPINE_JOINT = "franka_spine_vertical_joint"
GRIPPER_CLOSED_RAD = 0.8


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    key: str
    image_topic: str
    width: int
    height: int
    fps: int
    depth_topic: str | None = None
    camera_info_topic: str | None = None
    record_depth: bool = False

    @property
    def feature_key(self) -> str:
        return f"observation.images.{self.key}"


@dataclasses.dataclass(frozen=True)
class TimedMessage:
    message: Any
    arrival_ns: int
    stamp_ns: int | None


@dataclasses.dataclass
class RecorderConfig:
    fps: int = FPS
    output_root: Path = Path("datasets/franka_duo")
    dataset_name: str = "franka_duo_real"
    robot_type: str = "franka_duo_real"
    task: str = "Demonstration collected on the real Franka Duo."
    topics: dict[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_TOPICS))
    cameras: dict[str, CameraSpec] = dataclasses.field(default_factory=dict)
    depth_match_tolerance_ms: float = 16.0
    rgb_match_tolerance_ms: float = 45.0
    control_match_tolerance_ms: float = 50.0
    sync_wait_timeout_ms: float = 75.0
    max_message_age_ms: float = 150.0
    sync_history_size: int = 30
    depth_queue_size: int = 64
    depth_every: int = 1
    reject_episode_on_missing_depth: bool = True
    max_episode_time_s: float = 180.0
    max_episodes: int = 100
    streaming_encoding: bool = True
    rgb_vcodec: str | None = "auto"
    encoder_queue_maxsize: int = 90
    encoder_threads: int | None = 2
    image_writer_threads: int = 2
    image_writer_processes: int = 0
    gripper_closed_rad: float | None = None
    gripper_open_rad: float | None = None
    head_to_robot_base_transform: tuple[float, ...] | None = None


def _default_camera_specs() -> dict[str, CameraSpec]:
    return {
        key: CameraSpec(
            key=key,
            image_topic=str(value["image"]),
            depth_topic=value.get("depth"),
            camera_info_topic=value.get("camera_info"),
            width=int(value["width"]),
            height=int(value["height"]),
            fps=int(value.get("fps", FPS)),
            record_depth=bool(value.get("record_depth", False)),
        )
        for key, value in DEFAULT_CAMERAS.items()
    }


def _load_yaml_or_json(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config needs PyYAML; install it in the ROS environment") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config {path} must contain a mapping")
    return value


def load_config(path: Path | None) -> RecorderConfig:
    config = RecorderConfig(cameras=_default_camera_specs())
    if path is None:
        return config
    raw = _load_yaml_or_json(path)
    for key in (
        "fps",
        "dataset_name",
        "robot_type",
        "task",
        "depth_match_tolerance_ms",
        "rgb_match_tolerance_ms",
        "control_match_tolerance_ms",
        "sync_wait_timeout_ms",
        "max_message_age_ms",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    if "output_root" in raw:
        config.output_root = Path(str(raw["output_root"]))
    if "topics" in raw:
        config.topics.update({str(k): str(v) for k, v in dict(raw["topics"]).items()})
    camera_values = dict(raw.get("cameras", {}))
    specs: dict[str, CameraSpec] = {}
    for key, default in config.cameras.items():
        value = dict(camera_values.get(key, {}))
        specs[key] = CameraSpec(
            key=key,
            image_topic=str(value.get("image", default.image_topic)),
            depth_topic=value.get("depth", default.depth_topic),
            camera_info_topic=value.get("camera_info", default.camera_info_topic),
            width=int(value.get("width", default.width)),
            height=int(value.get("height", default.height)),
            fps=int(value.get("fps", default.fps)),
            record_depth=bool(value.get("record_depth", default.record_depth)),
        )
    config.cameras = specs
    for key in (
        "depth_queue_size",
        "depth_every",
        "sync_history_size",
        "max_episode_time_s",
        "max_episodes",
        "encoder_queue_maxsize",
        "encoder_threads",
        "image_writer_threads",
        "image_writer_processes",
        "depth_match_tolerance_ms",
        "gripper_closed_rad",
        "gripper_open_rad",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    if "streaming_encoding" in raw:
        config.streaming_encoding = bool(raw["streaming_encoding"])
    if "reject_episode_on_missing_depth" in raw:
        config.reject_episode_on_missing_depth = bool(raw["reject_episode_on_missing_depth"])
    if "rgb_vcodec" in raw:
        config.rgb_vcodec = raw["rgb_vcodec"]
    if raw.get("head_to_robot_base_transform") is not None:
        config.head_to_robot_base_transform = tuple(
            float(value) for value in raw["head_to_robot_base_transform"]
        )
    return config


def validate_config(config: RecorderConfig) -> None:
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.depth_every <= 0:
        raise ValueError("depth_every must be positive")
    for key in (
        "depth_match_tolerance_ms",
        "rgb_match_tolerance_ms",
        "control_match_tolerance_ms",
        "sync_wait_timeout_ms",
        "max_message_age_ms",
    ):
        if float(getattr(config, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.sync_history_size <= 0:
        raise ValueError("sync_history_size must be positive")
    if not config.cameras:
        raise ValueError("At least one RGB camera is required")
    for key, camera in config.cameras.items():
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError(f"Invalid dimensions/fps for camera {key}")
        if camera.fps != config.fps:
            raise ValueError(
                f"Camera {key} is configured for {camera.fps} FPS but the LeRobot dataset is "
                f"configured for {config.fps} FPS"
            )
        if camera.record_depth and not camera.depth_topic:
            raise ValueError(f"Camera {key} records depth but has no depth topic")
        if camera.record_depth and not camera.camera_info_topic:
            raise ValueError(f"Camera {key} records depth but has no CameraInfo topic")
    depth_camera_keys = [key for key, camera in config.cameras.items() if camera.record_depth]
    if depth_camera_keys != ["head"]:
        raise ValueError("Exactly the head camera must have record_depth=true")
    if config.gripper_closed_rad is None or config.gripper_open_rad is None:
        raise ValueError(
            "gripper_closed_rad and gripper_open_rad must be set from a real-hardware calibration; "
            "do not reuse the Isaac Sim value blindly"
        )
    if not math.isfinite(float(config.gripper_closed_rad)) or not math.isfinite(
        float(config.gripper_open_rad)
    ):
        raise ValueError("gripper calibration endpoints must be finite")
    if math.isclose(config.gripper_closed_rad, config.gripper_open_rad, abs_tol=1e-12):
        raise ValueError("gripper closed/open calibration endpoints must differ")
    if config.head_to_robot_base_transform is not None and len(config.head_to_robot_base_transform) != 16:
        raise ValueError("head_to_robot_base_transform must contain 16 row-major values")
    if config.head_to_robot_base_transform is not None:
        transform = np.asarray(config.head_to_robot_base_transform, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(transform).all() or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError("head_to_robot_base_transform must be a finite homogeneous 4x4 matrix")


def camera_specs_from_names(config: RecorderConfig, names: str | None) -> dict[str, CameraSpec]:
    if not names:
        return dict(config.cameras)
    requested = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in requested if name not in config.cameras]
    if unknown:
        raise ValueError(f"Unknown cameras {unknown}; available: {sorted(config.cameras)}")
    if not requested:
        raise ValueError("--cameras cannot be empty")
    return {name: config.cameras[name] for name in requested}


def _candidate_joint_names(side: str, index: int) -> tuple[str, ...]:
    side = side.lower()
    return (
        f"{side}_fr3v2_joint{index}",
        f"{side}_fr3v2_1_joint{index}",
        f"{side}_fr3_joint{index}",
        f"{side}_joint{index}",
        f"{side}_panda_joint{index}",
    )


def _resolve(mapping: Mapping[str, float], names: Sequence[str], default: float = math.nan) -> float:
    for name in names:
        value = mapping.get(name)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return default


def _arm_joints(mapping: Mapping[str, float], side: str) -> np.ndarray:
    values = np.asarray(
        [_resolve(mapping, _candidate_joint_names(side, i)) for i in range(1, 8)], dtype=np.float32
    )
    if not np.isfinite(values).all():
        missing = [i + 1 for i, value in enumerate(values) if not math.isfinite(float(value))]
        raise ValueError(f"Missing {side} arm joints {missing}")
    return values


def _gripper_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_right_finger_joint",
        f"{side}_fr3v2_finger_joint1",
        f"{side}_franka_finger_joint1",
        f"{side}_panda_finger_joint1",
        f"{side}_finger_joint1",
        f"{side}_gripper_width",
        f"{side}_gripper",
        f"{side}_hand_joint",
    )


def gripper_open_fraction(
    position_rad: float,
    closed_rad: float = GRIPPER_CLOSED_RAD,
    open_rad: float = 0.0,
) -> float:
    if not math.isfinite(float(position_rad)):
        return math.nan
    if not math.isfinite(float(closed_rad)) or not math.isfinite(float(open_rad)):
        raise ValueError("gripper calibration endpoints must be finite")
    if math.isclose(float(closed_rad), float(open_rad), abs_tol=1e-12):
        raise ValueError("gripper calibration endpoints must differ")
    fraction = (float(position_rad) - float(closed_rad)) / (float(open_rad) - float(closed_rad))
    return float(np.clip(fraction, 0.0, 1.0))


def _joint_map(message: Any) -> dict[str, float]:
    names = list(getattr(message, "name", ()))
    positions = list(getattr(message, "position", ()))
    return {str(name): float(position) for name, position in zip(names, positions, strict=False)}


def build_action(
    command: Mapping[str, float],
    measured: Mapping[str, float],
    *,
    closed_rad: float = GRIPPER_CLOSED_RAD,
    open_rad: float = 0.0,
    allow_measured_fallback: bool = False,
) -> np.ndarray:
    """Build the 17D action without base or world-frame fields."""

    source = command if command else measured if allow_measured_fallback else {}
    left = _arm_joints(source, "left")
    right = _arm_joints(source, "right")
    left_gripper = _resolve(source, _gripper_names("left"))
    right_gripper = _resolve(source, _gripper_names("right"))
    spine = _resolve(source, (SPINE_JOINT,))
    if allow_measured_fallback:
        if not math.isfinite(left_gripper):
            left_gripper = _resolve(measured, _gripper_names("left"))
        if not math.isfinite(right_gripper):
            right_gripper = _resolve(measured, _gripper_names("right"))
        if not math.isfinite(spine):
            spine = _resolve(measured, (SPINE_JOINT,))
    values = np.concatenate(
        (
            left,
            right,
            np.asarray(
                (
                    gripper_open_fraction(left_gripper, closed_rad, open_rad),
                    gripper_open_fraction(right_gripper, closed_rad, open_rad),
                    spine,
                ),
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
        raise ValueError("Action topic is missing a finite target for every required channel")
    return values


def build_state(
    measured: Mapping[str, float],
    *,
    closed_rad: float = GRIPPER_CLOSED_RAD,
    open_rad: float = 0.0,
) -> np.ndarray:
    """Build the 17D measured state; EE poses are intentionally not included."""

    left = _arm_joints(measured, "left")
    right = _arm_joints(measured, "right")
    left_gripper = _resolve(measured, _gripper_names("left"))
    right_gripper = _resolve(measured, _gripper_names("right"))
    spine = _resolve(measured, (SPINE_JOINT,))
    values = np.concatenate(
        (
            left,
            right,
            np.asarray(
                (
                    gripper_open_fraction(left_gripper, closed_rad, open_rad),
                    gripper_open_fraction(right_gripper, closed_rad, open_rad),
                    spine,
                ),
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    if values.shape != (STATE_DIM,) or not np.isfinite(values).all():
        raise ValueError("Joint state topic is missing a finite value for every required channel")
    return values


def _stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError):
        return None


def camera_info_to_dict(message: Any) -> dict[str, Any]:
    """Serialize intrinsics needed to project later ZED depth into points."""

    header = getattr(message, "header", None)
    return {
        "width": int(getattr(message, "width", 0)),
        "height": int(getattr(message, "height", 0)),
        "distortion_model": str(getattr(message, "distortion_model", "")),
        "d": [float(value) for value in getattr(message, "d", ())],
        "k": [float(value) for value in getattr(message, "k", ())],
        "r": [float(value) for value in getattr(message, "r", ())],
        "p": [float(value) for value in getattr(message, "p", ())],
        "frame_id": str(getattr(header, "frame_id", "")),
        "stamp_ns": _stamp_ns(message),
    }


def _image_rows(message: Any, dtype: np.dtype) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    itemsize = np.dtype(dtype).itemsize
    if step % itemsize:
        raise ValueError("Image step is not aligned to the pixel dtype")
    row_values = step // itemsize
    byte_order = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    message_dtype = np.dtype(dtype).newbyteorder(byte_order)
    raw = np.frombuffer(message.data, dtype=message_dtype)
    if raw.size < height * row_values:
        raise ValueError("Image message data is shorter than height*step")
    return raw[: height * row_values].reshape(height, row_values)[:, :width]


def _yuv422_to_rgb(message: Any, *, uyvy: bool) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if width % 2:
        raise ValueError("YUV422 images must have an even width")
    row_bytes = width * 2
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if step < row_bytes or raw.size < height * step:
        raise ValueError("YUV422 message step/data are inconsistent with width")
    packed = raw[: height * step].reshape(height, step)[:, :row_bytes].reshape(height, width // 2, 4)
    if uyvy:
        u, y0, v, y1 = (packed[..., index].astype(np.int32) for index in range(4))
    else:
        y0, u, y1, v = (packed[..., index].astype(np.int32) for index in range(4))
    y = np.empty((height, width), dtype=np.int32)
    y[:, 0::2] = y0
    y[:, 1::2] = y1
    u = np.repeat(u, 2, axis=1) - 128
    v = np.repeat(v, 2, axis=1) - 128
    c = np.maximum(y - 16, 0)
    red = (298 * c + 409 * v + 128) >> 8
    green = (298 * c - 100 * u - 208 * v + 128) >> 8
    blue = (298 * c + 516 * u + 128) >> 8
    return np.ascontiguousarray(np.clip(np.stack((red, green, blue), axis=-1), 0, 255), dtype=np.uint8)


def image_msg_to_rgb(message: Any, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """Decode common ROS image encodings into contiguous HWC RGB uint8."""

    encoding = str(getattr(message, "encoding", "")).lower()
    if encoding in {"yuyv", "yuy2", "yuv422_yuy2"}:
        result = _yuv422_to_rgb(message, uyvy=False)
        if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
            raise ValueError(f"RGB shape {result.shape} does not match configured {expected_shape}")
        return result
    if encoding in {"uyvy", "yuv422"}:
        result = _yuv422_to_rgb(message, uyvy=True)
        if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
            raise ValueError(f"RGB shape {result.shape} does not match configured {expected_shape}")
        return result
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        gray = _image_rows(message, np.uint8)
        result = np.repeat(gray[:, :, None], 3, axis=2)
        channels = 3
        rows = result
    else:
        raise ValueError(f"Unsupported RGB image encoding {encoding!r}")
    if encoding not in {"mono8", "8uc1"}:
        height = int(message.height)
        step = int(message.step)
        row_bytes = int(message.width) * channels
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if step < row_bytes or raw.size < height * step:
            raise ValueError("RGB message step/data are inconsistent with width and encoding")
        rows = raw[: height * step].reshape(height, step)[:, :row_bytes]
    if channels == 3:
        result = rows.reshape(int(message.height), int(message.width), 3)
        if encoding == "bgr8":
            result = result[:, :, ::-1]
    elif channels == 4:
        result = rows.reshape(int(message.height), int(message.width), 4)[:, :, :3]
        if encoding == "bgra8":
            result = result[:, :, ::-1]
    if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
        raise ValueError(f"RGB shape {result.shape} does not match configured {expected_shape}")
    return np.ascontiguousarray(result, dtype=np.uint8)


def depth_msg_to_meters(message: Any, depth_scale: float = 0.001) -> np.ndarray:
    """Decode ZED/ROS depth as float16 meters.

    ``32FC1`` is already meters.  ``16UC1`` is conventionally millimetres for
    ROS camera drivers and uses ``depth_scale`` as the conversion factor.
    """

    encoding = str(getattr(message, "encoding", "")).lower()
    if encoding in {"32fc1", "32fc"}:
        values = _image_rows(message, np.float32).astype(np.float32, copy=False)
    elif encoding in {"16uc1", "mono16"}:
        values = _image_rows(message, np.uint16).astype(np.float32) * float(depth_scale)
    else:
        raise ValueError(f"Unsupported depth encoding {encoding!r}")
    return np.ascontiguousarray(values.astype(np.float16))


class DepthSidecarWriter:
    """Write matched head RGB/depth pairs without blocking the ROS sampler."""

    def __init__(
        self,
        root: Path,
        *,
        queue_size: int = 64,
        head_to_robot_base_transform: Sequence[float] | None = None,
    ):
        self.root = Path(root)
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[tuple[int, int, int, np.ndarray] | None] | None = None
        self._thread: threading.Thread | None = None
        self._files: tuple[Any, Any, Any] | None = None
        self._episode_dir: Path | None = None
        self._shape: tuple[int, int] | None = None
        self._written = 0
        self._dropped = 0
        self._missing: list[int] = []
        self._camera_info: dict[str, Any] = {}
        self._sample_timestamp_layout: list[str] | None = None
        self._sample_timestamps: list[tuple[int, ...]] = []
        self._error: BaseException | None = None
        self._head_to_robot_base_transform = (
            [float(value) for value in head_to_robot_base_transform]
            if head_to_robot_base_transform is not None
            else None
        )

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def written(self) -> int:
        return self._written

    @property
    def missing(self) -> int:
        return len(self._missing)

    def start_episode(self, episode_index: int) -> None:
        self.finish_episode(save=False)
        self._episode_dir = self.root / f"episode_{episode_index:06d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._files = (
            (self._episode_dir / "depth_head.f16").open("wb"),
            (self._episode_dir / "depth_head_frame_index.i8").open("wb"),
            (self._episode_dir / "depth_head_stamp_ns.i8").open("wb"),
        )
        self._written = 0
        self._dropped = 0
        self._missing = []
        self._camera_info = {}
        self._sample_timestamp_layout = None
        self._sample_timestamps = []
        self._error = None
        self._shape = None
        self._thread = threading.Thread(target=self._worker, name="depth-writer", daemon=True)
        self._thread.start()

    def mark_missing(self, frame_index: int) -> None:
        self._missing.append(int(frame_index))

    def set_camera_info(self, camera_info: Mapping[str, Any]) -> None:
        incoming = dict(camera_info)
        if self._camera_info:
            previous = {
                key: {field: value for field, value in info.items() if field != "stamp_ns"}
                for key, info in self._camera_info.items()
            }
            current = {
                key: {field: value for field, value in info.items() if field != "stamp_ns"}
                for key, info in incoming.items()
            }
            if current != previous:
                raise ValueError("Camera calibration changed within an episode")
        self._camera_info = incoming

    def record_sample_timestamps(
        self,
        frame_index: int,
        *,
        state_stamp_ns: int,
        action_stamp_ns: int,
        rgb_stamps_ns: Mapping[str, int],
    ) -> None:
        camera_keys = sorted(rgb_stamps_ns)
        layout = ["frame_index", "state_stamp_ns", "action_stamp_ns"] + [
            f"rgb_{key}_stamp_ns" for key in camera_keys
        ]
        if self._sample_timestamp_layout is None:
            self._sample_timestamp_layout = layout
        elif layout != self._sample_timestamp_layout:
            raise ValueError("RGB timestamp camera layout changed within an episode")
        self._sample_timestamps.append(
            (
                int(frame_index),
                int(state_stamp_ns),
                int(action_stamp_ns),
                *(int(rgb_stamps_ns[key]) for key in camera_keys),
            )
        )

    def enqueue(self, frame_index: int, rgb_stamp_ns: int, depth_stamp_ns: int, depth: np.ndarray) -> bool:
        if self._queue is None:
            raise RuntimeError("start_episode must be called before enqueue")
        depth = np.ascontiguousarray(depth, dtype=np.float16)
        if self._shape is None:
            self._shape = tuple(depth.shape)
        elif tuple(depth.shape) != self._shape:
            raise ValueError(f"Depth shape {depth.shape} changed from {self._shape}")
        item = (int(frame_index), int(rgb_stamp_ns), int(depth_stamp_ns), depth)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1
            return False
        return True

    def _worker(self) -> None:
        assert self._queue is not None
        assert self._files is not None
        data_file, frame_file, stamp_file = self._files
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is None:
                    try:
                        frame_index, rgb_stamp_ns, depth_stamp_ns, depth = item
                        data_file.write(depth.tobytes(order="C"))
                        np.asarray((frame_index,), dtype="<i8").tofile(frame_file)
                        np.asarray((rgb_stamp_ns, depth_stamp_ns), dtype="<i8").tofile(stamp_file)
                        self._written += 1
                    except BaseException as exc:  # pragma: no cover - requires an I/O failure
                        self._error = exc
            finally:
                self._queue.task_done()

    def finish_episode(self, *, save: bool = True) -> dict[str, Any] | None:
        if self._queue is None:
            return None
        work_queue = self._queue
        thread = self._thread
        files = self._files
        episode_dir = self._episode_dir
        error: BaseException | None = None
        result = None
        try:
            work_queue.join()
            work_queue.put(None)
            if thread is not None:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    raise RuntimeError("Depth sidecar writer did not stop within 5 seconds")
            error = self._error
            if files is not None:
                for file in files:
                    try:
                        file.flush()
                    except BaseException as exc:  # pragma: no cover - requires an I/O failure
                        error = error or exc
                    try:
                        file.close()
                    except BaseException as exc:  # pragma: no cover - requires an I/O failure
                        error = error or exc
            if save and error is None and episode_dir is not None:
                timestamp_path = episode_dir / "sample_timestamps_ns.i8"
                timestamp_array = np.asarray(self._sample_timestamps, dtype="<i8")
                timestamp_array.tofile(timestamp_path)
                metadata = {
                    "format": "franka_duo_depth_sidecar_v1",
                    "camera": "head",
                    "dtype": "float16",
                    "unit": "meter",
                    "shape": list(self._shape or (0, 0)),
                    "depth_file": "depth_head.f16",
                    "frame_index_file": "depth_head_frame_index.i8",
                    "stamp_file": "depth_head_stamp_ns.i8",
                    "stamp_layout": ["rgb_stamp_ns", "depth_stamp_ns"],
                    "frames_written": self._written,
                    "frames_dropped": self._dropped,
                    "missing_rgb_frame_indices": self._missing,
                    "sample_timestamp_file": timestamp_path.name,
                    "sample_timestamp_layout": self._sample_timestamp_layout or [],
                    "samples_recorded": len(self._sample_timestamps),
                    "camera_info": self._camera_info,
                    "head_to_robot_base_transform": self._head_to_robot_base_transform,
                }
                (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                result = metadata
        except BaseException as exc:
            error = error or exc
        finally:
            if (not save or error is not None) and episode_dir is not None:
                shutil.rmtree(episode_dir, ignore_errors=True)
            self._queue = None
            self._thread = None
            self._files = None
            self._episode_dir = None
            self._error = None
        if error is not None:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error
            raise RuntimeError("Depth sidecar writer failed") from error
        return result


class TopicCache:
    """Thread-safe ROS message histories used for timestamp synchronization."""

    def __init__(self, cameras: Mapping[str, CameraSpec], history_size: int = 30):
        history_size = max(1, int(history_size))
        self._condition = threading.Condition()
        self._revision = 0
        self.joint_states: deque[TimedMessage] = deque(maxlen=history_size)
        self.applied_commands: deque[TimedMessage] = deque(maxlen=history_size)
        self.images: dict[str, deque[TimedMessage]] = {key: deque(maxlen=history_size) for key in cameras}
        self.camera_info: dict[str, TimedMessage] = {}
        self.depths: dict[str, deque[TimedMessage]] = {
            key: deque(maxlen=history_size) for key, camera in cameras.items() if camera.record_depth
        }
        self.cameras = cameras

    def _notify(self) -> None:
        self._revision += 1
        self._condition.notify_all()

    def store_joint_states(self, message: Any) -> None:
        with self._condition:
            self.joint_states.append(TimedMessage(message, time.monotonic_ns(), _stamp_ns(message)))
            self._notify()

    def store_applied_commands(self, message: Any) -> None:
        with self._condition:
            self.applied_commands.append(TimedMessage(message, time.monotonic_ns(), _stamp_ns(message)))
            self._notify()

    def store_image(self, key: str, message: Any) -> None:
        with self._condition:
            self.images[key].append(TimedMessage(message, time.monotonic_ns(), _stamp_ns(message)))
            self._notify()

    def store_depth(self, key: str, message: Any) -> None:
        with self._condition:
            self.depths[key].append(TimedMessage(message, time.monotonic_ns(), _stamp_ns(message)))
            self._notify()

    def store_camera_info(self, key: str, message: Any) -> None:
        with self._condition:
            self.camera_info[key] = TimedMessage(message, time.monotonic_ns(), _stamp_ns(message))
            self._notify()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "joint_states": tuple(self.joint_states),
            "applied_commands": tuple(self.applied_commands),
            "images": {key: tuple(items) for key, items in self.images.items()},
            "camera_info": dict(self.camera_info),
            "depths": {key: tuple(items) for key, items in self.depths.items()},
            "revision": self._revision,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()

    def wait_for_image_after(self, key: str, after_stamp_ns: int, timeout_s: float) -> TimedMessage | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while True:
                candidates = [
                    item
                    for item in self.images[key]
                    if item.stamp_ns is not None and item.stamp_ns > after_stamp_ns
                ]
                if candidates:
                    return min(candidates, key=lambda item: int(item.stamp_ns))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def wait_for_update(self, after_revision: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._revision <= after_revision:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


def _nearest_message(
    messages: Sequence[TimedMessage], target_stamp_ns: int | None, tolerance_ns: int
) -> TimedMessage | None:
    if target_stamp_ns is None:
        return None
    candidates = [item for item in messages if item.stamp_ns is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(int(item.stamp_ns) - int(target_stamp_ns)))
    if abs(int(best.stamp_ns) - int(target_stamp_ns)) > tolerance_ns:
        return None
    return best


def _nearest_depth(
    depths: Sequence[TimedMessage], rgb_stamp_ns: int | None, tolerance_ns: int
) -> TimedMessage | None:
    return _nearest_message(depths, rgb_stamp_ns, tolerance_ns)


def _latest_stamp(messages: Sequence[TimedMessage]) -> int | None:
    stamps = [int(item.stamp_ns) for item in messages if item.stamp_ns is not None]
    return max(stamps, default=None)


def _snapshot_covers_stamp(
    snapshot: Mapping[str, Any],
    cameras: Mapping[str, CameraSpec],
    target_stamp_ns: int,
    *,
    require_depth: bool,
) -> bool:
    histories: list[Sequence[TimedMessage]] = [
        snapshot["joint_states"],
        snapshot["applied_commands"],
        *(snapshot["images"].get(key, ()) for key in cameras),
    ]
    if require_depth:
        histories.extend(
            snapshot["depths"].get(key, ()) for key, camera in cameras.items() if camera.record_depth
        )
    return all(
        (latest := _latest_stamp(history)) is not None and latest >= target_stamp_ns for history in histories
    )


def validate_camera_info(message: Any, camera: CameraSpec) -> None:
    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    distortion = np.asarray(getattr(message, "d", ()), dtype=np.float64)
    intrinsics = np.asarray(getattr(message, "k", ()), dtype=np.float64)
    rectification = np.asarray(getattr(message, "r", ()), dtype=np.float64)
    projection = np.asarray(getattr(message, "p", ()), dtype=np.float64)
    frame_id = str(getattr(getattr(message, "header", None), "frame_id", ""))
    if (width, height) != (camera.width, camera.height):
        raise ValueError(
            f"CameraInfo for {camera.key} is {width}x{height}, expected {camera.width}x{camera.height}"
        )
    if intrinsics.shape != (9,) or rectification.shape != (9,) or projection.shape != (12,):
        raise ValueError(f"CameraInfo for {camera.key} must contain K[9], R[9], and P[12]")
    if not all(np.isfinite(values).all() for values in (distortion, intrinsics, rectification, projection)):
        raise ValueError(f"CameraInfo for {camera.key} contains non-finite calibration values")
    if intrinsics[0] <= 0 or intrinsics[4] <= 0 or not frame_id:
        raise ValueError(f"CameraInfo for {camera.key} has invalid focal lengths or frame_id")


def _select_synchronized_messages(
    snapshot: Mapping[str, Any],
    cameras: Mapping[str, CameraSpec],
    target_stamp_ns: int,
    *,
    rgb_tolerance_ns: int,
    control_tolerance_ns: int,
    max_age_ns: int,
    rgb_after_stamps: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    now = time.monotonic_ns()
    measured = _nearest_message(snapshot["joint_states"], target_stamp_ns, control_tolerance_ns)
    command = _nearest_message(snapshot["applied_commands"], target_stamp_ns, control_tolerance_ns)
    rgb_after_stamps = rgb_after_stamps or {}
    images = {}
    for key in cameras:
        history = [
            item
            for item in snapshot["images"].get(key, ())
            if item.stamp_ns is not None and item.stamp_ns > rgb_after_stamps.get(key, -1)
        ]
        images[key] = _nearest_message(history, target_stamp_ns, rgb_tolerance_ns)
    if measured is None or command is None or any(item is None for item in images.values()):
        raise ValueError("No state/action/RGB tuple satisfies the configured timestamp tolerances")
    selected = [measured, command, *images.values()]
    if any(now - item.arrival_ns > max_age_ns for item in selected):
        raise ValueError("Synchronized ROS messages became stale before they could be recorded")
    camera_info = snapshot["camera_info"]
    missing_info = [key for key in cameras if key not in camera_info]
    if missing_info:
        raise ValueError(f"CameraInfo has not arrived for cameras {missing_info}")
    for key, camera in cameras.items():
        validate_camera_info(camera_info[key].message, camera)
    return {
        "joint_states": measured,
        "applied_commands": command,
        "images": images,
        "camera_info": camera_info,
    }


def next_dataset_path(output_root: Path, dataset_name: str) -> tuple[Path, str, int]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        name = f"{dataset_name}_v{version}"
        path = output_root / name
        if not path.exists():
            return path, name, version
        version += 1


def existing_dataset_versions(output_root: Path, dataset_name: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"{re.escape(dataset_name)}_v([0-9]+)")
    versions: list[tuple[int, Path]] = []
    for path in Path(output_root).glob(f"{dataset_name}_v*"):
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_dir():
            versions.append((int(match.group(1)), path))
    return sorted(versions)


def build_features(cameras: Mapping[str, CameraSpec]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION_NAMES)},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": list(STATE_NAMES)},
    }
    for camera in cameras.values():
        features[camera.feature_key] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _dataset_kwargs(config: RecorderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "image_writer_processes": config.image_writer_processes,
        "image_writer_threads": config.image_writer_threads * max(1, len(config.cameras)),
        "streaming_encoding": config.streaming_encoding,
        "encoder_queue_maxsize": config.encoder_queue_maxsize,
        "encoder_threads": config.encoder_threads,
    }
    if config.rgb_vcodec:
        from lerobot.configs.video import VideoEncoderConfig

        kwargs["camera_encoder"] = VideoEncoderConfig(vcodec=config.rgb_vcodec)
    return kwargs


def _codec_family(codec: str) -> str:
    codec = codec.lower()
    if codec == "av1" or codec == "libsvtav1":
        return "av1"
    if codec.startswith("h264"):
        return "h264"
    if codec.startswith("hevc"):
        return "hevc"
    return codec


def _validate_resume_contract(
    dataset: Any, config: RecorderConfig, cameras: Mapping[str, CameraSpec]
) -> None:
    if int(dataset.meta.fps) != int(config.fps):
        raise ValueError(f"Cannot resume: dataset FPS {dataset.meta.fps} != configured {config.fps}")
    if dataset.meta.robot_type != config.robot_type:
        raise ValueError(
            f"Cannot resume: robot_type {dataset.meta.robot_type!r} != configured {config.robot_type!r}"
        )

    expected = build_features(cameras)
    for key, expected_feature in expected.items():
        actual = dataset.meta.features.get(key)
        if actual is None:
            raise ValueError(f"Cannot resume: dataset is missing feature {key!r}")
        if str(actual.get("dtype")) != str(expected_feature["dtype"]):
            raise ValueError(
                f"Cannot resume: {key} dtype {actual.get('dtype')!r} != {expected_feature['dtype']!r}"
            )
        if tuple(actual.get("shape", ())) != tuple(expected_feature["shape"]):
            raise ValueError(
                f"Cannot resume: {key} shape {actual.get('shape')} != {expected_feature['shape']}"
            )
        if list(actual.get("names") or []) != list(expected_feature.get("names") or []):
            raise ValueError(f"Cannot resume: {key} channel names do not match the recorder contract")

    expected_video_keys = {camera.feature_key for camera in cameras.values()}
    if set(dataset.meta.video_keys) != expected_video_keys:
        raise ValueError(
            "Cannot resume: video features differ; expected "
            f"{sorted(expected_video_keys)}, found {sorted(dataset.meta.video_keys)}"
        )

    encoder = getattr(getattr(dataset, "writer", None), "_camera_encoder", None)
    if encoder is None:
        return
    desired_family = _codec_family(str(encoder.vcodec))
    for key in expected_video_keys:
        info = dataset.meta.features[key].get("info") or {}
        existing_codec = info.get("video.codec")
        existing_pix_fmt = info.get("video.pix_fmt")
        if existing_codec is not None and _codec_family(str(existing_codec)) != desired_family:
            raise ValueError(
                f"Cannot resume: {key} codec {existing_codec!r} is incompatible with {encoder.vcodec!r}"
            )
        if existing_pix_fmt is not None and str(existing_pix_fmt) != str(encoder.pix_fmt):
            raise ValueError(f"Cannot resume: {key} pixel format {existing_pix_fmt!r} != {encoder.pix_fmt!r}")


def build_recording_manifest(config: RecorderConfig, cameras: Mapping[str, CameraSpec]) -> dict[str, Any]:
    return {
        "format": "franka_duo_real_recording_v1",
        "lerobot_format": "v3.0",
        "lerobot_timestamp_semantics": "frame_index/fps; ROS source stamps are in each episode sidecar",
        "fps": config.fps,
        "robot_type": config.robot_type,
        "action": {"shape": [ACTION_DIM], "names": list(ACTION_NAMES)},
        "observation.state": {"shape": [STATE_DIM], "names": list(STATE_NAMES)},
        "topics": dict(sorted(config.topics.items())),
        "cameras": {key: dataclasses.asdict(camera) for key, camera in sorted(cameras.items())},
        "sync": {
            "depth_match_tolerance_ms": config.depth_match_tolerance_ms,
            "rgb_match_tolerance_ms": config.rgb_match_tolerance_ms,
            "control_match_tolerance_ms": config.control_match_tolerance_ms,
            "sync_wait_timeout_ms": config.sync_wait_timeout_ms,
            "max_message_age_ms": config.max_message_age_ms,
            "depth_every": config.depth_every,
            "reject_episode_on_missing_depth": config.reject_episode_on_missing_depth,
        },
        "gripper_calibration": {
            "closed_position": config.gripper_closed_rad,
            "open_position": config.gripper_open_rad,
            "unit": "source JointState.position unit",
        },
        "head_to_robot_base_transform": (
            list(config.head_to_robot_base_transform)
            if config.head_to_robot_base_transform is not None
            else None
        ),
        "streaming_encoding": config.streaming_encoding,
        "requested_rgb_vcodec": config.rgb_vcodec,
    }


def _recording_manifest_path(dataset_path: Path) -> Path:
    return dataset_path / "franka_duo_extras" / "recording_manifest.json"


def _write_or_validate_recording_manifest(
    dataset_path: Path,
    config: RecorderConfig,
    cameras: Mapping[str, CameraSpec],
    *,
    resume: bool,
) -> None:
    manifest_path = _recording_manifest_path(dataset_path)
    expected = build_recording_manifest(config, cameras)
    if resume:
        if not manifest_path.is_file():
            raise ValueError(f"Cannot resume: recording manifest is missing at {manifest_path}")
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "Cannot resume: ROS topics, synchronization, camera, gripper, or encoding "
                "settings differ from recording_manifest.json"
            )
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")


def _create_dataset(config: RecorderConfig, cameras: Mapping[str, CameraSpec], *, resume: bool):
    from lerobot.datasets import LeRobotDataset

    dataset_path, repo_name, version = next_dataset_path(config.output_root, config.dataset_name)
    if resume:
        versions = existing_dataset_versions(config.output_root, config.dataset_name)
        if not versions:
            raise FileNotFoundError(f"No existing dataset under {config.output_root} to resume")
        version, dataset_path = versions[-1]
        repo_name = dataset_path.name
        dataset = LeRobotDataset.resume(
            repo_id=f"local/{repo_name}",
            root=dataset_path,
            **_dataset_kwargs(config),
        )
        try:
            _validate_resume_contract(dataset, config, cameras)
            _write_or_validate_recording_manifest(
                dataset_path,
                config,
                cameras,
                resume=True,
            )
        except BaseException:
            dataset.finalize()
            raise
        return dataset, dataset_path, version
    dataset = LeRobotDataset.create(
        repo_id=f"local/{repo_name}",
        root=dataset_path,
        fps=config.fps,
        features=build_features(cameras),
        robot_type=config.robot_type,
        use_videos=True,
        **_dataset_kwargs(config),
    )
    try:
        _write_or_validate_recording_manifest(dataset_path, config, cameras, resume=False)
    except BaseException:
        dataset.finalize()
        raise
    return dataset, dataset_path, version


def _ros_node(config: RecorderConfig, cameras: Mapping[str, CameraSpec]):
    """Construct the ROS node lazily so pure helpers remain testable offline."""

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
    except ImportError as exc:  # pragma: no cover - depends on the robot host
        raise RuntimeError(
            "ROS 2 Python packages are unavailable. Source /opt/ros/jazzy/setup.bash "
            "or run this script in the robot ROS environment."
        ) from exc

    cache = TopicCache(cameras, history_size=config.sync_history_size)

    class RealRecorderNode(Node):
        def __init__(self):
            super().__init__("franka_duo_real_recorder")
            self.create_subscription(
                JointState, config.topics["joint_states"], cache.store_joint_states, qos_profile_sensor_data
            )
            self.create_subscription(
                JointState,
                config.topics["applied_commands"],
                cache.store_applied_commands,
                qos_profile_sensor_data,
            )
            for key, camera in cameras.items():
                self.create_subscription(
                    Image,
                    camera.image_topic,
                    lambda message, key=key: cache.store_image(key, message),
                    qos_profile_sensor_data,
                )
                if camera.record_depth:
                    self.create_subscription(
                        Image,
                        camera.depth_topic,
                        lambda message, key=key: cache.store_depth(key, message),
                        qos_profile_sensor_data,
                    )
                if camera.camera_info_topic:
                    self.create_subscription(
                        CameraInfo,
                        camera.camera_info_topic,
                        lambda message, key=key: cache.store_camera_info(key, message),
                        qos_profile_sensor_data,
                    )

    return rclpy, RealRecorderNode(), cache


class KeyboardCommands:
    def __init__(self):
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self.quit_requested = False
        self._thread = threading.Thread(target=self._run, name="recorder-keyboard", daemon=True)
        self._stdin_fd: int | None = None
        self._terminal_settings: list[Any] | None = None

    @property
    def active(self) -> bool:
        return self._stdin_fd is not None

    def start(self) -> None:
        if sys.stdin.isatty():
            self._stdin_fd = sys.stdin.fileno()
            self._terminal_settings = termios.tcgetattr(self._stdin_fd)
            try:
                tty.setcbreak(self._stdin_fd)
                self._thread.start()
            except BaseException:
                self._restore_terminal()
                raise

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                value = sys.stdin.read(1).lower()
                if value:
                    self._commands.put(value)

    def get(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._stdin_fd is not None and self._terminal_settings is not None:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._terminal_settings)
        self._stdin_fd = None
        self._terminal_settings = None


def _ready_snapshot(snapshot: Mapping[str, Any], cameras: Mapping[str, CameraSpec], max_age_ns: int) -> bool:
    now = time.monotonic_ns()
    histories = [
        snapshot["joint_states"],
        snapshot["applied_commands"],
        *(snapshot["images"].get(key, ()) for key in cameras),
    ]
    if any(not history or history[-1].stamp_ns is None for history in histories):
        return False
    if any(now - history[-1].arrival_ns > max_age_ns for history in histories):
        return False
    return all(key in snapshot["camera_info"] for key in cameras)


def _encoder_dropped_frames(dataset: Any) -> dict[str, int]:
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    dropped = getattr(encoder, "_dropped_frames", {})
    return {str(key): int(value) for key, value in dict(dropped).items() if int(value) > 0}


def _episode_integrity_errors(
    dataset: Any,
    depth_writer: DepthSidecarWriter,
    frames: int,
    *,
    reject_missing_depth: bool = True,
) -> list[str]:
    errors = []
    if frames == 0:
        errors.append("episode contains no synchronized frames")
    encoder_drops = _encoder_dropped_frames(dataset)
    if encoder_drops:
        errors.append(f"streaming video encoder dropped frames: {encoder_drops}")
    if depth_writer.dropped:
        errors.append(f"depth writer queue dropped {depth_writer.dropped} frame(s)")
    if reject_missing_depth and depth_writer.missing:
        errors.append(f"depth synchronization missed {depth_writer.missing} requested frame(s)")
    return errors


def record_episode(
    dataset: Any,
    node_cache: TopicCache,
    cameras: Mapping[str, CameraSpec],
    config: RecorderConfig,
    depth_writer: DepthSidecarWriter,
    episode_index: int,
    *,
    command: KeyboardCommands | None = None,
) -> tuple[bool, int, int]:
    """Record one episode; return ``(saved, frames, dropped_samples)``."""

    depth_writer.start_episode(episode_index)
    started = time.monotonic()
    frame_index = 0
    dropped = 0
    depth_tolerance_ns = int(config.depth_match_tolerance_ms * 1_000_000)
    rgb_tolerance_ns = int(config.rgb_match_tolerance_ms * 1_000_000)
    control_tolerance_ns = int(config.control_match_tolerance_ms * 1_000_000)
    sync_wait_timeout_s = config.sync_wait_timeout_ms / 1000.0
    max_age_ns = int(config.max_message_age_ms * 1_000_000)
    reference_key = "head" if "head" in cameras else next(iter(cameras))
    initial_snapshot = node_cache.snapshot()
    last_rgb_stamps = {key: _latest_stamp(initial_snapshot["images"].get(key, ())) or -1 for key in cameras}
    last_reference_stamp = last_rgb_stamps[reference_key]
    last_depth_stamps = {key: -1 for key, camera in cameras.items() if camera.record_depth}

    while time.monotonic() - started < config.max_episode_time_s:
        if command is not None:
            key = command.get()
            if key in {"s", "d", "q"}:
                saved = key == "s"
                if key == "q":
                    command.quit_requested = True
                return saved, frame_index, dropped

        remaining_episode_s = config.max_episode_time_s - (time.monotonic() - started)
        reference = node_cache.wait_for_image_after(
            reference_key,
            last_reference_stamp,
            timeout_s=min(0.1, max(0.0, remaining_episode_s)),
        )
        if reference is None:
            continue
        assert reference.stamp_ns is not None
        target_stamp_ns = int(reference.stamp_ns)
        last_reference_stamp = target_stamp_ns

        require_depth = frame_index % config.depth_every == 0 and any(
            camera.record_depth for camera in cameras.values()
        )
        sync_deadline = time.monotonic() + sync_wait_timeout_s
        selected: dict[str, Any] | None = None
        sync_error: ValueError | None = None
        snapshot: Mapping[str, Any] = {}
        while time.monotonic() < sync_deadline:
            snapshot = node_cache.snapshot()
            if _snapshot_covers_stamp(
                snapshot,
                cameras,
                target_stamp_ns,
                require_depth=require_depth,
            ):
                try:
                    selected = _select_synchronized_messages(
                        snapshot,
                        cameras,
                        target_stamp_ns,
                        rgb_tolerance_ns=rgb_tolerance_ns,
                        control_tolerance_ns=control_tolerance_ns,
                        max_age_ns=max_age_ns,
                        rgb_after_stamps=last_rgb_stamps,
                    )
                    break
                except ValueError as exc:
                    sync_error = exc
            remaining_sync_s = sync_deadline - time.monotonic()
            if remaining_sync_s <= 0:
                break
            node_cache.wait_for_update(int(snapshot.get("revision", -1)), remaining_sync_s)
        if selected is None:
            dropped += 1
            reason = (
                str(sync_error) if sync_error is not None else "timed out waiting for synchronized topics"
            )
            LOGGER.warning("Dropped head frame at stamp %d: %s", target_stamp_ns, reason)
            continue

        try:
            depth_writer.set_camera_info(
                {key: camera_info_to_dict(selected["camera_info"][key].message) for key in cameras}
            )
            measured = _joint_map(selected["joint_states"].message)
            command_map = _joint_map(selected["applied_commands"].message)
            frame = {
                "action": build_action(
                    command_map,
                    measured,
                    closed_rad=config.gripper_closed_rad,
                    open_rad=config.gripper_open_rad,
                ),
                "observation.state": build_state(
                    measured,
                    closed_rad=config.gripper_closed_rad,
                    open_rad=config.gripper_open_rad,
                ),
                "task": config.task,
            }
            rgb_stamps: dict[str, int] = {}
            for key, camera in cameras.items():
                image_item = selected["images"][key]
                assert image_item.stamp_ns is not None
                rgb_stamps[key] = int(image_item.stamp_ns)
                frame[camera.feature_key] = image_msg_to_rgb(
                    image_item.message, expected_shape=(camera.height, camera.width, 3)
                )

            depth_sample: tuple[str, TimedMessage, np.ndarray] | None = None
            if require_depth:
                depth_keys = [key for key, camera in cameras.items() if camera.record_depth]
                if len(depth_keys) != 1:
                    raise ValueError("Exactly one depth camera is supported by this recorder")
                depth_key = depth_keys[0]
                available_depths = [
                    item
                    for item in snapshot["depths"].get(depth_key, ())
                    if item.stamp_ns is not None and item.stamp_ns > last_depth_stamps[depth_key]
                ]
                depth_item = _nearest_depth(
                    available_depths,
                    rgb_stamps[depth_key],
                    depth_tolerance_ns,
                )
                if depth_item is not None and depth_item.stamp_ns is not None:
                    depth = depth_msg_to_meters(depth_item.message)
                    camera = cameras[depth_key]
                    if depth.shape != (camera.height, camera.width):
                        raise ValueError(
                            f"Depth shape {depth.shape} does not match {depth_key} RGB/CameraInfo "
                            f"shape {(camera.height, camera.width)}"
                        )
                    depth_sample = (depth_key, depth_item, depth)

            dataset.add_frame(frame)
            if require_depth:
                if depth_sample is None:
                    depth_writer.mark_missing(frame_index)
                else:
                    depth_key, depth_item, depth = depth_sample
                    assert depth_item.stamp_ns is not None
                    if depth_writer.enqueue(
                        frame_index,
                        rgb_stamps[depth_key],
                        int(depth_item.stamp_ns),
                        depth,
                    ):
                        last_depth_stamps[depth_key] = int(depth_item.stamp_ns)
                    else:
                        LOGGER.warning("Depth queue full at frame %d", frame_index)
            assert selected["joint_states"].stamp_ns is not None
            assert selected["applied_commands"].stamp_ns is not None
            depth_writer.record_sample_timestamps(
                frame_index,
                state_stamp_ns=int(selected["joint_states"].stamp_ns),
                action_stamp_ns=int(selected["applied_commands"].stamp_ns),
                rgb_stamps_ns=rgb_stamps,
            )
            last_rgb_stamps.update(rgb_stamps)
            frame_index += 1
        except (ValueError, TypeError) as exc:
            dropped += 1
            LOGGER.warning("Dropped sample at frame %d: %s", frame_index, exc)
    return True, frame_index, dropped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON topic and recorder config")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-time-s", type=float, default=None)
    parser.add_argument(
        "--cameras", default=None, help="Comma-separated subset, e.g. head,wrist_left,wrist_right"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--auto-start", action="store_true", help="Start recording immediately and continue until --episodes"
    )
    parser.add_argument(
        "--no-streaming-encoding", dest="streaming_encoding", action="store_false", default=None
    )
    parser.add_argument("--rgb-vcodec", default=None, help="Video codec: auto, h264_nvenc, libsvtav1, ...")
    parser.add_argument(
        "--depth-every", type=int, default=None, help="Save one head depth for every N RGB frames"
    )
    parser.add_argument("--depth-match-tolerance-ms", type=float, default=None)
    parser.add_argument("--rgb-match-tolerance-ms", type=float, default=None)
    parser.add_argument("--control-match-tolerance-ms", type=float, default=None)
    parser.add_argument(
        "--allow-missing-depth",
        dest="reject_episode_on_missing_depth",
        action="store_false",
        default=None,
        help="Allow saving episodes with explicitly marked missing head depth frames",
    )
    parser.add_argument(
        "--gripper-closed-rad",
        type=float,
        default=None,
        help="Required real-hardware closed-position calibration used to normalize both grippers",
    )
    parser.add_argument(
        "--gripper-open-rad",
        type=float,
        default=None,
        help="Required real-hardware open-position calibration in the JointState topic unit",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    for key, value in (
        ("output_root", args.output_root),
        ("dataset_name", args.dataset_name),
        ("task", args.task),
        ("fps", args.fps),
        ("max_episodes", args.episodes),
        ("max_episode_time_s", args.episode_time_s),
        ("depth_every", args.depth_every),
        ("depth_match_tolerance_ms", args.depth_match_tolerance_ms),
        ("rgb_match_tolerance_ms", args.rgb_match_tolerance_ms),
        ("control_match_tolerance_ms", args.control_match_tolerance_ms),
        ("reject_episode_on_missing_depth", args.reject_episode_on_missing_depth),
        ("gripper_closed_rad", args.gripper_closed_rad),
        ("gripper_open_rad", args.gripper_open_rad),
        ("rgb_vcodec", args.rgb_vcodec),
        ("streaming_encoding", args.streaming_encoding),
    ):
        if value is not None:
            setattr(config, key, value)
    validate_config(config)
    cameras = camera_specs_from_names(config, args.cameras)
    config.cameras = cameras

    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - depends on the robot host
        raise RuntimeError("ROS 2 Python packages are unavailable") from exc
    rclpy.init()
    _, node, cache = _ros_node(config, cameras)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), name="ros-spin", daemon=True)
    spin_thread.start()
    dataset: Any | None = None
    depth_writer: DepthSidecarWriter | None = None
    keyboard: KeyboardCommands | None = None
    try:
        dataset, dataset_path, version = _create_dataset(config, cameras, resume=args.resume)
        extras_root = dataset_path / "franka_duo_extras"
        depth_writer = DepthSidecarWriter(
            extras_root,
            queue_size=config.depth_queue_size,
            head_to_robot_base_transform=config.head_to_robot_base_transform,
        )
        keyboard = KeyboardCommands()
        keyboard.start()
        if not args.auto_start and not keyboard.active:
            raise RuntimeError("Interactive mode requires a TTY; use --auto-start in a non-interactive shell")
        depth_keys = [key for key, camera in cameras.items() if camera.record_depth]
        LOGGER.info(
            "Dataset: %s (v%d), action/state=%d/%d, cameras=%s, depth=%s every %d RGB frames",
            dataset_path,
            version,
            ACTION_DIM,
            STATE_DIM,
            ",".join(cameras),
            ",".join(depth_keys) or "none",
            config.depth_every,
        )
        LOGGER.info("Idle controls: r=start, q=quit. Recording: s=save, d=discard, q=discard+quit.")
        saved_episodes = int(getattr(dataset.meta, "total_episodes", 0))
        auto = args.auto_start
        quit_requested = False
        while saved_episodes < config.max_episodes and not quit_requested:
            if not auto:
                while True:
                    key = keyboard.get()
                    if key == "q":
                        quit_requested = True
                        break
                    if key == "r":
                        break
                    time.sleep(0.05)
                if quit_requested:
                    break
            saved, frames, dropped = record_episode(
                dataset,
                cache,
                cameras,
                config,
                depth_writer,
                saved_episodes,
                command=keyboard,
            )
            integrity_errors = _episode_integrity_errors(
                dataset,
                depth_writer,
                frames,
                reject_missing_depth=config.reject_episode_on_missing_depth,
            )
            if saved and not integrity_errors:
                sidecar_dir = extras_root / f"episode_{saved_episodes:06d}"
                depth_writer.finish_episode(save=True)
                try:
                    dataset.save_episode()
                except BaseException:
                    shutil.rmtree(sidecar_dir, ignore_errors=True)
                    with contextlib.suppress(Exception):
                        dataset.clear_episode_buffer()
                    raise
                saved_episodes += 1
                LOGGER.info(
                    "Saved episode %d: %d frames (%d dropped samples)", saved_episodes - 1, frames, dropped
                )
            else:
                if integrity_errors:
                    LOGGER.error("Rejected episode: %s", "; ".join(integrity_errors))
                depth_writer.finish_episode(save=False)
                dataset.clear_episode_buffer()
                LOGGER.info("Discarded episode after %d frames", frames)
            auto = args.auto_start
            if keyboard.quit_requested or keyboard.get() == "q":
                quit_requested = True
        dataset.finalize()
        return 0
    finally:
        if keyboard is not None:
            keyboard.close()
        if depth_writer is not None:
            with contextlib.suppress(Exception):
                depth_writer.finish_episode(save=False)
        if dataset is not None:
            with contextlib.suppress(Exception):
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
            with contextlib.suppress(Exception):
                dataset.finalize()
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
