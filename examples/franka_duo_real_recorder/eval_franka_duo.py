#!/usr/bin/env python3
"""Run an exported Franka Duo policy on live ROS camera input.

The executable is intentionally a relay-oriented evaluator.  It prints the
20D Cartesian action by default and never publishes to a robot controller
unless both ``--publish`` and ``--enable-robot`` are supplied.  The relay topic
and message type are explicit configuration so this tool cannot silently send
an action to an unrelated Franka driver.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .action_spec import FrankaDuoActionSpec
from .franka_duo_eval_io import (
    EvalCameraConfig,
    EvalObservationCache,
    PointCloudConfig,
    SynchronizedObservation,
    SynchronizedObservationReader,
)
from .record_franka_duo import _load_yaml_or_json
from .rl100_eval_policy import PolicyBundle, load_policy_bundle

LOGGER = logging.getLogger("franka_duo_eval")


@dataclasses.dataclass
class EvalConfig:
    topics: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "head_image": "/isaac/head_camera/image_raw",
            "head_depth": "/isaac/head_camera/depth",
            "head_camera_info": "/isaac/head_camera/camera_info",
            "wrist_left_image": "/isaac/left_wrist_camera/image_raw",
            "wrist_right_image": "/isaac/right_wrist_camera/image_raw",
            "joint_states": "/isaac/joint_states_full",
        }
    )
    cameras: dict[str, EvalCameraConfig] = dataclasses.field(default_factory=dict)
    fps: int = 30
    sync_history_size: int = 30
    rgb_match_tolerance_ms: float = 45.0
    depth_match_tolerance_ms: float = 16.0
    state_match_tolerance_ms: float = 50.0
    max_message_age_ms: float = 150.0
    sync_wait_timeout_ms: float = 75.0
    state_gripper_closed: float | None = None
    state_gripper_open: float | None = None
    publish_topic: str = "/franka_duo/policy_action"
    output_jsonl: Path | None = None
    inference_timeout_ms: float = 100.0
    max_steps: int | None = None


def _default_cameras(topics: Mapping[str, str]) -> dict[str, EvalCameraConfig]:
    return {
        "head": EvalCameraConfig(
            key="head",
            image_topic=topics["head_image"],
            depth_topic=topics["head_depth"],
            camera_info_topic=topics["head_camera_info"],
            width=1280,
            height=720,
            fps=30,
            depth_scale=0.001,
        ),
        "wrist_left": EvalCameraConfig(
            key="wrist_left",
            image_topic=topics["wrist_left_image"],
            width=480,
            height=270,
            fps=30,
        ),
        "wrist_right": EvalCameraConfig(
            key="wrist_right",
            image_topic=topics["wrist_right_image"],
            width=480,
            height=270,
            fps=30,
        ),
    }


def load_eval_config(path: Path | None) -> EvalConfig:
    config = EvalConfig()
    config.cameras = _default_cameras(config.topics)
    if path is None:
        return config
    raw = _load_yaml_or_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError("eval config must contain a mapping")
    if isinstance(raw.get("topics"), Mapping):
        config.topics.update({str(key): str(value) for key, value in raw["topics"].items()})
    for key in (
        "fps",
        "sync_history_size",
        "rgb_match_tolerance_ms",
        "depth_match_tolerance_ms",
        "state_match_tolerance_ms",
        "max_message_age_ms",
        "sync_wait_timeout_ms",
        "state_gripper_closed",
        "state_gripper_open",
        "publish_topic",
        "inference_timeout_ms",
        "max_steps",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    if raw.get("output_jsonl") is not None:
        config.output_jsonl = Path(str(raw["output_jsonl"]))
    camera_values = raw.get("cameras", {})
    if not isinstance(camera_values, Mapping):
        raise ValueError("eval config cameras must contain a mapping")
    defaults = _default_cameras(config.topics)
    cameras: dict[str, EvalCameraConfig] = {}
    for key, default in defaults.items():
        value = camera_values.get(key, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"camera config {key} must contain a mapping")
        cameras[key] = EvalCameraConfig(
            key=key,
            image_topic=str(value.get("image", default.image_topic)),
            depth_topic=value.get("depth", default.depth_topic),
            camera_info_topic=value.get("camera_info", default.camera_info_topic),
            width=int(value.get("width", default.width)),
            height=int(value.get("height", default.height)),
            fps=int(value.get("fps", default.fps)),
            depth_scale=float(value.get("depth_scale", default.depth_scale)),
            depth_registered=bool(value.get("depth_registered", default.depth_registered)),
        )
    config.cameras = cameras
    if config.fps <= 0 or config.sync_history_size <= 0:
        raise ValueError("fps and sync_history_size must be positive")
    for key in (
        "rgb_match_tolerance_ms",
        "depth_match_tolerance_ms",
        "state_match_tolerance_ms",
        "max_message_age_ms",
        "sync_wait_timeout_ms",
        "inference_timeout_ms",
    ):
        if float(getattr(config, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.max_steps is not None and int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive when set")
    if not config.cameras["head"].depth_topic or not config.cameras["head"].camera_info_topic:
        raise ValueError("head depth and camera_info topics are required for ZED point-cloud eval")
    for key, camera in config.cameras.items():
        if camera.fps != config.fps:
            raise ValueError(f"camera {key} fps {camera.fps} does not match eval fps {config.fps}")
    return config


def _source_frame(observation: SynchronizedObservation, source: str | Sequence[str]) -> np.ndarray:
    sources = {
        "head": observation.head_rgb,
        "wrist_left": observation.wrist_left_rgb,
        "wrist_right": observation.wrist_right_rgb,
        "point_cloud": observation.point_cloud,
        "state": observation.state,
    }
    if isinstance(source, str):
        if source not in sources or sources[source] is None:
            raise ValueError(f"Policy input source {source!r} is not available in this observation")
        return np.asarray(sources[source])
    source_names = [str(item) for item in source]
    if not source_names:
        raise ValueError("Policy input source list cannot be empty")
    frames = [_source_frame(observation, item) for item in source_names]
    if any(frame.ndim != 3 for frame in frames) or any(frame.shape[0] != frames[0].shape[0] for frame in frames):
        raise ValueError("Stacked image sources must be HWC arrays with matching heights")
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def build_policy_observation(observation: SynchronizedObservation, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Map physical sensor names to the exact keys used during policy training."""

    raw_inputs = manifest.get("inputs", manifest.get("observation", {}))
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("manifest.inputs must be a mapping")
    point_key = raw_inputs.get("point_cloud_key", "observation.point_cloud")
    if point_key is None:
        raise ValueError("manifest.inputs.point_cloud_key is required")
    result: dict[str, np.ndarray] = {str(point_key): observation.point_cloud}
    state_key = raw_inputs.get("state_key")
    if state_key:
        if observation.state is None:
            raise ValueError(f"Policy requires {state_key}, but no synchronized state is available")
        result[str(state_key)] = observation.state
    image_keys = raw_inputs.get("image_keys", {})
    if not isinstance(image_keys, Mapping):
        raise ValueError("manifest.inputs.image_keys must be a mapping")
    for model_key, source in image_keys.items():
        if not isinstance(source, str) and not isinstance(source, Sequence):
            raise ValueError(f"manifest image source for {model_key} must be a string or string list")
        result[str(model_key)] = _source_frame(observation, source)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Exported model bundle directory")
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Bare checkpoint directory; pair with --manifest when manifest.json is not inside it",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit manifest for --checkpoint (manifest paths are resolved separately from weights)",
    )
    parser.add_argument("--config", type=Path, default=None, help="ROS topics/camera YAML or JSON")
    parser.add_argument("--device", default="auto", help="torch device used by the policy")
    parser.add_argument("--publish", action="store_true", help="Publish to the configured relay topic")
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="Required second gate for publishing; use only after checking relay wiring",
    )
    parser.add_argument("--once", action="store_true", help="Infer one synchronized frame and exit")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Also append actions to JSONL")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _ros_node(config: EvalConfig, cache: EvalObservationCache, *, require_state: bool):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
    except ImportError as exc:  # pragma: no cover - requires robot host
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source the robot ROS environment first"
        ) from exc

    class FrankaDuoEvalNode(Node):
        def __init__(self):
            super().__init__("franka_duo_policy_eval")
            self.create_subscription(
                Image,
                config.cameras["head"].image_topic,
                lambda msg: cache.store_image("head", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                config.cameras["head"].depth_topic,
                cache.store_depth,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                config.cameras["head"].camera_info_topic,
                cache.store_camera_info,
                qos_profile_sensor_data,
            )
            for key in ("wrist_left", "wrist_right"):
                self.create_subscription(
                    Image,
                    config.cameras[key].image_topic,
                    lambda msg, key=key: cache.store_image(key, msg),
                    qos_profile_sensor_data,
                )
            if require_state:
                self.create_subscription(
                    JointState,
                    config.topics["joint_states"],
                    cache.store_joint_states,
                    qos_profile_sensor_data,
                )

    return rclpy, FrankaDuoEvalNode()


def _action_record(observation: SynchronizedObservation, action: np.ndarray) -> dict[str, Any]:
    return {
        "stamp_ns": int(observation.stamp_ns),
        "source_stamps_ns": observation.source_stamps_ns,
        "action": [float(value) for value in action],
    }


def run(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    checkpoint = args.bundle if args.bundle is not None else args.checkpoint
    if args.manifest is not None and args.bundle is not None:
        raise ValueError("--manifest is only needed with --checkpoint")
    bundle: PolicyBundle = load_policy_bundle(
        checkpoint,
        device=args.device,
        manifest_path=args.manifest,
    )
    action_spec: FrankaDuoActionSpec = bundle.action_spec
    pointcloud: PointCloudConfig = bundle.pointcloud_config
    require_state = bool(getattr(bundle, "requires_state", False))
    if require_state and (config.state_gripper_closed is None or config.state_gripper_open is None):
        raise ValueError("This bundle consumes state; set state_gripper_closed/state_gripper_open in eval config")
    if args.publish and not args.enable_robot:
        raise ValueError("--publish requires the explicit second safety gate --enable-robot")
    if args.publish and not config.publish_topic:
        raise ValueError("publish_topic must be configured explicitly")
    if args.publish and action_spec.workspace_min is None:
        raise ValueError("publishing requires manifest workspace_min/workspace_max safety limits")
    max_steps = 1 if args.once else args.max_steps if args.max_steps is not None else config.max_steps
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    cache = EvalObservationCache(history_size=config.sync_history_size)
    rclpy, node = _ros_node(config, cache, require_state=require_state)
    publisher = None
    if args.publish:
        from std_msgs.msg import Float32MultiArray

        publisher = node.create_publisher(Float32MultiArray, config.publish_topic, 10)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), name="franka-eval-ros-spin", daemon=True)
    spin_thread.start()
    reader = SynchronizedObservationReader(
        cache,
        config.cameras,
        pointcloud,
        rgb_tolerance_ms=config.rgb_match_tolerance_ms,
        depth_tolerance_ms=config.depth_match_tolerance_ms,
        state_tolerance_ms=config.state_match_tolerance_ms,
        max_message_age_ms=config.max_message_age_ms,
        sync_wait_timeout_ms=config.sync_wait_timeout_ms,
        state_gripper_closed=config.state_gripper_closed,
        state_gripper_open=config.state_gripper_open,
    )
    output_file = args.output_jsonl or config.output_jsonl
    output_handle = None
    try:
        output_handle = output_file.open("a", encoding="utf-8") if output_file else None
        bundle.reset()
        steps = 0
        while max_steps is None or steps < max_steps:
            observation = reader.next(timeout_s=1.0, require_state=require_state)
            model_observation = build_policy_observation(observation, bundle.manifest)
            started = time.monotonic()
            action = np.asarray(bundle.predict(model_observation), dtype=np.float32).reshape(-1)
            inference_ms = (time.monotonic() - started) * 1000.0
            if inference_ms > float(config.inference_timeout_ms):
                raise TimeoutError(
                    f"policy inference took {inference_ms:.1f} ms, exceeding {config.inference_timeout_ms:.1f} ms"
                )
            action = action_spec.validate(action)
            record = _action_record(observation, action)
            record["inference_ms"] = inference_ms
            print(json.dumps(record, separators=(",", ":"), ensure_ascii=True), flush=True)
            if output_handle is not None:
                output_handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")
                output_handle.flush()
            if publisher is not None:
                from std_msgs.msg import Float32MultiArray

                message = Float32MultiArray()
                message.data = action.tolist()
                publisher.publish(message)
            steps += 1
    finally:
        if output_handle is not None:
            output_handle.close()
        bundle.reset()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
