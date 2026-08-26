#!/usr/bin/env python3
"""Manually collect rewarded Franka Duo episodes into LeRobot v3.

This is the reward-annotation front end for ``record_franka_duo.py``.  The
ROS synchronizer, depth sidecar writer, and LeRobot video writer remain shared
with the regular recorder.  Press ``r`` while idle to start an episode; press
``e`` (or the legacy ``s`` key) to end it.  After the terminal is restored,
the CLI asks for one scalar episode reward and writes it to the final
transition as ``next.reward``/``next.done``/``next.truncated``.

The reward is intentionally terminal-only: all preceding transitions have
zero reward, the final transition has the entered reward, ``next.done`` is
true, and ``next.truncated`` is false.  This makes the episode-level human
judgement explicit while retaining the canonical LeRobot RL field names.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support both ``python -m`` and direct script execution.
    from .record_franka_duo import (
        DepthSidecarWriter,
        KeyboardCommands,
        RecorderConfig,
        _create_dataset,
        _episode_integrity_errors,
        _ros_node,
        camera_specs_from_names,
        load_config,
        record_episode,
        validate_config,
    )
except ImportError:  # pragma: no cover - exercised by direct ROS-host invocation
    from examples.franka_duo_real_recorder.record_franka_duo import (
        DepthSidecarWriter,
        KeyboardCommands,
        RecorderConfig,
        _create_dataset,
        _episode_integrity_errors,
        _ros_node,
        camera_specs_from_names,
        load_config,
        record_episode,
        validate_config,
    )

LOGGER = logging.getLogger("franka_duo_manual_recorder")


def annotate_episode_reward(dataset: Any, reward: float, *, truncated: bool = False) -> None:
    """Annotate the pending LeRobot episode with one terminal reward.

    ``LeRobotDataset.add_frame`` stores values in the writer's episode buffer.
    Patching the buffer after capture avoids delaying video encoding or keeping
    all RGB arrays in a second Python-side list while the operator enters the
    reward.  The function is pure with respect to saved data: callers still
    decide whether to call ``save_episode`` or ``clear_episode_buffer``.
    """

    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError("reward must be a finite number")
    writer = getattr(dataset, "writer", None)
    buffer = getattr(writer, "episode_buffer", None)
    if not isinstance(buffer, dict) or int(buffer.get("size", 0)) <= 0:
        raise ValueError("No pending episode frames are available for reward annotation")
    size = int(buffer["size"])
    required = ("next.reward", "next.done", "next.truncated")
    missing = [key for key in required if key not in buffer]
    if missing:
        raise ValueError(f"Dataset does not contain transition annotation fields: {missing}")

    buffer["next.reward"] = [np.zeros((1,), dtype=np.float32) for _ in range(size)]
    buffer["next.done"] = [np.zeros((1,), dtype=np.bool_) for _ in range(size)]
    buffer["next.truncated"] = [np.zeros((1,), dtype=np.bool_) for _ in range(size)]
    buffer["next.reward"][-1][0] = np.float32(reward)
    buffer["next.done"][-1][0] = True
    buffer["next.truncated"][-1][0] = bool(truncated)


def parse_reward(value: str) -> float:
    """Parse one finite scalar reward from terminal CLI input."""

    try:
        reward = float(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reward {value!r}; enter a finite number") from exc
    if not math.isfinite(reward):
        raise ValueError(f"Invalid reward {value!r}; enter a finite number")
    return reward


def prompt_reward(input_fn=input, output_fn=print) -> float:
    """Prompt until the operator enters a finite scalar reward."""

    while True:
        try:
            value = input_fn("Episode reward (finite scalar): ")
        except EOFError as exc:
            raise RuntimeError("Reward input ended before an episode reward was provided") from exc
        try:
            return parse_reward(value)
        except ValueError as exc:
            output_fn(f"{exc}. Please try again.")


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
    parser.add_argument("--depth-every", type=int, default=None)
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
    parser.add_argument("--gripper-closed-rad", type=float, default=None)
    parser.add_argument("--gripper-open-rad", type=float, default=None)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _apply_overrides(config: RecorderConfig, args: argparse.Namespace) -> None:
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
    ):
        if value is not None:
            setattr(config, key, value)


def _prompt_reward_with_keyboard_paused(keyboard: KeyboardCommands) -> float:
    """Restore cooked terminal input before asking for a line of text."""

    keyboard.close()
    return prompt_reward()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    _apply_overrides(config, args)
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
    depth_writer: Any | None = None
    keyboard: KeyboardCommands | None = None
    try:
        dataset, dataset_path, version = _create_dataset(
            config,
            cameras,
            resume=args.resume,
            include_transition_fields=True,
        )
        extras_root = dataset_path / "franka_duo_extras"
        depth_writer = DepthSidecarWriter(
            extras_root,
            queue_size=config.depth_queue_size,
            head_to_robot_base_transform=config.head_to_robot_base_transform,
        )
        keyboard = KeyboardCommands()
        keyboard.start()
        if not keyboard.active:
            raise RuntimeError("Manual collection requires a TTY; use the terminal directly")
        LOGGER.info(
            "Dataset: %s (v%d), manual reward fields enabled, cameras=%s",
            dataset_path,
            version,
            ",".join(cameras),
        )
        LOGGER.info("Idle: r=start, q=quit. Recording: e=end+reward, s=end+reward, d=discard.")

        saved_episodes = int(getattr(dataset.meta, "total_episodes", 0))
        quit_requested = False
        while saved_episodes < config.max_episodes and not quit_requested:
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
                include_transition_fields=True,
            )
            if not saved:
                depth_writer.finish_episode(save=False)
                dataset.clear_episode_buffer()
                if keyboard.quit_requested:
                    quit_requested = True
                LOGGER.info("Discarded episode after %d frames", frames)
                continue

            if frames <= 0:
                # There is no terminal transition to annotate or save. This
                # can happen when the operator presses the end key before the
                # first synchronized frame arrives.
                depth_writer.finish_episode(save=False)
                dataset.clear_episode_buffer()
                LOGGER.warning("Discarded empty episode; wait for a synchronized frame before ending")
                continue

            # ``record_episode`` returns on e/s while the keyboard is in cbreak
            # mode.  Restore it before input() and start a fresh listener after
            # the reward line has been consumed.
            end_command = keyboard.last_command
            try:
                reward = _prompt_reward_with_keyboard_paused(keyboard)
            except (KeyboardInterrupt, RuntimeError):
                depth_writer.finish_episode(save=False)
                dataset.clear_episode_buffer()
                LOGGER.warning("Reward input aborted; discarded episode after %d frames", frames)
                break
            keyboard = KeyboardCommands()
            keyboard.start()

            # A wall-clock timeout is a truncation rather than an operator
            # termination.  Explicit ``e``/``s`` ends are normal done states.
            annotate_episode_reward(
                dataset,
                reward,
                truncated=end_command not in {"e", "s"},
            )
            integrity_errors = _episode_integrity_errors(
                dataset,
                depth_writer,
                frames,
                reject_missing_depth=config.reject_episode_on_missing_depth,
            )
            if integrity_errors:
                LOGGER.error("Rejected episode: %s", "; ".join(integrity_errors))
                depth_writer.finish_episode(save=False)
                dataset.clear_episode_buffer()
                continue

            sidecar_dir = extras_root / f"episode_{saved_episodes:06d}"
            depth_writer.finish_episode(save=True)
            try:
                dataset.save_episode()
            except BaseException:
                shutil.rmtree(sidecar_dir, ignore_errors=True)
                with contextlib.suppress(Exception):
                    dataset.clear_episode_buffer()
                raise
            LOGGER.info(
                "Saved episode %d: %d frames, reward=%s (%d dropped samples)",
                saved_episodes,
                frames,
                reward,
                dropped,
            )
            saved_episodes += 1

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
