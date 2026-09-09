#!/usr/bin/env python3
"""Assemble the native TMR split ROS topics into recorder joint messages."""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

LOGGER = logging.getLogger("franka_duo_recorder.tmr_topics")

TMR_TOPIC_KEYS = (
    "left_measured_joint_states",
    "right_measured_joint_states",
    "left_desired_joint_states",
    "right_desired_joint_states",
    "left_gripper_target",
    "right_gripper_target",
    "left_gripper_joint_states",
    "right_gripper_joint_states",
)


@dataclasses.dataclass(frozen=True)
class TmrTimedMessage:
    message: Any
    arrival_ns: int
    stamp_ns: int | None


def message_stamp_ns(message: Any) -> int | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _header(stamp_ns: int, frame_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000),
        frame_id=frame_id,
    )


def _joint_values(message: Any) -> tuple[list[str], list[float]]:
    names = [str(name) for name in getattr(message, "name", ())]
    positions = [float(value) for value in getattr(message, "position", ())]
    if len(names) != len(positions) or not names:
        raise ValueError("JointState name/position arrays must be non-empty and have equal lengths")
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("JointState contains a non-finite position")
    return names, positions


def _select_joint_position(message: Any, configured_name: str | None, side: str) -> float:
    names, positions = _joint_values(message)
    if configured_name:
        try:
            return positions[names.index(configured_name)]
        except ValueError as exc:
            raise ValueError(
                f"{side} gripper JointState does not contain configured joint {configured_name!r}; "
                f"available names: {names}"
            ) from exc
    if len(positions) == 1:
        return positions[0]
    raise ValueError(
        f"{side} gripper JointState contains multiple joints {names}; set "
        f"tmr_gripper_actual_joint_names.{side} in the recorder config"
    )


class TmrSplitJointAssembler:
    """Join TMR arm/gripper streams into the recorder's semantic JointState pair.

    Arm messages retain their hardware timestamps. Gripper target messages are
    ``std_msgs/Float32`` and have no header, so their arrival age is checked and
    the assembled command uses the desired-arm timestamp as its anchor.
    """

    def __init__(
        self,
        *,
        on_measured: Callable[[Any], None],
        on_command: Callable[[Any], None],
        gripper_closed_position: float,
        gripper_open_position: float,
        gripper_actual_joint_names: Mapping[str, str | None] | None = None,
        sync_tolerance_ms: float = 20.0,
        max_age_ms: float = 200.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        if sync_tolerance_ms <= 0 or max_age_ms <= 0:
            raise ValueError("TMR split sync tolerance and max age must be positive")
        if not math.isfinite(gripper_closed_position) or not math.isfinite(gripper_open_position):
            raise ValueError("TMR gripper calibration endpoints must be finite")
        if math.isclose(gripper_closed_position, gripper_open_position, abs_tol=1e-12):
            raise ValueError("TMR gripper calibration endpoints must differ")
        self.on_measured = on_measured
        self.on_command = on_command
        self.closed = float(gripper_closed_position)
        self.open = float(gripper_open_position)
        self.actual_names = {"left": None, "right": None}
        if gripper_actual_joint_names:
            self.actual_names.update(
                {
                    str(side): None if name is None else str(name)
                    for side, name in gripper_actual_joint_names.items()
                }
            )
        self.sync_tolerance_ns = int(sync_tolerance_ms * 1_000_000)
        self.max_age_ns = int(max_age_ms * 1_000_000)
        self.clock_ns = clock_ns
        self.sources: dict[str, TmrTimedMessage] = {}
        self._last_arm_stamps = {"measured": (-1, -1), "command": (-1, -1)}
        self._last_error: dict[str, str] = {}

    def store(self, key: str, message: Any) -> None:
        if key not in TMR_TOPIC_KEYS:
            raise KeyError(f"Unknown TMR split source {key!r}")
        self.sources[key] = TmrTimedMessage(message, self.clock_ns(), message_stamp_ns(message))
        self._try_emit("measured")
        self._try_emit("command")

    def _fresh_sources(self, keys: Sequence[str]) -> list[TmrTimedMessage] | None:
        if any(key not in self.sources for key in keys):
            return None
        values = [self.sources[key] for key in keys]
        now = self.clock_ns()
        if any(now - value.arrival_ns > self.max_age_ns for value in values):
            return None
        return values

    def _arm_anchor(self, left: TmrTimedMessage, right: TmrTimedMessage) -> int | None:
        if left.stamp_ns is None or right.stamp_ns is None:
            return None
        if abs(left.stamp_ns - right.stamp_ns) > self.sync_tolerance_ns:
            return None
        return max(left.stamp_ns, right.stamp_ns)

    def _is_new_arm_pair(self, kind: str, left: TmrTimedMessage, right: TmrTimedMessage) -> bool:
        """Require both arm streams to advance before reusing their latest pair."""

        assert left.stamp_ns is not None and right.stamp_ns is not None
        previous_left, previous_right = self._last_arm_stamps[kind]
        return left.stamp_ns > previous_left and right.stamp_ns > previous_right

    def _try_emit(self, kind: str) -> None:
        try:
            if kind == "measured":
                keys = (
                    "left_measured_joint_states",
                    "right_measured_joint_states",
                    "left_gripper_joint_states",
                    "right_gripper_joint_states",
                )
                values = self._fresh_sources(keys)
                if values is None:
                    return
                left_arm, right_arm, left_gripper, right_gripper = values
                anchor = self._arm_anchor(left_arm, right_arm)
                if anchor is None or not self._is_new_arm_pair(kind, left_arm, right_arm):
                    return
                for source in (left_gripper, right_gripper):
                    if source.stamp_ns is not None and abs(source.stamp_ns - anchor) > self.sync_tolerance_ns:
                        return
                left_names, left_positions = _joint_values(left_arm.message)
                right_names, right_positions = _joint_values(right_arm.message)
                names = left_names + right_names + ["left_gripper", "right_gripper"]
                positions = (
                    left_positions
                    + right_positions
                    + [
                        _select_joint_position(left_gripper.message, self.actual_names["left"], "left"),
                        _select_joint_position(right_gripper.message, self.actual_names["right"], "right"),
                    ]
                )
                message = SimpleNamespace(
                    header=_header(anchor, "tmr_split_measured"), name=names, position=positions
                )
                self.on_measured(message)
            elif kind == "command":
                keys = (
                    "left_desired_joint_states",
                    "right_desired_joint_states",
                    "left_gripper_target",
                    "right_gripper_target",
                )
                values = self._fresh_sources(keys)
                if values is None:
                    return
                left_arm, right_arm, left_gripper, right_gripper = values
                anchor = self._arm_anchor(left_arm, right_arm)
                if anchor is None or not self._is_new_arm_pair(kind, left_arm, right_arm):
                    return
                left_names, left_positions = _joint_values(left_arm.message)
                right_names, right_positions = _joint_values(right_arm.message)
                targets = [float(left_gripper.message.data), float(right_gripper.message.data)]
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in targets):
                    raise ValueError(f"TMR gripper targets must be finite fractions in [0,1], got {targets}")
                raw_targets = [self.closed + value * (self.open - self.closed) for value in targets]
                message = SimpleNamespace(
                    header=_header(anchor, "tmr_split_desired"),
                    name=left_names + right_names + ["left_gripper", "right_gripper"],
                    position=left_positions + right_positions + raw_targets,
                )
                self.on_command(message)
            else:
                raise ValueError(f"Unknown assembled stream {kind!r}")
            self._last_arm_stamps[kind] = (left_arm.stamp_ns, right_arm.stamp_ns)
            self._last_error.pop(kind, None)
        except (AttributeError, TypeError, ValueError) as exc:
            message = str(exc)
            if self._last_error.get(kind) != message:
                LOGGER.warning("Cannot assemble TMR %s stream: %s", kind, message)
                self._last_error[kind] = message
