"""Generic real-robot environment adapter for RL-100 online training.

The RL-100 online trainer intentionally depends only on the small Gymnasium
vector-environment surface (``num_envs``, ``reset``, ``step`` and ``close``).
This module provides a single-robot adapter for Franka Duo/other hardware
backends that expose callbacks rather than a native Gym environment.

The adapter does *not* contain ROS or robot-driver imports.  A caller supplies
callbacks for reading observations, sending actions, and computing reward and
termination.  This keeps hardware setup outside the training process while
making the real-robot contract explicit and testable on a workstation.

Observation callbacks should return policy-keyed mappings, for example:

``observation.state`` ``[34]`` (or ``[1, 34]``), two wrist images in CHW/HWC
layout, and ``observation.point_cloud`` ``[2048, 3]``.  Images are converted
from HWC uint8 to CHW float32 in ``[0, 1]``; all other floating features are
validated but not normalized (the policy checkpoint processor performs that
step).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

Observation = Mapping[str, Any]


@runtime_checkable
class VectorEnvProtocol(Protocol):
    """Minimal Gymnasium vector-environment API consumed by ``OnlineTrainer``."""

    num_envs: int

    def reset(self, *, seed: int | Sequence[int] | None = None) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...

    def close(self) -> None: ...


def validate_vector_env(env: Any, *, expected_num_envs: int | None = None) -> int:
    """Validate an environment before handing it to the online trainer."""

    num_envs = getattr(env, "num_envs", None)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("online environment must expose a positive integer num_envs")
    if expected_num_envs is not None and num_envs != expected_num_envs:
        raise ValueError(
            f"online environment num_envs={num_envs} disagrees with requested num_envs={expected_num_envs}"
        )
    for method in ("reset", "step", "close"):
        if not callable(getattr(env, method, None)):
            raise ValueError(f"online environment must provide callable {method}()")
    return num_envs


def _prepare_observation(observation: Observation) -> dict[str, np.ndarray]:
    if not isinstance(observation, Mapping) or not observation:
        raise ValueError("real-robot observation must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if not isinstance(key, str) or not key:
            raise ValueError("real-robot observation keys must be non-empty strings")
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"observation feature {key!r} must have at least one dimension")

        # The trainer requires a leading world axis.  A single-robot adapter
        # accepts either one frame ([...]) or an already batched [1, ...].
        if array.shape[0] != 1:
            if key == "observation.state" or key == "state":
                array = array[None, ...]
            elif key.startswith("observation.images.") or key == "observation.image":
                # HWC RGB image (the common ROS/OpenCV representation).
                if array.ndim == 3 and (array.shape[-1] == 3 or array.shape[0] == 3):
                    array = array[None, ...]
                else:
                    raise ValueError(f"image feature {key!r} must be CHW or HWC, got {array.shape}")
            else:
                # Point clouds and auxiliary observations are also single
                # frames for this adapter.
                array = array[None, ...]

        if key.startswith("observation.images.") or key == "observation.image":
            if array.ndim == 4 and array.shape[-1] == 3 and array.shape[1] != 3:
                array = np.transpose(array, (0, 3, 1, 2))
            if array.ndim != 4 or array.shape[1] != 3:
                raise ValueError(
                    f"image feature {key!r} must have batched CHW shape [1,3,H,W], got {array.shape}"
                )
            if array.dtype == np.uint8:
                array = array.astype(np.float32) / 255.0
            else:
                array = array.astype(np.float32, copy=False)
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)

        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"observation feature {key!r} contains non-finite values")
        result[key] = np.ascontiguousarray(array)
    if "observation.state" not in result and "state" in result:
        result["observation.state"] = result.pop("state")
    if "observation.state" not in result:
        raise ValueError("real-robot observation must contain 'observation.state'")
    return result


ReadObservation = Callable[[], Observation]
SendAction = Callable[[np.ndarray], Any]
ResetRobot = Callable[[], Any]
CloseRobot = Callable[[], Any]
RewardFn = Callable[[Observation, np.ndarray, Observation, Mapping[str, Any]], float]
DoneFn = Callable[[Observation, np.ndarray, Observation, Mapping[str, Any]], bool]
InfoFn = Callable[[Observation, np.ndarray, Observation], Mapping[str, Any]]


class RealRobotEnvAdapter:
    """Expose callback-based hardware as a one-world Gymnasium environment.

    Reward and termination callbacks are optional so callers can implement
    either sparse terminal-success labels or dense environment rewards. When
    omitted, the adapter returns reward ``0`` and a nonterminal transition;
    production callers should provide at least a termination/truncation rule
    so a real-robot rollout cannot run indefinitely.
    """

    num_envs = 1

    def __init__(
        self,
        *,
        read_observation: ReadObservation,
        send_action: SendAction,
        reset_robot: ResetRobot | None = None,
        close_robot: CloseRobot | None = None,
        reward_fn: RewardFn | None = None,
        terminated_fn: DoneFn | None = None,
        truncated_fn: DoneFn | None = None,
        info_fn: InfoFn | None = None,
        action_space: Any = None,
    ) -> None:
        for name, callback in (
            ("read_observation", read_observation),
            ("send_action", send_action),
        ):
            if not callable(callback):
                raise ValueError(f"{name} must be callable")
        callbacks: tuple[tuple[str, object], ...] = (
            ("reset_robot", reset_robot),
            ("close_robot", close_robot),
            ("reward_fn", reward_fn),
            ("terminated_fn", terminated_fn),
            ("truncated_fn", truncated_fn),
            ("info_fn", info_fn),
        )
        for name, callback_obj in callbacks:
            if callback_obj is not None and not callable(callback_obj):
                raise ValueError(f"{name} must be callable when supplied")
        self._read_observation = read_observation
        self._send_action = send_action
        self._reset_robot = reset_robot
        self._close_robot = close_robot
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._truncated_fn = truncated_fn
        self._info_fn = info_fn
        self.action_space = action_space
        self._previous_observation: dict[str, np.ndarray] | None = None

    def reset(
        self, *, seed: int | Sequence[int] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del seed  # Hardware reset/calibration is owned by the caller callback.
        if self._reset_robot is not None:
            self._reset_robot()
        observation = _prepare_observation(self._read_observation())
        self._previous_observation = observation
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float32)
        if command.ndim == 1:
            command = command[None, :]
        if command.ndim != 2 or command.shape[0] != 1:
            raise ValueError(f"real-robot action must have shape [1, action_dim], got {command.shape}")
        if not np.isfinite(command).all():
            raise ValueError("real-robot action contains non-finite values")
        if self._previous_observation is None:
            raise RuntimeError("reset() must be called before step()")
        previous = self._previous_observation
        self._send_action(command[0].copy())
        observation = _prepare_observation(self._read_observation())
        info = {} if self._info_fn is None else dict(self._info_fn(previous, command[0], observation))
        reward = (
            0.0
            if self._reward_fn is None
            else float(self._reward_fn(previous, command[0], observation, info))
        )
        terminated = (
            False
            if self._terminated_fn is None
            else bool(self._terminated_fn(previous, command[0], observation, info))
        )
        truncated = (
            False
            if self._truncated_fn is None
            else bool(self._truncated_fn(previous, command[0], observation, info))
        )
        if not np.isfinite(reward):
            raise ValueError("real-robot reward must be finite")
        self._previous_observation = observation
        return (
            observation,
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated], dtype=np.bool_),
            np.asarray([truncated], dtype=np.bool_),
            info,
        )

    def close(self) -> None:
        if self._close_robot is not None:
            self._close_robot()


__all__ = ["RealRobotEnvAdapter", "VectorEnvProtocol", "validate_vector_env"]
