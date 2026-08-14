# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import copy
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np


class MoyaNewtonVectorEnv(gym.vector.VectorWrapper):
    """Adapt Moya's fused vector environment to LeRobot's rollout contract."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        *,
        episode_length: int,
        task: str,
        task_description: str,
        success_min_final_lift_height: float = 0.015,
    ) -> None:
        super().__init__(env)
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if not np.isfinite(success_min_final_lift_height) or success_min_final_lift_height < 0.0:
            raise ValueError("success_min_final_lift_height must be finite and non-negative")

        self.single_observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(39,),
                    dtype=np.float32,
                )
            }
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            env.num_envs,
        )
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self._max_episode_steps = int(episode_length)
        self.task = task
        self.task_description = task_description
        self.success_min_final_lift_height = float(success_min_final_lift_height)
        self._true_grasp_ever = np.zeros(env.num_envs, dtype=np.bool_)
        self._clear_table_ever = np.zeros(env.num_envs, dtype=np.bool_)
        self._closed = False

    @staticmethod
    def _fold_seed(seed: int | Sequence[int] | None, num_envs: int) -> int | None:
        if seed is None:
            return None
        if isinstance(seed, Integral):
            return int(seed)
        if isinstance(seed, (str, bytes)) or not isinstance(seed, Sequence):
            raise TypeError("seed must be an integer, a sequence of integers, or None")
        if len(seed) != num_envs:
            raise ValueError(f"Expected {num_envs} seeds, got {len(seed)}")
        if not all(isinstance(value, Integral) for value in seed):
            raise TypeError("seed sequence must contain only integers")
        values = np.asarray(seed, dtype=np.uint32)
        return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])

    def _validate_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation)
        expected_shape = (self.num_envs, 39)
        if array.shape != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, got {array.shape}")
        if array.dtype != np.float32:
            raise TypeError(f"Expected observation dtype float32, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Moya observation must contain only finite values")
        return array

    def _validate_action(self, action: Any) -> np.ndarray:
        array = np.asarray(action)
        expected_shape = (self.num_envs, 14)
        if array.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {array.shape}")
        try:
            finite = bool(np.all(np.isfinite(array)))
        except TypeError as exc:
            raise TypeError("Moya action must be numeric") from exc
        if not finite:
            raise ValueError("Moya action must contain only finite values")
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _component_array(info: dict[str, Any], name: str, size: int) -> np.ndarray:
        components = info.get("reward_components")
        if not isinstance(components, dict) or name not in components:
            return np.zeros(size, dtype=np.float32)
        value = np.asarray(components[name], dtype=np.float32)
        if value.ndim == 0:
            return np.full(size, float(value), dtype=np.float32)
        if value.shape != (size,):
            raise RuntimeError(
                f"reward_components[{name!r}] must have shape {(size,)}, got {value.shape}"
            )
        return value

    def _update_history(self, info: dict[str, Any]) -> None:
        self._true_grasp_ever |= self._component_array(info, "true_grasp", self.num_envs) > 0.0
        self._clear_table_ever |= self._component_array(info, "clear_table", self.num_envs) > 0.0

    def _update_terminal_history(
        self,
        final_info: np.ndarray,
        final_mask: np.ndarray,
    ) -> None:
        for index in np.flatnonzero(final_mask):
            item = final_info[index]
            if not isinstance(item, dict):
                raise RuntimeError(f"final_info[{index}] must be a dictionary")
            components = item.get("reward_components", {})
            if isinstance(components, dict):
                self._true_grasp_ever[index] |= float(components.get("true_grasp", 0.0)) > 0.0
                self._clear_table_ever[index] |= float(components.get("clear_table", 0.0)) > 0.0

    def _terminal_success(self, final_info: dict[str, Any], index: int) -> bool:
        required_keys = (
            "charger_lift_height",
            "charger_table_contacts",
            "right_hand_charger_contacts",
        )
        missing = [key for key in required_keys if key not in final_info]
        if missing:
            raise RuntimeError(f"final_info is missing required success fields: {missing}")
        return bool(
            self._true_grasp_ever[index]
            and self._clear_table_ever[index]
            and float(final_info["charger_lift_height"])
            >= self.success_min_final_lift_height
            and int(final_info["charger_table_contacts"]) == 0
            and int(final_info["right_hand_charger_contacts"]) > 0
        )

    @classmethod
    def _collate_values(cls, values: list[Any]) -> Any:
        present = [value for value in values if value is not None]
        if not present:
            return np.asarray(values, dtype=object)
        if all(isinstance(value, dict) for value in present):
            keys = sorted({key for value in present for key in value})
            return {
                key: cls._collate_values(
                    [value.get(key) if isinstance(value, dict) else None for value in values]
                )
                for key in keys
            }

        sample = present[0]
        filled: list[Any] = []
        for value in values:
            if value is not None:
                filled.append(value)
            elif isinstance(sample, np.ndarray):
                filled.append(np.zeros_like(sample))
            elif isinstance(sample, (bool, np.bool_)):
                filled.append(False)
            elif isinstance(sample, (int, np.integer)):
                filled.append(0)
            elif isinstance(sample, (float, np.floating)):
                filled.append(0.0)
            else:
                filled.append(None)
        try:
            if isinstance(sample, np.ndarray):
                return np.stack(filled)
            return np.asarray(filled)
        except (TypeError, ValueError):
            return np.asarray(filled, dtype=object)

    @classmethod
    def _collate_final_info(cls, final_info: np.ndarray, size: int) -> dict[str, Any]:
        if final_info.shape != (size,):
            raise RuntimeError(f"final_info must have shape {(size,)}, got {final_info.shape}")
        keys = sorted(
            {
                key
                for item in final_info
                if isinstance(item, dict)
                for key in item
            }
        )
        return {
            key: cls._collate_values(
                [item.get(key) if isinstance(item, dict) else None for item in final_info]
            )
            for key in keys
        }

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        scalar_seed = self._fold_seed(seed, self.num_envs)
        observation, info = self.env.reset(seed=scalar_seed, options=options)
        self._true_grasp_ever.fill(False)
        self._clear_table_ever.fill(False)
        return {"agent_pos": self._validate_observation(observation)}, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        observation, reward, terminated, truncated, info = self.env.step(
            self._validate_action(action)
        )
        if not isinstance(info, dict):
            raise RuntimeError(f"Moya info must be a dictionary, got {type(info).__name__}")

        self._update_history(info)
        done = np.logical_or(terminated, truncated).astype(np.bool_, copy=False)
        terminal_success: np.ndarray = np.zeros(self.num_envs, dtype=np.bool_)
        native_success = np.asarray(
            info.get("success", np.zeros(self.num_envs, dtype=np.bool_)),
            dtype=np.bool_,
        ).reshape(self.num_envs)

        if "final_info" in info:
            raw_final_info = info["final_info"]
            if not isinstance(raw_final_info, np.ndarray) or raw_final_info.dtype != object:
                raise RuntimeError("Moya final_info must be an object ndarray")
            final_mask = np.asarray(info.get("_final_info", done), dtype=np.bool_).reshape(
                self.num_envs
            )
            self._update_terminal_history(raw_final_info, final_mask)
            for index in np.flatnonzero(final_mask):
                item = raw_final_info[index]
                if not isinstance(item, dict):
                    raise RuntimeError(f"final_info[{index}] must be a dictionary")
                terminal_success[index] = self._terminal_success(item, int(index))
                item["is_success"] = bool(terminal_success[index])
                item["true_grasp_ever"] = bool(self._true_grasp_ever[index])
                item["clear_table_ever"] = bool(self._clear_table_ever[index])
                item["native_success"] = bool(
                    item.get("success", item.get("charger_success", False))
                )
            info["final_info"] = self._collate_final_info(raw_final_info, self.num_envs)
            info["_final_info"] = final_mask

        info["is_success"] = terminal_success
        info["native_success"] = native_success
        self._true_grasp_ever[done] = False
        self._clear_table_ever[done] = False
        return (
            {"agent_pos": self._validate_observation(observation)},
            reward,
            terminated,
            truncated,
            info,
        )

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        value = getattr(self, name)
        value = value(*args, **kwargs) if callable(value) else value
        return tuple(copy.deepcopy(value) for _ in range(self.num_envs))

    def get_attr(self, name: str) -> tuple[Any, ...]:
        return self.call(name)

    def state(self) -> dict[str, np.ndarray]:
        return {"agent_pos": self._validate_observation(self.env.state())}

    def environment_contract(self) -> dict[str, Any]:
        return copy.deepcopy(self.env.environment_contract())

    def render(self) -> None:
        raise NotImplementedError("Rendering is disabled for the Moya Newton environment")

    def close(self, **kwargs: Any) -> None:
        if self._closed:
            return
        self.env.close(**kwargs)
        self._closed = True
