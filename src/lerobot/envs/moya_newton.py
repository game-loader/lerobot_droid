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
import importlib
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from numbers import Integral
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import gymnasium as gym
import numpy as np

EXPECTED_MOYA_CONTRACT_SHA256 = "a06cb5e5a51c3b53a1972e77d803a15c0916395f9c15ecd7172433cc1e6c293e"
MOYA_CONTRACT_SCHEMA_VERSION = 14
_RANDOMIZED_GRASP_V1 = MappingProxyType(
    {
        "MOYA_IK_ITERS": "24",
        "MOYA_RIGHT_IK_ROTATION_WEIGHT": "1.0",
        "MOYA_CHARGER_GRASP_WRIST_ACTION_SCALE": "0.02",
        "MOYA_CHARGER_GRASP_ROTATION_ACTION_SCALE": "0.1",
        "MOYA_HAND_CLOSE_CONTROL_RADIUS": "0.03",
        "MOYA_CHARGER_X": "0.37",
        "MOYA_CHARGER_Y": "0.00",
        "MOYA_CHARGER_Z": "1.095",
        "MOYA_CHARGER_MASS": "0.5",
        "MOYA_CHARGER_BOX_SCALE": "2.0",
        "MOYA_CHARGER_CROSS_HALF": "0.0125",
        "MOYA_CHARGER_GRASP_REFERENCE_X_OFFSET": "-0.04",
        "MOYA_CHARGER_GRASP_REFERENCE_Y_OFFSET": "-0.03",
    }
)
MOYA_PRESETS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {"randomized_grasp_v1": _RANDOMIZED_GRASP_V1}
)

_BackendLoader = Callable[[], tuple[Callable[..., gym.vector.VectorEnv], Any]]
_MISSING = object()


def _default_submodule_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "moya_newton_sim"


@contextmanager
def _preset_environment(preset: Mapping[str, str]) -> Iterator[None]:
    original_moya = {name: value for name, value in os.environ.items() if name.startswith("MOYA_")}
    original_pyglet = os.environ.get("PYGLET_HEADLESS", _MISSING)
    for name in tuple(os.environ):
        if name.startswith("MOYA_"):
            del os.environ[name]
    os.environ.update(preset)
    os.environ["PYGLET_HEADLESS"] = "1"
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith("MOYA_"):
                del os.environ[name]
        os.environ.update(original_moya)
        if original_pyglet is _MISSING:
            os.environ.pop("PYGLET_HEADLESS", None)
        else:
            os.environ["PYGLET_HEADLESS"] = str(original_pyglet)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(value)


def _module_origins(module: ModuleType) -> list[Path]:
    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(module_file).resolve())
    module_path = getattr(module, "__path__", None)
    if module_path:
        origins.extend(Path(path).resolve() for path in module_path)
    return origins


def _validate_moya_modules(submodule_root: Path, *, require_unloaded: bool) -> None:
    for name, module in tuple(sys.modules.items()):
        if not (name in {"moya_batched_env", "moya_model", "rewards"} or name.startswith("rewards.")):
            continue
        if module is None:
            continue
        if require_unloaded:
            raise RuntimeError(
                f"Python module {name!r} was already loaded before the current Moya dataset preset import. "
                "Start evaluation in a fresh Python process without importing Moya modules first."
            )
        origins = _module_origins(module)
        if not origins or any(not origin.is_relative_to(submodule_root) for origin in origins):
            raise RuntimeError(
                f"Python module {name!r} is already loaded outside the pinned Moya submodule. "
                "Start evaluation in a fresh Python process without another Moya checkout on PYTHONPATH."
            )


def _load_moya_backend(
    submodule_root: Path,
) -> tuple[Callable[..., gym.vector.VectorEnv], ModuleType]:
    required_paths = (
        submodule_root / "moya_batched_env.py",
        submodule_root / "moya_model.py",
        submodule_root / "rewards" / "reward_api.py",
    )
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(
            f"Moya Newton submodule is missing or uninitialized at {submodule_root}. "
            f"Missing required files: {missing_paths}. "
            "Run `git submodule update --init --recursive`."
        )

    submodule_root = submodule_root.resolve()
    _validate_moya_modules(submodule_root, require_unloaded=True)
    with _prepend_sys_path(submodule_root):
        try:
            env_module = importlib.import_module("moya_batched_env")
            model_module = importlib.import_module("moya_model")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Moya Newton dependencies are not installed. Run "
                "`UV_PROJECT_ENVIRONMENT=.venv uv sync --extra moya_newton`."
            ) from exc

    _validate_moya_modules(submodule_root, require_unloaded=False)
    backend_cls = getattr(env_module, "MoyaBatchedChargerGraspEnv", None)
    if backend_cls is None:
        raise RuntimeError("Pinned Moya module does not define MoyaBatchedChargerGraspEnv")
    return backend_cls, model_module


def _close_backend(backend: Any) -> None:
    with suppress(Exception):
        backend.close()


def create_moya_newton_env(
    *,
    num_envs: int,
    episode_length: int,
    device: str,
    headless: bool,
    sim_substeps: int,
    preset: str,
    task: str,
    task_description: str,
    success_min_final_lift_height: float,
    submodule_root: Path | None = None,
    _backend_loader: _BackendLoader | None = None,
) -> MoyaNewtonVectorEnv:
    """Construct and validate the pinned Moya backend under a dataset preset."""

    if preset not in MOYA_PRESETS:
        raise ValueError(f"Unknown Moya preset {preset!r}. Available presets: {sorted(MOYA_PRESETS)}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if sim_substeps <= 0:
        raise ValueError("sim_substeps must be positive")
    if not headless:
        raise ValueError("Moya LeRobot evaluation currently supports headless=True only")

    preset_values = MOYA_PRESETS[preset]
    backend: gym.vector.VectorEnv | None = None
    with _preset_environment(preset_values):
        loader = _backend_loader or (lambda: _load_moya_backend(submodule_root or _default_submodule_root()))
        backend_cls, model_module = loader()
        backend = backend_cls(
            num_envs=num_envs,
            headless=headless,
            episode_length=episode_length,
            device=device,
            sim_substeps=sim_substeps,
        )

    try:
        if not isinstance(backend, gym.vector.VectorEnv):
            raise TypeError(
                "MoyaBatchedChargerGraspEnv must be a gymnasium.vector.VectorEnv, "
                f"got {type(backend).__name__}"
            )
        contract = backend.environment_contract()
        if not isinstance(contract, dict):
            raise RuntimeError("Moya environment_contract() must return a dictionary")
        if contract.get("schema_version") != MOYA_CONTRACT_SCHEMA_VERSION:
            raise RuntimeError(
                "Moya environment contract schema mismatch: expected "
                f"{MOYA_CONTRACT_SCHEMA_VERSION}, got {contract.get('schema_version')!r}"
            )
        if contract.get("sha256") != EXPECTED_MOYA_CONTRACT_SHA256:
            raise RuntimeError(
                "Moya environment contract SHA-256 mismatch: expected "
                f"{EXPECTED_MOYA_CONTRACT_SHA256}, got {contract.get('sha256')!r}"
            )
        expected_mass = float(preset_values["MOYA_CHARGER_MASS"])
        actual_mass = float(getattr(model_module, "CHARGER_MASS", float("nan")))
        if not np.isclose(actual_mass, expected_mass, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"Moya charger mass mismatch: expected {expected_mass}, got {actual_mass}")
    except Exception:
        _close_backend(backend)
        raise

    return MoyaNewtonVectorEnv(
        backend,
        episode_length=episode_length,
        task=task,
        task_description=task_description,
        success_min_final_lift_height=success_min_final_lift_height,
    )


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
            raise RuntimeError(f"reward_components[{name!r}] must have shape {(size,)}, got {value.shape}")
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
            and float(final_info["charger_lift_height"]) >= self.success_min_final_lift_height
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
        keys = sorted({key for item in final_info if isinstance(item, dict) for key in item})
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
        observation, reward, terminated, truncated, info = self.env.step(self._validate_action(action))
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
            final_mask = np.asarray(info.get("_final_info", done), dtype=np.bool_).reshape(self.num_envs)
            self._update_terminal_history(raw_final_info, final_mask)
            for index in np.flatnonzero(final_mask):
                item = raw_final_info[index]
                if not isinstance(item, dict):
                    raise RuntimeError(f"final_info[{index}] must be a dictionary")
                terminal_success[index] = self._terminal_success(item, int(index))
                item["is_success"] = bool(terminal_success[index])
                item["true_grasp_ever"] = bool(self._true_grasp_ever[index])
                item["clear_table_ever"] = bool(self._clear_table_ever[index])
                item["native_success"] = bool(item.get("success", item.get("charger_success", False)))
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
