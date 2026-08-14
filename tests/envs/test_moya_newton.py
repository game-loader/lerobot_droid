#!/usr/bin/env python

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from lerobot.envs.moya_newton import (
    EXPECTED_MOYA_CONTRACT_SHA256,
    MOYA_PRESETS,
    MoyaNewtonVectorEnv,
    create_moya_newton_env,
)
from lerobot.envs.utils import preprocess_observation


class FakeMoyaVectorEnv(gym.vector.VectorEnv):
    metadata = {
        "render_modes": [None],
        "render_fps": 60,
        "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
    }

    def __init__(
        self,
        *,
        final_lift_height: float = 0.015,
        final_table_contacts: int = 0,
        final_hand_contacts: int = 1,
        true_grasp: bool = True,
        clear_table: bool = True,
        done_after_steps: int = 1,
    ) -> None:
        self.num_envs = 2
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.final_lift_height = final_lift_height
        self.final_table_contacts = final_table_contacts
        self.final_hand_contacts = final_hand_contacts
        self.true_grasp = true_grasp
        self.clear_table = clear_table
        self.done_after_steps = done_after_steps
        self.last_seed: int | None = None
        self.last_action: np.ndarray | None = None
        self.close_calls = 0
        self.step_count = 0
        self._state = np.zeros((self.num_envs, 39), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        self.last_seed = seed
        self.step_count = 0
        self._state.fill(0.0)
        return self._state.copy(), {"reset": np.ones(self.num_envs, dtype=np.bool_)}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.last_action = action.copy()
        self.step_count += 1
        self._state.fill(1.0)
        true_grasp_reward = 100.0 if self.true_grasp else 0.0
        clear_table_reward = 1000.0 if self.clear_table else 0.0
        final_info = np.asarray(
            [
                {
                    "success": env_id == 0,
                    "charger_lift_height": self.final_lift_height,
                    "charger_table_contacts": self.final_table_contacts,
                    "right_hand_charger_contacts": self.final_hand_contacts,
                    "reward_components": {
                        "true_grasp": true_grasp_reward,
                        "clear_table": clear_table_reward,
                    },
                }
                for env_id in range(self.num_envs)
            ],
            dtype=object,
        )
        done = np.full(self.num_envs, self.step_count >= self.done_after_steps, dtype=np.bool_)
        info = {
            "success": np.asarray([True, False], dtype=np.bool_),
            "reward_components": {
                "true_grasp": np.full(self.num_envs, true_grasp_reward, dtype=np.float32),
                "clear_table": np.full(self.num_envs, clear_table_reward, dtype=np.float32),
            },
        }
        if np.any(done):
            info.update(
                {
                    "final_info": final_info,
                    "_final_info": done.copy(),
                    "final_obs": np.asarray(
                        [self._state[i].copy() for i in range(self.num_envs)], dtype=object
                    ),
                    "_final_obs": done.copy(),
                }
            )
        return (
            self._state.copy(),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            done,
            info,
        )

    def state(self) -> np.ndarray:
        return self._state.copy()

    def environment_contract(self) -> dict[str, Any]:
        return {"schema_version": 14, "sha256": "fake"}

    def close(self, **kwargs: Any) -> None:
        del kwargs
        self.close_calls += 1


class ConfiguredFakeMoyaVectorEnv(FakeMoyaVectorEnv):
    observed_environment: dict[str, str | None] = {}
    contract_sha256 = EXPECTED_MOYA_CONTRACT_SHA256
    last_instance: ConfiguredFakeMoyaVectorEnv | None = None

    def __init__(self, **kwargs: Any) -> None:
        assert kwargs == {
            "num_envs": 2,
            "headless": True,
            "episode_length": 930,
            "device": "cpu",
            "sim_substeps": 8,
        }
        type(self).observed_environment = {
            name: os.environ.get(name) for name in MOYA_PRESETS["randomized_grasp_v1"]
        }
        type(self).observed_environment["MOYA_UNRELATED"] = os.environ.get("MOYA_UNRELATED")
        super().__init__()
        type(self).last_instance = self

    def environment_contract(self) -> dict[str, Any]:
        return {"schema_version": 14, "sha256": self.contract_sha256}


def make_env(**backend_kwargs: Any) -> tuple[MoyaNewtonVectorEnv, FakeMoyaVectorEnv]:
    backend = FakeMoyaVectorEnv(**backend_kwargs)
    env = MoyaNewtonVectorEnv(
        backend,
        episode_length=930,
        task="randomized_grasp_charger",
        task_description="grasp and lift randomized charger",
        success_min_final_lift_height=0.015,
    )
    return env, backend


def test_reset_wraps_agent_pos_and_folds_seed_list() -> None:
    env, backend = make_env()

    observation, info = env.reset(seed=[11, 12])
    first_seed = backend.last_seed
    env.reset(seed=[11, 12])

    assert observation["agent_pos"].shape == (2, 39)
    assert observation["agent_pos"].dtype == np.float32
    assert info["reset"].tolist() == [True, True]
    assert isinstance(first_seed, int)
    assert backend.last_seed == first_seed
    assert preprocess_observation(observation)["observation.state"].shape == (2, 39)


def test_reset_rejects_wrong_seed_count() -> None:
    env, _ = make_env()

    with pytest.raises(ValueError, match="2 seeds"):
        env.reset(seed=[11])


def test_action_shape_and_values_are_checked_without_rescaling() -> None:
    env, backend = make_env()
    env.reset(seed=0)
    action = np.linspace(-0.5, 0.5, 28, dtype=np.float64).reshape(2, 14)

    env.step(action)

    assert backend.last_action is not None
    assert backend.last_action.dtype == np.float32
    np.testing.assert_allclose(backend.last_action, action.astype(np.float32))

    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros((1, 14), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full((2, 14), np.nan, dtype=np.float32))


def test_final_info_is_dict_of_arrays_and_15mm_is_success() -> None:
    env, _ = make_env()
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.zeros((2, 14), dtype=np.float32))

    final_info = info["final_info"]
    assert isinstance(final_info, dict)
    assert final_info["is_success"].tolist() == [True, True]
    assert final_info["true_grasp_ever"].tolist() == [True, True]
    assert final_info["clear_table_ever"].tolist() == [True, True]
    assert final_info["native_success"].tolist() == [True, False]
    assert final_info["reward_components"]["clear_table"].shape == (2,)
    assert info["is_success"].tolist() == [True, True]
    assert info["_final_info"].tolist() == [True, True]


@pytest.mark.parametrize(
    ("backend_kwargs", "expected"),
    [
        ({"final_lift_height": 0.014999}, False),
        ({"final_table_contacts": 1}, False),
        ({"final_hand_contacts": 0}, False),
        ({"true_grasp": False}, False),
        ({"clear_table": False}, False),
        ({}, True),
    ],
)
def test_success_requires_every_dataset_condition(
    backend_kwargs: dict[str, Any], expected: bool
) -> None:
    env, _ = make_env(**backend_kwargs)
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.zeros((2, 14), dtype=np.float32))

    assert info["final_info"]["is_success"].tolist() == [expected, expected]


def test_grasp_and_clear_history_persist_until_terminal_step() -> None:
    env, backend = make_env(done_after_steps=2)
    env.reset(seed=0)

    _, _, _, truncated, info = env.step(np.zeros((2, 14), dtype=np.float32))
    assert truncated.tolist() == [False, False]
    assert "final_info" not in info

    backend.true_grasp = False
    backend.clear_table = False
    _, _, _, truncated, info = env.step(np.zeros((2, 14), dtype=np.float32))

    assert truncated.tolist() == [True, True]
    assert info["final_info"]["true_grasp_ever"].tolist() == [True, True]
    assert info["final_info"]["clear_table_ever"].tolist() == [True, True]
    assert info["final_info"]["is_success"].tolist() == [True, True]


def test_call_get_attr_state_contract_and_close_are_vector_compatible() -> None:
    env, backend = make_env()
    env.reset(seed=0)

    assert env.call("_max_episode_steps") == (930, 930)
    assert env.call("task_description") == (
        "grasp and lift randomized charger",
        "grasp and lift randomized charger",
    )
    assert env.get_attr("task") == ("randomized_grasp_charger",) * 2
    assert env.state()["agent_pos"].shape == (2, 39)
    assert env.environment_contract()["sha256"] == "fake"
    with pytest.raises(NotImplementedError, match="Rendering is disabled"):
        env.render()

    env.close()
    env.close()
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "observation",
    [
        np.zeros((2, 38), dtype=np.float32),
        np.zeros((2, 39), dtype=np.float64),
        np.full((2, 39), np.inf, dtype=np.float32),
    ],
)
def test_reset_rejects_invalid_observation(observation: np.ndarray) -> None:
    env, backend = make_env()
    backend.reset = lambda **kwargs: (observation, {})  # type: ignore[method-assign]

    with pytest.raises((TypeError, ValueError)):
        env.reset(seed=0)


def test_factory_applies_and_restores_dataset_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOYA_IK_ITERS", "999")
    monkeypatch.setenv("MOYA_UNRELATED", "ambient")
    model_module = SimpleNamespace(CHARGER_MASS=0.5)

    env = create_moya_newton_env(
        num_envs=2,
        episode_length=930,
        device="cpu",
        headless=True,
        sim_substeps=8,
        preset="randomized_grasp_v1",
        task="randomized_grasp_charger",
        task_description="grasp and lift randomized charger",
        success_min_final_lift_height=0.015,
        _backend_loader=lambda: (ConfiguredFakeMoyaVectorEnv, model_module),
    )

    assert isinstance(env, MoyaNewtonVectorEnv)
    assert ConfiguredFakeMoyaVectorEnv.observed_environment == {
        **MOYA_PRESETS["randomized_grasp_v1"],
        "MOYA_UNRELATED": None,
    }
    assert os.environ["MOYA_IK_ITERS"] == "999"
    assert os.environ["MOYA_UNRELATED"] == "ambient"
    env.close()


def test_factory_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown Moya preset"):
        create_moya_newton_env(
            num_envs=2,
            episode_length=930,
            device="cpu",
            headless=True,
            sim_substeps=8,
            preset="unknown",
            task="task",
            task_description="task",
            success_min_final_lift_height=0.015,
            _backend_loader=lambda: (ConfiguredFakeMoyaVectorEnv, SimpleNamespace(CHARGER_MASS=0.5)),
        )


def test_factory_closes_backend_on_contract_mismatch() -> None:
    class BadContractEnv(ConfiguredFakeMoyaVectorEnv):
        contract_sha256 = "wrong"

    with pytest.raises(RuntimeError, match="contract SHA-256"):
        create_moya_newton_env(
            num_envs=2,
            episode_length=930,
            device="cpu",
            headless=True,
            sim_substeps=8,
            preset="randomized_grasp_v1",
            task="task",
            task_description="task",
            success_min_final_lift_height=0.015,
            _backend_loader=lambda: (BadContractEnv, SimpleNamespace(CHARGER_MASS=0.5)),
        )

    assert BadContractEnv.last_instance is not None
    assert BadContractEnv.last_instance.close_calls == 1


def test_factory_rejects_wrong_charger_mass() -> None:
    with pytest.raises(RuntimeError, match="charger mass"):
        create_moya_newton_env(
            num_envs=2,
            episode_length=930,
            device="cpu",
            headless=True,
            sim_substeps=8,
            preset="randomized_grasp_v1",
            task="task",
            task_description="task",
            success_min_final_lift_height=0.015,
            _backend_loader=lambda: (
                ConfiguredFakeMoyaVectorEnv,
                SimpleNamespace(CHARGER_MASS=1.0),
            ),
        )


def test_missing_submodule_error_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git submodule update --init --recursive"):
        create_moya_newton_env(
            num_envs=1,
            episode_length=1,
            device="cpu",
            headless=True,
            sim_substeps=8,
            preset="randomized_grasp_v1",
            task="task",
            task_description="task",
            success_min_final_lift_height=0.015,
            submodule_root=tmp_path,
        )


def test_missing_dependency_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lerobot.envs.moya_newton as moya_newton

    (tmp_path / "moya_batched_env.py").touch()
    (tmp_path / "moya_model.py").touch()
    (tmp_path / "rewards").mkdir()
    (tmp_path / "rewards" / "reward_api.py").touch()

    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError("No module named 'newton'", name="newton")

    monkeypatch.setattr(moya_newton.importlib, "import_module", missing_dependency)
    with pytest.raises(RuntimeError, match="uv sync --extra moya_newton"):
        create_moya_newton_env(
            num_envs=1,
            episode_length=1,
            device="cpu",
            headless=True,
            sim_substeps=8,
            preset="randomized_grasp_v1",
            task="task",
            task_description="task",
            success_min_final_lift_height=0.015,
            submodule_root=tmp_path,
        )


def test_environment_config_is_registered_without_importing_newton() -> None:
    from lerobot.envs import MoyaNewtonEnvConfig, make_env_config

    was_loaded = "newton" in sys.modules
    cfg = make_env_config("moya_newton")

    assert isinstance(cfg, MoyaNewtonEnvConfig)
    assert cfg.features["agent_pos"].shape == (39,)
    assert cfg.features["action"].shape == (14,)
    assert cfg.features_map["agent_pos"] == "observation.state"
    assert cfg.features_map["action"] == "action"
    assert ("newton" in sys.modules) is was_loaded


def test_config_create_envs_uses_native_fused_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.envs import MoyaNewtonEnvConfig

    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr("lerobot.envs.moya_newton.create_moya_newton_env", fake_create)
    cfg = MoyaNewtonEnvConfig(device="cpu")

    result = cfg.create_envs(n_envs=2, use_async_envs=True)

    assert result == {"moya_newton": {0: sentinel}}
    assert calls == [
        {
            "num_envs": 2,
            "episode_length": 930,
            "device": "cpu",
            "headless": True,
            "sim_substeps": 8,
            "preset": "randomized_grasp_v1",
            "task": "randomized_grasp_charger",
            "task_description": "grasp and lift randomized charger",
            "success_min_final_lift_height": 0.015,
        }
    ]
