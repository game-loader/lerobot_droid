# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the Moya SAME_STEP adapter (no Newton runtime required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from RL.adapters.moya_newton import (
    create_moya_env,
    extract_state_observation,
    extract_terminal_success,
    select_transition_next_state,
)


def _states(first: float = 9.0, second: float = 8.0) -> np.ndarray:
    value = np.full((2, 39), first, dtype=np.float32)
    value[1] = second
    return value


def _final_states(first: float = 3.0, second: float = 4.0) -> np.ndarray:
    value = np.full((2, 39), first, dtype=np.float32)
    value[1] = second
    return value


def test_terminal_next_state_uses_final_obs_and_keeps_non_done_row() -> None:
    selected = select_transition_next_state(
        _states(),
        {"final_obs": _final_states()},
        np.array([True, False]),
    )
    assert selected.shape == (2, 39)
    assert np.all(selected[0] == 3.0)
    assert np.all(selected[1] == 8.0)


def test_final_observation_alias_and_explicit_mask() -> None:
    selected = select_transition_next_state(
        {"agent_pos": _states()},
        {
            "final_observation": {"agent_pos": _final_states()},
            "_final_observation": np.array([True, False]),
        },
        np.array([True, False]),
    )
    np.testing.assert_allclose(selected[0], 3.0)
    np.testing.assert_allclose(selected[1], 8.0)


def test_collated_dict_of_arrays_final_observation() -> None:
    final = _final_states()
    selected = select_transition_next_state(
        _states(),
        {
            "final_obs": {"agent_pos": final},
            "_final_obs": np.array([True, False]),
        },
        np.array([True, False]),
    )
    np.testing.assert_allclose(selected[0], final[0])
    np.testing.assert_allclose(selected[1], _states()[1])


def test_per_environment_dict_object_array_final_observation() -> None:
    final = np.empty(2, dtype=object)
    final[0] = {"agent_pos": np.full(39, 3.0, dtype=np.float32)}
    final[1] = {"agent_pos": np.full(39, 4.0, dtype=np.float32)}
    selected = select_transition_next_state(
        _states(),
        {"final_obs": final},
        np.array([True, False]),
    )
    np.testing.assert_allclose(selected[0], 3.0)
    np.testing.assert_allclose(selected[1], 8.0)


def test_per_environment_tuple_is_data_not_an_alias_marker() -> None:
    final = (
        np.full(39, 3.0, dtype=np.float32),
        np.full(39, 4.0, dtype=np.float32),
    )
    selected = select_transition_next_state(
        _states(),
        {"final_obs": final, "_final_obs": (True, False)},
        np.array([True, False]),
    )
    np.testing.assert_allclose(selected[0], 3.0)
    np.testing.assert_allclose(selected[1], 8.0)


def test_missing_final_state_for_done_row_is_rejected() -> None:
    final = np.empty(2, dtype=object)
    final[0] = None
    final[1] = np.full(39, 4.0, dtype=np.float32)
    with pytest.raises(ValueError, match="missing final observation"):
        select_transition_next_state(_states(), {"final_obs": final}, np.array([True, False]))


def test_final_mask_must_match_done() -> None:
    with pytest.raises(ValueError, match="mask does not match done"):
        select_transition_next_state(
            _states(),
            {"final_obs": _final_states(), "_final_obs": np.array([False, False])},
            np.array([True, False]),
        )


@pytest.mark.parametrize(
    "bad_final",
    [
        np.zeros((2, 38), dtype=np.float32),
        np.array([[np.nan] * 39, [0.0] * 39], dtype=np.float32),
    ],
)
def test_final_state_shape_and_finite_values_are_validated(bad_final: np.ndarray) -> None:
    with pytest.raises(ValueError):
        select_transition_next_state(_states(), {"final_obs": bad_final}, np.array([True, False]))


def test_no_final_payload_is_allowed_when_no_world_is_done() -> None:
    selected = select_transition_next_state(_states(), {}, np.array([False, False]))
    np.testing.assert_allclose(selected, _states())


def test_extract_state_observation_accepts_lerobot_mapping() -> None:
    selected = extract_state_observation({"agent_pos": _states()})
    assert selected.dtype == np.float32
    np.testing.assert_allclose(selected, _states())


def test_terminal_success_uses_collated_final_info_not_top_level_info() -> None:
    success = extract_terminal_success(
        {
            "is_success": np.array([False, True]),
            "final_info": {"is_success": np.array([True, False])},
            "_final_info": np.array([True, False]),
        },
        np.array([True, False]),
    )
    assert success.tolist() == [True, False]


def test_terminal_success_accepts_per_environment_final_info() -> None:
    final_info = np.empty(2, dtype=object)
    final_info[0] = {"is_success": False}
    final_info[1] = {"is_success": True}
    success = extract_terminal_success(
        {"final_info": final_info, "_final_info": np.array([False, True])},
        np.array([False, True]),
    )
    assert success.tolist() == [False, True]


def test_terminal_success_requires_matching_final_info_mask() -> None:
    with pytest.raises(ValueError, match="final info mask does not match done"):
        extract_terminal_success(
            {
                "final_info": {"is_success": np.array([True, False])},
                "_final_info": np.array([False, False]),
            },
            np.array([True, False]),
        )


def test_create_moya_env_uses_sync_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, int, bool]] = []

    class FakeConfig:
        type = "moya_newton"

    class FakeEnv:
        def reset(self):  # pragma: no cover - only contract check
            return None, {}

        def step(self, action):  # pragma: no cover - only contract check
            del action
            return None, None, None, None, {}

    fake_env = FakeEnv()
    fake_module = SimpleNamespace(
        MoyaNewtonEnvConfig=FakeConfig,
        make_env=lambda cfg, n_envs, use_async_envs: (
            calls.append((cfg, n_envs, use_async_envs)) or {"moya_newton": {0: fake_env}}
        ),
    )

    # ``create_moya_env`` imports from lerobot.envs lazily; patch the module
    # attributes rather than importing any Newton backend.
    import lerobot.envs

    monkeypatch.setattr(lerobot.envs, "MoyaNewtonEnvConfig", FakeConfig)
    monkeypatch.setattr(lerobot.envs, "make_env", fake_module.make_env)
    assert create_moya_env(num_envs=2) is fake_env
    assert calls == [(calls[0][0], 2, False)]


def test_create_moya_env_rejects_changed_success_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        type = "moya_newton"

        def __init__(self, threshold: float = 0.015) -> None:
            self.success_min_final_lift_height = threshold

    import lerobot.envs

    monkeypatch.setattr(lerobot.envs, "MoyaNewtonEnvConfig", FakeConfig)
    monkeypatch.setattr(
        lerobot.envs,
        "make_env",
        lambda *_args, **_kwargs: pytest.fail("make_env must not run for an invalid threshold"),
    )
    with pytest.raises(ValueError, match="0.015"):
        create_moya_env(config=FakeConfig(0.01), num_envs=2)
