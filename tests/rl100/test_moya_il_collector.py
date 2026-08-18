# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the pure Moya sparse-episode collector state machine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from RL.collectors.moya_il import (
    EpisodeBatch,
    StepDiagnostics,
    batch_seeds,
    extract_step_diagnostics,
    recording_mask,
)


def states(*values: float, num_envs: int | None = None) -> np.ndarray:
    if num_envs is None:
        num_envs = len(values)
    if not values:
        values = (0.0,) * num_envs
    return np.stack(
        [np.full(39, value, dtype=np.float32) for value in values], axis=0
    ).astype(np.float32)


def actions(*values: float, num_envs: int | None = None) -> np.ndarray:
    if num_envs is None:
        num_envs = len(values)
    if not values:
        values = (0.0,) * num_envs
    return np.stack(
        [np.full(14, value, dtype=np.float32) for value in values], axis=0
    ).astype(np.float32)


def diagnostics(
    *,
    true_grasp: tuple[float, ...] = (1.0, 1.0),
    clear_table: tuple[float, ...] = (1.0, 1.0),
    lift: tuple[float, ...] = (0.0, 0.0),
    table_contacts: tuple[float, ...] = (0.0, 0.0),
    hand_contacts: tuple[float, ...] = (1.0, 1.0),
    native_done: tuple[bool, ...] = (False, False),
) -> StepDiagnostics:
    return StepDiagnostics(
        true_grasp=np.asarray(true_grasp, dtype=np.float32),
        clear_table=np.asarray(clear_table, dtype=np.float32),
        lift_height=np.asarray(lift, dtype=np.float32),
        table_contacts=np.asarray(table_contacts, dtype=np.float32),
        hand_contacts=np.asarray(hand_contacts, dtype=np.float32),
        native_done=np.asarray(native_done, dtype=np.bool_),
    )


def test_first_acceptance_step_freezes_episode() -> None:
    batch = EpisodeBatch.create(np.array([True, True]))
    batch.append_step(states(1.0, 1.0), actions(1.0, 1.0), diagnostics(lift=(0.014, 0.0)))
    batch.append_step(states(2.0, 2.0), actions(2.0, 2.0), diagnostics(lift=(0.015, 0.0)))
    batch.append_step(
        states(3.0, 3.0),
        actions(3.0, 3.0),
        diagnostics(lift=(0.020, 0.0), native_done=(False, True)),
    )

    episode = batch.finalized()[0]
    assert episode.states.shape == (2, 39)
    assert episode.actions.shape == (2, 14)
    assert episode.rewards[:, 0].tolist() == [0.0, 1.0]
    assert episode.dones[:, 0].tolist() == [False, True]
    assert episode.truncated[:, 0].tolist() == [False, False]
    assert episode.metadata["success"] is True
    assert episode.metadata["terminal_reason"] == "success"

    failed = batch.finalized()[1]
    assert failed.rewards[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert failed.dones[:, 0].tolist() == [False, False, True]
    assert failed.truncated[:, 0].tolist() == [False, False, True]
    assert batch.complete()
    assert batch.is_complete


def test_success_requires_all_five_conditions() -> None:
    cases: list[dict[str, tuple[float, ...]]] = [
        {"true_grasp": (0.0, 0.0)},
        {"clear_table": (0.0, 0.0)},
        {"lift": (0.014999, 0.014999)},
        {"table_contacts": (1.0, 1.0)},
        {"hand_contacts": (0.0, 0.0)},
    ]
    for overrides in cases:
        batch = EpisodeBatch.create(np.array([True, True]))
        kwargs: dict[str, Any] = {"lift": (0.015, 0.015)}
        kwargs.update(overrides)
        batch.append_step(
            states(0.0, 0.0),
            actions(0.0, 0.0),
            diagnostics(**kwargs, native_done=(True, True)),
        )
        assert all(not episode.metadata["success"] for episode in batch.finalized())
        assert all(episode.rewards[-1, 0] == 0.0 for episode in batch.finalized())


def test_exact_fifteen_millimetres_is_success() -> None:
    batch = EpisodeBatch.create(np.array([True]))
    batch.append_step(
        states(0.0, num_envs=1),
        actions(0.0, num_envs=1),
        diagnostics(
            true_grasp=(1.0,),
            clear_table=(1.0,),
            lift=(0.015,),
            table_contacts=(0.0,),
            hand_contacts=(1.0,),
            native_done=(False,),
        ),
    )
    episode = batch.finalized()[0]
    assert episode.metadata["success"] is True
    assert episode.rewards[-1, 0] == 1.0
    assert not episode.truncated[-1, 0]


def test_horizon_success_takes_precedence_over_truncation() -> None:
    batch = EpisodeBatch.create(np.array([True]))
    batch.append_step(
        states(0.0, num_envs=1),
        actions(0.0, num_envs=1),
        diagnostics(
            true_grasp=(1.0,),
            clear_table=(1.0,),
            lift=(0.015,),
            table_contacts=(0.0,),
            hand_contacts=(1.0,),
            native_done=(True,),
        ),
    )
    episode = batch.finalized()[0]
    assert episode.rewards.tolist() == [[1.0]]
    assert episode.dones.tolist() == [[True]]
    assert episode.truncated.tolist() == [[False]]
    assert episode.metadata["terminal_reason"] == "success"


def test_native_success_flag_is_not_part_of_acceptance_rule() -> None:
    batch = EpisodeBatch.create(np.array([True]))
    batch.append_step(
        states(0.0, num_envs=1),
        actions(0.0, num_envs=1),
        diagnostics(
            true_grasp=(1.0,),
            clear_table=(1.0,),
            lift=(0.015,),
            table_contacts=(0.0,),
            hand_contacts=(1.0,),
            native_done=(False,),
        ),
    )
    assert batch.finalized()[0].metadata["success"] is True


def test_recording_mask_is_a_low_index_tail_mask() -> None:
    np.testing.assert_array_equal(
        recording_mask(4, 16),
        np.array([True, True, True, True] + [False] * 12, dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        recording_mask(20, 16), np.ones(16, dtype=np.bool_)
    )
    with pytest.raises(ValueError):
        recording_mask(0, 16)
    with pytest.raises(ValueError):
        recording_mask(1, 0)
    with pytest.raises(ValueError, match="select at least one"):
        EpisodeBatch.create(np.array([False, False]))


def test_batch_seeds_are_deterministic_and_tail_independent() -> None:
    assert batch_seeds(1000, 0, 16) == (
        list(range(1000, 1016)),
        1_001_000,
    )
    assert batch_seeds(1000, 2, 16) == (
        list(range(1032, 1048)),
        1_001_002,
    )


def _collated_info(num_envs: int = 2) -> dict[str, Any]:
    return {
        "reward_components": {
            "true_grasp": np.ones(num_envs, dtype=np.float32),
            "clear_table": np.ones(num_envs, dtype=np.float32),
        },
        "charger_lift_height": np.zeros(num_envs, dtype=np.float32),
        "charger_table_contacts": np.zeros(num_envs, dtype=np.int32),
        "right_hand_charger_contacts": np.ones(num_envs, dtype=np.int32),
    }


def test_extract_diagnostics_uses_same_step_final_info_for_done_rows() -> None:
    info = _collated_info()
    info.update(
        {
            "final_info": {
                "reward_components": {
                    "true_grasp": np.array([100.0, 0.0], dtype=np.float32),
                    "clear_table": np.array([100.0, 0.0], dtype=np.float32),
                },
                "charger_lift_height": np.array([0.015, 0.0], dtype=np.float32),
                "charger_table_contacts": np.array([0, 1], dtype=np.int32),
                "right_hand_charger_contacts": np.array([1, 0], dtype=np.int32),
            },
            "_final_info": np.array([True, False], dtype=np.bool_),
        }
    )
    diagnostics_result = extract_step_diagnostics(
        info,
        terminated=np.array([False, False]),
        truncated=np.array([True, False]),
        num_envs=2,
    )
    assert diagnostics_result.native_done.tolist() == [True, False]
    assert diagnostics_result.true_grasp.tolist() == [100.0, 1.0]
    assert diagnostics_result.clear_table.tolist() == [100.0, 1.0]
    assert diagnostics_result.lift_height.tolist() == [0.014999999664723873, 0.0]
    assert diagnostics_result.table_contacts.tolist() == [0.0, 0.0]
    assert diagnostics_result.hand_contacts.tolist() == [1.0, 1.0]


def test_extract_diagnostics_accepts_per_environment_final_info() -> None:
    info = _collated_info()
    final_info = np.empty(2, dtype=object)
    final_info[0] = {
        "reward_components": {"true_grasp": 100.0, "clear_table": 100.0},
        "charger_lift_height": 0.015,
        "charger_table_contacts": 0,
        "right_hand_charger_contacts": 1,
    }
    final_info[1] = None
    info.update({"final_info": final_info, "_final_info": np.array([True, False])})
    result = extract_step_diagnostics(
        info,
        terminated=np.array([False, False]),
        truncated=np.array([True, False]),
        num_envs=2,
    )
    assert result.true_grasp.tolist() == [100.0, 1.0]
    assert result.lift_height[0] == np.float32(0.015)


def test_final_info_mask_must_match_native_done() -> None:
    info = _collated_info()
    info.update(
        {
            "final_info": {
                "reward_components": {
                    "true_grasp": np.ones(2, dtype=np.float32),
                    "clear_table": np.ones(2, dtype=np.float32),
                },
                "charger_lift_height": np.zeros(2, dtype=np.float32),
                "charger_table_contacts": np.zeros(2, dtype=np.int32),
                "right_hand_charger_contacts": np.ones(2, dtype=np.int32),
            },
            "_final_info": np.array([False, False], dtype=np.bool_),
        }
    )
    with pytest.raises(ValueError, match="final info mask"):
        extract_step_diagnostics(
            info,
            terminated=np.array([False, False]),
            truncated=np.array([True, False]),
            num_envs=2,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "charger_lift_height",
        "charger_table_contacts",
        "right_hand_charger_contacts",
    ],
)
def test_missing_component_keys_are_rejected(missing: str) -> None:
    info = _collated_info()
    del info[missing]
    with pytest.raises(ValueError, match="missing"):
        extract_step_diagnostics(
            info,
            terminated=np.array([False, False]),
            truncated=np.array([False, False]),
            num_envs=2,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.zeros((2, 1), dtype=np.float32),
        np.array([np.nan, 0.0], dtype=np.float32),
    ],
)
def test_malformed_or_nonfinite_diagnostics_are_rejected(bad_value: np.ndarray) -> None:
    info = _collated_info()
    info["charger_lift_height"] = bad_value
    with pytest.raises(ValueError):
        extract_step_diagnostics(
            info,
            terminated=np.array([False, False]),
            truncated=np.array([False, False]),
            num_envs=2,
        )


def test_state_and_action_contracts_are_validated() -> None:
    batch = EpisodeBatch.create(np.array([True]))
    with pytest.raises(ValueError, match="state"):
        batch.append_step(
            np.zeros((1, 38), dtype=np.float32),
            actions(0.0, num_envs=1),
            diagnostics(
                true_grasp=(1.0,),
                clear_table=(1.0,),
                lift=(0.0,),
                table_contacts=(0.0,),
                hand_contacts=(1.0,),
                native_done=(False,),
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        batch.append_step(
            np.full((1, 39), np.nan, dtype=np.float32),
            actions(0.0, num_envs=1),
            diagnostics(
                true_grasp=(1.0,),
                clear_table=(1.0,),
                lift=(0.0,),
                table_contacts=(0.0,),
                hand_contacts=(1.0,),
                native_done=(False,),
            ),
        )
