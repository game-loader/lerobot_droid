# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the pure Moya sparse-episode collector state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from RL.adapters.lerobot_v3 import (
    SUCCESS_MIN_FINAL_LIFT_HEIGHT as ADAPTER_LIFT_THRESHOLD,
    LeRobotV3DecisionDataset,
    terminal_success,
)
from RL.collectors.moya_il import (
    CollectedEpisode,
    EpisodeBatch,
    StepDiagnostics,
    batch_seeds,
    collect_rollouts,
    extract_step_diagnostics,
    publish_collection,
    recording_mask,
)
from RL.config import RLConfig


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
    assert terminal_success(episode.metadata)
    assert pytest.approx(
        float(np.float32(0.015)), abs=0.0
    ) == ADAPTER_LIFT_THRESHOLD


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


def test_missing_sparse_reward_components_are_zero() -> None:
    info = _collated_info()
    info["reward_components"] = {}
    result = extract_step_diagnostics(
        info,
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        num_envs=2,
    )
    assert result.true_grasp.tolist() == [0.0, 0.0]
    assert result.clear_table.tolist() == [0.0, 0.0]


def test_missing_final_sparse_reward_components_replace_reset_values_with_zero() -> None:
    info = _collated_info()
    info.update(
        {
            "final_info": {
                "reward_components": {},
                "charger_lift_height": np.array([0.0, 0.0], dtype=np.float32),
                "charger_table_contacts": np.array([1, 0], dtype=np.int32),
                "right_hand_charger_contacts": np.array([0, 1], dtype=np.int32),
            },
            "_final_info": np.array([True, False], dtype=np.bool_),
        }
    )
    result = extract_step_diagnostics(
        info,
        terminated=np.array([False, False]),
        truncated=np.array([True, False]),
        num_envs=2,
    )
    assert result.true_grasp.tolist() == [0.0, 1.0]
    assert result.clear_table.tolist() == [0.0, 1.0]


def test_present_sparse_reward_component_remains_strict() -> None:
    info = _collated_info()
    info["reward_components"]["true_grasp"] = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        extract_step_diagnostics(
            info,
            terminated=np.array([False, False]),
            truncated=np.array([False, False]),
            num_envs=2,
        )


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


def test_collected_episode_requires_canonical_terminal_tuple() -> None:
    kwargs = {
        "states": states(0.0, num_envs=1),
        "actions": actions(0.0, num_envs=1),
        "dones": np.array([[True]], dtype=np.bool_),
        "metadata": {"success": False},
    }
    with pytest.raises(ValueError, match="failed terminal transition"):
        CollectedEpisode(
            rewards=np.array([[0.0]], dtype=np.float32),
            truncated=np.array([[False]], dtype=np.bool_),
            **kwargs,
        )
    with pytest.raises(ValueError, match="successful terminal transition"):
        CollectedEpisode(
            rewards=np.array([[1.0]], dtype=np.float32),
            truncated=np.array([[True]], dtype=np.bool_),
            **{**kwargs, "metadata": {"success": True}},
        )


class _FakeRolloutPolicy:
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls = 0
        self.returned_actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, raw_states: np.ndarray) -> np.ndarray:
        self.calls += 1
        action = np.full((raw_states.shape[0], 14), self.calls, dtype=np.float32)
        self.returned_actions.append(action.copy())
        return action


class _FakeRolloutEnv:
    num_envs = 4

    def __init__(self) -> None:
        self.step_count = 0
        self.executed_actions: list[np.ndarray] = []
        self.seeds: list[int] | int | None = None

    def reset(self, *, seed: list[int] | int | None = None):
        self.step_count = 0
        self.seeds = seed
        return np.zeros((self.num_envs, 39), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.executed_actions.append(action.copy())
        lift = np.zeros(self.num_envs, dtype=np.float32)
        lift[:2] = 0.015 if self.step_count >= 2 else 0.0
        lift[2] = 0.015 if self.step_count >= 3 else 0.0
        info = {
            "reward_components": {
                "true_grasp": np.ones(self.num_envs, dtype=np.float32),
                "clear_table": np.ones(self.num_envs, dtype=np.float32),
            },
            "charger_lift_height": lift,
            "charger_table_contacts": np.zeros(self.num_envs, dtype=np.int32),
            "right_hand_charger_contacts": np.ones(self.num_envs, dtype=np.int32),
        }
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        return (
            np.full((self.num_envs, 39), self.step_count, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.float32),
            terminated,
            truncated,
            info,
        )


def test_collect_rollouts_keeps_tail_order_and_executed_actions() -> None:
    env = _FakeRolloutEnv()
    policy = _FakeRolloutPolicy()
    result = collect_rollouts(
        env,
        policy,
        target_episodes=3,
        episode_length=3,
        base_seed=100,
    )
    assert [episode.metadata["episode_index"] for episode in result.episodes] == [0, 1, 2]
    assert [episode.metadata["world_index"] for episode in result.episodes] == [0, 1, 2]
    assert policy.reset_count == 1
    assert len(env.executed_actions) == policy.calls
    for episode in result.episodes:
        for row in episode.actions:
            assert any(np.array_equal(row, action[0]) for action in env.executed_actions)
    assert all(episode.metadata["success"] for episode in result.episodes)


def _manual_episode(*, success: bool) -> CollectedEpisode:
    return CollectedEpisode(
        states=np.zeros((2, 39), dtype=np.float32),
        actions=np.zeros((2, 14), dtype=np.float32),
        rewards=np.array([[0.0], [1.0 if success else 0.0]], dtype=np.float32),
        dones=np.array([[False], [True]], dtype=np.bool_),
        truncated=np.array([[False], [not success]], dtype=np.bool_),
        metadata={
            "episode_index": 0 if success else 1,
            "true_grasp_ever": success,
            "clear_table_ever": success,
            "final_lift_height_m": float(np.float32(0.015)) if success else 0.0,
            "final_table_contacts": 0,
            "final_hand_contacts": 1 if success else 0,
            "success": success,
            "terminal_reason": "success" if success else "horizon",
        },
    )


def test_publish_collection_round_trips_canonical_v3_without_video(tmp_path: Path) -> None:
    output = tmp_path / "collection"
    success = _manual_episode(success=True)
    failure = _manual_episode(success=False)
    published = publish_collection(
        output,
        repo_id="local/moya-il-test",
        episodes=(success, failure),
        summary={"checkpoint_model_sha256": "a" * 64},
        fps=60,
    )
    assert published == output
    assert (output / "dataset" / "meta" / "info.json").is_file()
    payload = json.loads((output / "collection_summary.json").read_text())
    assert payload["complete"] is True
    assert payload["episodes_saved"] == 2
    assert terminal_success(payload["episodes"][0])
    assert not (output / "dataset" / "videos").exists()
    assert not (output / "dataset" / "images").exists()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(
        "local/moya-il-test", root=output / "dataset", download_videos=False
    )
    assert loaded.meta.info.codebase_version == "v3.0"
    assert loaded.num_episodes == 2
    assert tuple(loaded.features["observation.state"]["shape"]) == (39,)
    assert tuple(loaded.features["action"]["shape"]) == (14,)
    assert tuple(loaded.features["next.reward"]["shape"]) == (1,)
    row = loaded.get_raw_item(1)
    assert np.asarray(row["next.reward"]).reshape(-1).tolist() == [1.0]
    assert np.asarray(row["next.done"]).reshape(-1).tolist() == [True]
    assert np.asarray(row["next.truncated"]).reshape(-1).tolist() == [False]

    decisions = LeRobotV3DecisionDataset.from_root(
        dataset_root=output / "dataset",
        repo_id="local/moya-il-test",
        summary_path=output / "collection_summary.json",
        config=RLConfig(state_dim=39, action_dim=14, chunk_size=2),
    )
    assert len(decisions) == 2
    assert [decisions[index].reward.item() for index in range(len(decisions))] == [1.0, 0.0]


def test_publish_failure_leaves_only_incomplete_staging(tmp_path: Path, monkeypatch) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def fail_create(*_args, **_kwargs):
        raise RuntimeError("writer failure")

    monkeypatch.setattr(LeRobotDataset, "create", fail_create)
    output = tmp_path / "collection"
    with pytest.raises(RuntimeError, match="writer failure"):
        publish_collection(
            output,
            repo_id="local/moya-il-failure",
            episodes=(_manual_episode(success=True),),
            summary={},
        )
    assert not output.exists()
    staging = list(tmp_path.glob("collection.incomplete-*"))
    assert len(staging) == 1
    payload = json.loads((staging[0] / "collection_summary.json").read_text())
    assert payload["complete"] is False
