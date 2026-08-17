# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from RL.adapters import lerobot_v3
from RL.adapters.lerobot_v3 import (
    LeRobotV3DecisionDataset,
    build_episode_decisions,
    load_episode_labels,
    terminal_success,
)
from RL.config import RLConfig


def _episode_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "episode_index": 0,
        "true_grasp_ever": True,
        "clear_table_ever": True,
        "final_lift_height_m": 0.015,
        "final_table_contacts": 0,
        "final_hand_contacts": 1,
    }
    metadata.update(overrides)
    return metadata


def _fake_episode(
    *,
    length: int,
    state_dim: int = 39,
    action_dim: int = 14,
    include_image: bool = False,
) -> dict[str, torch.Tensor]:
    episode = {
        "observation.state": torch.arange(length * state_dim, dtype=torch.float32).reshape(
            length, state_dim
        ),
        "action": torch.arange(length * action_dim, dtype=torch.float32).reshape(
            length, action_dim
        ),
        "frame_index": torch.arange(length),
        "episode_index": torch.zeros(length, dtype=torch.int64),
    }
    if include_image:
        episode["observation.images.front"] = torch.arange(
            length * 4, dtype=torch.uint8
        ).reshape(length, 1, 2, 2)
    return episode


def test_terminal_success_uses_fifteen_millimeters() -> None:
    assert terminal_success(_episode_metadata(final_lift_height_m=0.015))
    assert not terminal_success(_episode_metadata(final_lift_height_m=0.014999))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("true_grasp_ever", False),
        ("clear_table_ever", False),
        ("final_table_contacts", 1),
        ("final_hand_contacts", 0),
    ],
)
def test_terminal_success_requires_all_acceptance_conditions(field: str, value: object) -> None:
    assert not terminal_success(_episode_metadata(**{field: value}))


def test_terminal_success_rejects_missing_acceptance_fields() -> None:
    metadata = _episode_metadata()
    metadata.pop("true_grasp_ever")

    with pytest.raises(ValueError, match="true_grasp_ever"):
        terminal_success(metadata)


@pytest.mark.parametrize(
    "overrides",
    [
        {"true_grasp_ever": "false"},
        {"clear_table_ever": 1},
        {"final_table_contacts": 0.5},
        {"final_hand_contacts": "1"},
    ],
)
def test_terminal_success_rejects_malformed_acceptance_types(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        terminal_success(_episode_metadata(**overrides))


def test_build_decisions_adds_only_terminal_sparse_reward() -> None:
    decisions = build_episode_decisions(
        _fake_episode(length=35),
        success=True,
        n_obs_steps=2,
        chunk_size=32,
        gamma=0.99,
    )

    assert len(decisions) == 2
    assert decisions[0].reward.item() == 0.0
    assert decisions[1].reward.item() == 1.0
    assert decisions[0].action_valid.sum().item() == 32
    assert decisions[1].action_valid.sum().item() == 3
    assert not decisions[0].done.item()
    assert decisions[1].done.item()
    assert decisions[0].discount.item() == pytest.approx(0.99**32)
    assert decisions[1].discount.item() == pytest.approx(0.99**3)


def test_failed_episode_keeps_terminal_reward_zero() -> None:
    decision = build_episode_decisions(
        _fake_episode(length=2),
        success=False,
        n_obs_steps=2,
        chunk_size=32,
        gamma=0.99,
    )[0]

    assert decision.done.item()
    assert decision.reward.item() == 0.0


def test_build_decisions_accepts_zero_discount() -> None:
    decision = build_episode_decisions(
        _fake_episode(length=2),
        success=False,
        n_obs_steps=2,
        chunk_size=2,
        gamma=0.0,
    )[0]

    assert decision.discount.item() == 0.0


def test_decision_history_and_action_padding() -> None:
    episode = _fake_episode(length=3, state_dim=2, action_dim=2)
    decisions = build_episode_decisions(
        episode,
        success=False,
        n_obs_steps=2,
        chunk_size=2,
        gamma=0.9,
    )

    torch.testing.assert_close(
        decisions[0].observation.features["observation.state"],
        episode["observation.state"][[0, 0]].unsqueeze(0),
    )
    torch.testing.assert_close(
        decisions[0].next_observation.features["observation.state"],
        episode["observation.state"][[1, 2]].unsqueeze(0),
    )
    torch.testing.assert_close(
        decisions[1].action,
        episode["action"][[2, 2]].unsqueeze(0),
    )
    assert decisions[1].action_valid.tolist() == [[True, False]]


def test_image_keys_survive_decision_construction() -> None:
    episode = _fake_episode(length=3, state_dim=2, action_dim=2, include_image=True)
    decision = build_episode_decisions(
        episode,
        success=False,
        n_obs_steps=2,
        chunk_size=2,
        gamma=0.99,
    )[0]

    assert "observation.images.front" in decision.observation.features
    torch.testing.assert_close(
        decision.observation.features["observation.images.front"],
        episode["observation.images.front"][[0, 0]].unsqueeze(0),
    )
    torch.testing.assert_close(
        decision.next_observation.features["observation.images.front"],
        episode["observation.images.front"][[1, 2]].unsqueeze(0),
    )


def test_build_decisions_supports_custom_state_key() -> None:
    episode = _fake_episode(length=2, state_dim=3, action_dim=2, include_image=True)
    episode["robot.state"] = episode.pop("observation.state")

    decision = build_episode_decisions(
        episode,
        success=False,
        n_obs_steps=2,
        chunk_size=2,
        gamma=0.99,
        state_key="robot.state",
    )[0]

    assert "robot.state" in decision.observation.features
    assert "observation.images.front" in decision.observation.features


@pytest.mark.parametrize("problem", ["nonfinite", "nonmonotonic", "mixed_episode"])
def test_build_decisions_rejects_invalid_episode_tensors(problem: str) -> None:
    episode = _fake_episode(length=3)
    if problem == "nonfinite":
        episode["action"][1, 0] = torch.nan
    elif problem == "nonmonotonic":
        episode["frame_index"] = torch.tensor([0, 2, 1])
    else:
        episode["episode_index"] = torch.tensor([0, 0, 1])

    with pytest.raises(ValueError, match={
        "nonfinite": "action",
        "nonmonotonic": "frame_index",
        "mixed_episode": "episode_index",
    }[problem]):
        build_episode_decisions(
            episode,
            success=True,
            n_obs_steps=2,
            chunk_size=2,
            gamma=0.99,
        )


def test_load_episode_labels_rejects_duplicates_and_index_mismatches(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"episodes": [_episode_metadata(), _episode_metadata()]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_episode_labels(summary_path, expected_episode_count=2)

    summary_path.write_text(
        json.dumps({"episodes": [_episode_metadata(episode_index=1)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="indices"):
        load_episode_labels(summary_path, expected_episode_count=1)

    summary_path.write_text(
        json.dumps({"episodes": [_episode_metadata(episode_index=0.5)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="episode_index"):
        load_episode_labels(summary_path, expected_episode_count=1)


class _FakeHFDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int | slice) -> dict[str, object]:
        if isinstance(index, int):
            return self.rows[index]
        selected = self.rows[index]
        keys = selected[0]
        batch: dict[str, object] = {}
        for key in keys:
            values = [row[key] for row in selected]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        return batch


class _FakeLeRobotDataset:
    last_instance: "_FakeLeRobotDataset | None" = None

    def __init__(self, repo_id: str, root: str | Path, **_: object) -> None:
        type(self).last_instance = self
        self.repo_id = repo_id
        self.root = Path(root)
        self.features = {
            "observation.state": {"shape": [3], "dtype": "float32"},
            "action": {"shape": [2], "dtype": "float32"},
            "frame_index": {"shape": [1], "dtype": "int64"},
            "episode_index": {"shape": [1], "dtype": "int64"},
        }
        lengths = [3, 2]
        rows: list[dict[str, object]] = []
        episode_rows: list[dict[str, int]] = []
        start = 0
        for episode_index, length in enumerate(lengths):
            for frame_index in range(length):
                rows.append(
                    {
                        "observation.state": torch.full((3,), float(frame_index)),
                        "action": torch.full((2,), float(frame_index)),
                        "frame_index": frame_index,
                        "episode_index": episode_index,
                    }
                )
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "length": length,
                    "dataset_from_index": start,
                    "dataset_to_index": start + length,
                }
            )
            start += length
        self.hf_dataset = _FakeHFDataset(rows)
        self.meta = SimpleNamespace(episodes=episode_rows)
        self.num_episodes = len(lengths)
        self.num_frames = sum(lengths)


class _FakeCameraLeRobotDataset(_FakeLeRobotDataset):
    last_instance: "_FakeCameraLeRobotDataset | None" = None

    def __init__(self, repo_id: str, root: str | Path, **kwargs: object) -> None:
        super().__init__(repo_id, root, **kwargs)
        type(self).last_instance = self
        self.features["observation.image"] = {"shape": [3, 2, 2], "dtype": "video"}
        self.meta.camera_keys = ["observation.image"]
        self.decoded_indices: list[int] = []

    def __getitem__(self, index: int) -> dict[str, object]:
        self.decoded_indices.append(index)
        row = dict(self.hf_dataset[index])
        row["observation.image"] = torch.full((3, 2, 2), index, dtype=torch.uint8)
        return row


def test_decision_dataset_reports_label_and_chunk_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "episodes": [
                    _episode_metadata(episode_index=0),
                    _episode_metadata(episode_index=1, true_grasp_ever=False),
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lerobot_v3, "LeRobotDataset", _FakeLeRobotDataset)

    dataset = LeRobotV3DecisionDataset.from_root(
        dataset_root=tmp_path / "dataset",
        repo_id="test/fake",
        summary_path=summary_path,
        config=RLConfig(state_dim=3, action_dim=2, chunk_size=2),
    )

    assert len(dataset) == 3
    assert dataset[0].action.shape == (1, 2, 2)
    assert dataset.inspection_summary() == {
        "dataset_path": str(tmp_path / "dataset"),
        "episodes": 2,
        "frames": 5,
        "decisions": 3,
        "positive_labels": 1,
        "negative_labels": 1,
        "state_shape": [3],
        "action_shape": [2],
        "image_keys": [],
        "partial_chunk_count": 1,
    }

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert batch.action.shape == (2, 2, 2)
    assert batch.observation.features["observation.state"].shape == (2, 2, 3)


def test_camera_features_are_detected_from_metadata_and_loaded_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "episodes": [
                    _episode_metadata(episode_index=0),
                    _episode_metadata(episode_index=1, true_grasp_ever=False),
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lerobot_v3, "LeRobotDataset", _FakeCameraLeRobotDataset)

    dataset = LeRobotV3DecisionDataset.from_root(
        dataset_root=tmp_path / "dataset",
        repo_id="test/camera",
        summary_path=summary_path,
        config=RLConfig(state_dim=3, action_dim=2, chunk_size=2),
    )
    source = _FakeCameraLeRobotDataset.last_instance
    assert source is not None
    assert source.decoded_indices == []
    assert dataset.inspection_summary()["image_keys"] == ["observation.image"]

    decision = dataset[0]

    assert "observation.image" in decision.observation.features
    assert "observation.image" in decision.next_observation.features
    assert sorted(set(source.decoded_indices)) == [0, 1, 2]
