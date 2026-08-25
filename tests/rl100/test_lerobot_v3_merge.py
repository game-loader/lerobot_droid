from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from RL.datasets.merge_lerobot_v3 import (
    LeRobotV3MergeSource,
    merge_lerobot_v3_datasets,
)

STATE_NAMES = ["x", "y"]
ACTION_NAMES = ["dx"]


def _summary_record(index: int, length: int, *, success: bool) -> dict[str, object]:
    return {
        "episode_index": index,
        "episode_length": length,
        "frames": length,
        "true_grasp_ever": success,
        "clear_table_ever": success,
        "final_lift_height_m": float(np.float32(0.015)) if success else 0.0,
        "final_table_contacts": 0,
        "final_hand_contacts": 1 if success else 0,
        "success": success,
        "terminal_reason": "success" if success else "horizon",
    }


def _write_source(
    root: Path,
    repo_id: str,
    *,
    task: str,
    lengths: tuple[int, ...],
    successes: tuple[bool, ...],
    canonical_fields: bool,
    robot_type: str | None = "moya_newton",
    state_names: list[str] | None = None,
    stored_successes: tuple[bool, ...] | None = None,
    include_success_field: bool = True,
) -> Path:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (1,),
            "names": ACTION_NAMES if state_names is not None else None,
        },
    }
    if canonical_fields:
        features.update(
            {
                "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "next.done": {"dtype": "bool", "shape": (1,), "names": None},
                "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
            }
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=60,
        robot_type=robot_type,
        features=features,
        root=root,
        use_videos=False,
    )
    records: list[dict[str, object]] = []
    if stored_successes is None:
        stored_successes = successes
    for episode_index, (length, success, stored_success) in enumerate(
        zip(lengths, successes, stored_successes, strict=True)
    ):
        for frame_index in range(length):
            frame: dict[str, object] = {
                "task": task,
                "observation.state": np.asarray([episode_index, frame_index], dtype=np.float32),
                "action": np.asarray([frame_index], dtype=np.float32),
            }
            if canonical_fields:
                terminal = frame_index == length - 1
                frame.update(
                    {
                        "next.reward": np.asarray(
                            [1.0 if terminal and stored_success else 0.0], dtype=np.float32
                        ),
                        "next.done": np.asarray([terminal], dtype=np.bool_),
                        "next.truncated": np.asarray([terminal and not stored_success], dtype=np.bool_),
                    }
                )
            dataset.add_frame(frame)
        dataset.save_episode()
        record = _summary_record(episode_index, length, success=success)
        if not include_success_field:
            record.pop("success")
        records.append(record)
    dataset.finalize()
    summary: dict[str, object] = {
        "repo_id": repo_id,
        "task": task,
        "fps": 60,
        "episodes_saved": len(records),
        "episodes": records,
    }
    if canonical_fields:
        summary["complete"] = True
        summary["success_count"] = sum(successes)
        summary["failure_count"] = len(successes) - sum(successes)
    (root.parent / f"{root.name}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root.parent / f"{root.name}_summary.json"


def _source(root: Path, repo_id: str, summary: Path) -> LeRobotV3MergeSource:
    return LeRobotV3MergeSource(dataset_root=root, repo_id=repo_id, summary_path=summary)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_merge_materializes_legacy_base_and_rollout(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="grasp and lift randomized charger",
        lengths=(2, 3),
        successes=(True, False),
        canonical_fields=False,
        state_names=STATE_NAMES,
        include_success_field=False,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="moya_charger_grasp",
        lengths=(1, 2),
        successes=(False, True),
        canonical_fields=True,
        robot_type=None,
        state_names=None,
    )
    base_before = _tree_digest(base_root)
    rollout_before = _tree_digest(rollout_root)

    output = merge_lerobot_v3_datasets(
        tmp_path / "merged",
        output_repo_id="local/merged",
        base=_source(base_root, "local/base", base_summary),
        rollout=_source(rollout_root, "local/rollout", rollout_summary),
        canonical_task="grasp and lift randomized charger",
    )

    assert output == tmp_path / "merged"
    merged = LeRobotDataset("local/merged", root=output / "dataset", download_videos=False)
    assert merged.num_episodes == 4
    assert merged.num_frames == 8
    assert merged.meta.robot_type == "moya_newton"
    assert merged.features["observation.state"]["names"] == STATE_NAMES
    assert merged.features["action"]["names"] == ACTION_NAMES
    for key in ("next.reward", "next.done", "next.truncated"):
        assert key in merged.features
        assert tuple(merged.features[key]["shape"]) == (1,)

    rows = merged.hf_dataset[: merged.num_frames]
    np.testing.assert_array_equal(rows["index"], np.arange(merged.num_frames))
    np.testing.assert_array_equal(rows["episode_index"], [0, 0, 1, 1, 1, 2, 3, 3])
    np.testing.assert_array_equal(rows["next.reward"], [0, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(rows["next.done"], [False, True, False, False, True, True, False, True])
    np.testing.assert_array_equal(
        rows["next.truncated"], [False, False, False, False, True, True, False, False]
    )
    assert set(merged.meta.tasks.index) == {"grasp and lift randomized charger"}

    summary = json.loads((output / "collection_summary.json").read_text())
    assert summary["complete"] is True
    assert summary["episodes_saved"] == 4
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 2
    assert len(summary["lineage"]["sources"]) == 2
    assert [record["source_episode_index"] for record in summary["episodes"]] == [0, 1, 0, 1]
    assert [record["source_task"] for record in summary["episodes"]] == [
        "grasp and lift randomized charger",
        "grasp and lift randomized charger",
        "moya_charger_grasp",
        "moya_charger_grasp",
    ]
    assert (output / "lineage.json").is_file()
    assert _tree_digest(base_root) == base_before
    assert _tree_digest(rollout_root) == rollout_before


def test_merge_requires_equal_tasks_without_canonical_task(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="base task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="rollout task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )
    with pytest.raises(ValueError, match="canonical_task|task"):
        merge_lerobot_v3_datasets(
            tmp_path / "merged",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
        )


def test_merge_accepts_prior_canonical_merge_provenance(tmp_path: Path) -> None:
    """A merged source may retain an episode task different from its canonical label."""

    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="original task",
        lengths=(1,),
        successes=(True,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="canonical task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )
    first_output = merge_lerobot_v3_datasets(
        tmp_path / "first",
        output_repo_id="local/first",
        base=_source(base_root, "local/base", base_summary),
        rollout=_source(rollout_root, "local/rollout", rollout_summary),
        canonical_task="canonical task",
    )

    second_rollout_root = tmp_path / "rollout2"
    second_rollout_summary = _write_source(
        second_rollout_root,
        "local/rollout2",
        task="canonical task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )
    second_output = merge_lerobot_v3_datasets(
        tmp_path / "second",
        output_repo_id="local/second",
        base=_source(
            first_output / "dataset",
            "local/first",
            first_output / "collection_summary.json",
        ),
        rollout=_source(second_rollout_root, "local/rollout2", second_rollout_summary),
        canonical_task="canonical task",
    )

    merged = LeRobotDataset("local/second", root=second_output / "dataset", download_videos=False)
    assert merged.num_episodes == 3
    summary = json.loads((second_output / "collection_summary.json").read_text())
    assert summary["task"] == "canonical task"
    assert summary["episodes"][0]["source_task"] == "canonical task"


def test_rollout_fields_must_match_summary(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="task",
        lengths=(1,),
        successes=(True,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
        stored_successes=(False,),
    )

    with pytest.raises(ValueError, match="summary|success|reward"):
        merge_lerobot_v3_datasets(
            tmp_path / "merged",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
        )
    assert not (tmp_path / "merged").exists()


def test_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=["different", "names"],
    )
    with pytest.raises(ValueError, match="schema|names|feature"):
        merge_lerobot_v3_datasets(
            tmp_path / "merged",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
            canonical_task="task",
        )


def test_writer_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )
    original_create = LeRobotDataset.create

    def fail_create(*args: object, **kwargs: object) -> LeRobotDataset:
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(LeRobotDataset, "create", fail_create)
    with pytest.raises(RuntimeError, match="injected writer failure"):
        merge_lerobot_v3_datasets(
            tmp_path / "merged",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
            canonical_task="task",
        )
    monkeypatch.setattr(LeRobotDataset, "create", original_create)
    assert not (tmp_path / "merged").exists()
    staging = list(tmp_path.glob("merged.incomplete-*"))
    assert len(staging) == 1
    assert json.loads((staging[0] / "collection_summary.json").read_text())["complete"] is False


def test_output_inside_source_is_rejected_before_staging(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )
    with pytest.raises(ValueError, match="outside every source"):
        merge_lerobot_v3_datasets(
            base_root / "nested-output",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
            canonical_task="task",
        )
    assert not list(base_root.glob("nested-output.incomplete-*"))


def test_canonical_merge_output_can_be_used_as_next_round_base(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="original task",
        lengths=(1,),
        successes=(True,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="moya_charger_grasp",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )

    first_output = merge_lerobot_v3_datasets(
        tmp_path / "merged-1",
        output_repo_id="local/merged-1",
        base=_source(base_root, "local/base", base_summary),
        rollout=_source(rollout_root, "local/rollout", rollout_summary),
        canonical_task="moya_charger_grasp",
    )
    first_summary = first_output / "collection_summary.json"
    first_payload = json.loads(first_summary.read_text())
    assert first_payload["task"] == "moya_charger_grasp"
    assert first_payload["episodes"][0]["source_task"] == "original task"

    second_output = merge_lerobot_v3_datasets(
        tmp_path / "merged-2",
        output_repo_id="local/merged-2",
        base=_source(first_output / "dataset", "local/merged-1", first_summary),
        rollout=_source(rollout_root, "local/rollout", rollout_summary),
        canonical_task="moya_charger_grasp",
    )

    second_summary = json.loads((second_output / "collection_summary.json").read_text())
    assert second_summary["task"] == "moya_charger_grasp"
    assert all(record["source_task"] == "moya_charger_grasp" for record in second_summary["episodes"])


def test_explicit_empty_source_task_is_rejected(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    rollout_root = tmp_path / "rollout"
    base_summary = _write_source(
        base_root,
        "local/base",
        task="original task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=False,
        state_names=STATE_NAMES,
    )
    payload = json.loads(base_summary.read_text())
    payload["episodes"][0]["source_task"] = None
    base_summary.write_text(json.dumps(payload), encoding="utf-8")
    rollout_summary = _write_source(
        rollout_root,
        "local/rollout",
        task="canonical task",
        lengths=(1,),
        successes=(False,),
        canonical_fields=True,
        robot_type=None,
        state_names=STATE_NAMES,
    )

    with pytest.raises(ValueError, match="source_task must be a nonempty string"):
        merge_lerobot_v3_datasets(
            tmp_path / "merged",
            output_repo_id="local/merged",
            base=_source(base_root, "local/base", base_summary),
            rollout=_source(rollout_root, "local/rollout", rollout_summary),
            canonical_task="canonical task",
        )
