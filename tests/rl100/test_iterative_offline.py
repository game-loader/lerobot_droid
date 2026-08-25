# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lerobot.scripts.lerobot_train import _save_periodic_eval_provenance
from RL.cli import train_iterative_offline as iterative
from RL.cli.train_iterative_offline import (
    BestCheckpoint,
    _parser,
    _run_one_round,
    select_best_checkpoint,
)


def _evaluated_checkpoint(root: Path, label: str, success: float, episodes: int = 100) -> None:
    checkpoint = root / "checkpoints" / label / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    evaluation = root / "eval" / label
    evaluation.mkdir(parents=True)
    (evaluation / "eval_info.json").write_text(
        f'{{"overall": {{"pc_success": {success}, "n_episodes": {episodes}}}}}',
        encoding="utf-8",
    )


def _checkpoint(path: Path, *, model: bytes = b"model") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(model)
    (path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (path / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    return path.resolve()


def _run_il_eval_for_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint: Path,
    calls: list[Path],
) -> iterative.ILEvaluation:
    def fake_newton_eval(**kwargs: Any) -> Path:
        output_dir = Path(kwargs["output_dir"])
        assert not output_dir.exists()
        calls.append(Path(kwargs["policy_path"]).resolve())
        output_dir.mkdir(parents=True)
        eval_info = output_dir / "eval_info.json"
        eval_info.write_text(
            '{"overall": {"pc_success": 61, "n_episodes": 100}}',
            encoding="utf-8",
        )
        return eval_info.resolve()

    monkeypatch.setattr(iterative, "_run_newton_eval", fake_newton_eval)
    return iterative._run_il_evaluation_stage(
        eval_output=tmp_path / "train",
        specs=[(5_000, "step_005000", checkpoint)],
        expected_episodes=100,
        batch_size=16,
        inference_steps=10,
        policy_device="cuda",
        env_device="cuda:0",
        env_type="moya_newton",
        seed=101_000,
        log_root=tmp_path / "logs",
    )


def test_best_checkpoint_is_selected_by_measured_success(tmp_path: Path) -> None:
    _evaluated_checkpoint(tmp_path, "sync_005", 56.0)
    _evaluated_checkpoint(tmp_path, "sync_010", 65.0)
    _evaluated_checkpoint(tmp_path, "sync_015", 59.0)

    selected = select_best_checkpoint(tmp_path)

    assert selected.label == "sync_010"
    assert selected.success_rate == 65.0
    assert selected.episodes == 100


def test_best_checkpoint_tie_uses_earlier_sync(tmp_path: Path) -> None:
    _evaluated_checkpoint(tmp_path, "sync_010", 65.0)
    _evaluated_checkpoint(tmp_path, "sync_005", 65.0)

    assert select_best_checkpoint(tmp_path).label == "sync_005"


def test_best_checkpoint_requires_the_requested_evaluation_size(tmp_path: Path) -> None:
    _evaluated_checkpoint(tmp_path, "sync_005", 99.0, episodes=1)
    _evaluated_checkpoint(tmp_path, "sync_010", 65.0, episodes=100)

    selected = select_best_checkpoint(tmp_path, required_episodes=100)

    assert selected.label == "sync_010"
    assert selected.episodes == 100


def test_sync_only_selection_ignores_actor_evaluations(tmp_path: Path) -> None:
    _evaluated_checkpoint(tmp_path, "actor_000005", 99.0)
    _evaluated_checkpoint(tmp_path, "sync_005", 65.0)

    assert select_best_checkpoint(tmp_path, sync_only=True).label == "sync_005"


def test_il_checkpoint_selection_prefers_highest_newton_success(tmp_path: Path) -> None:
    evaluations = [
        iterative.ILEvaluation(tmp_path / "step5", "step_005000", 5_000, 51.0, 100, tmp_path / "e5"),
        iterative.ILEvaluation(tmp_path / "step10", "step_010000", 10_000, 63.0, 100, tmp_path / "e10"),
        iterative.ILEvaluation(tmp_path / "step15", "step_015000", 15_000, 63.0, 100, tmp_path / "e15"),
    ]

    selected = iterative._select_best_il_checkpoint(evaluations)

    assert selected.label == "step_010000"


def test_offline_il_checkpoint_can_be_pinned_to_final(tmp_path: Path) -> None:
    best_checkpoint = _checkpoint(tmp_path / "best")
    final_checkpoint = _checkpoint(tmp_path / "final")
    best = iterative.ILEvaluation(
        checkpoint=best_checkpoint,
        label="step_005000",
        step=5_000,
        success_rate=99.0,
        episodes=100,
        eval_info=tmp_path / "eval.json",
    )

    assert (
        iterative._select_offline_il_checkpoint(
            best,
            final_checkpoint,
            use_final=False,
        )
        == best_checkpoint
    )
    assert (
        iterative._select_offline_il_checkpoint(
            best,
            final_checkpoint,
            use_final=True,
        )
        == final_checkpoint
    )


def test_il_checkpoint_specs_include_ten_5k_evaluation_points(tmp_path: Path) -> None:
    train_output = tmp_path / "train"
    for step in range(5_000, 50_001, 5_000):
        _checkpoint(train_output / "checkpoints" / f"{step:06d}" / "pretrained_model")

    specs = iterative._il_checkpoint_specs(
        train_output,
        steps=50_000,
        save_freq=5_000,
        eval_every_steps=5_000,
    )

    assert [(step, label) for step, label, _checkpoint_path in specs] == [
        (step, f"step_{step:06d}") for step in range(5_000, 50_001, 5_000)
    ]


def test_il_evaluation_reuses_only_matching_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    calls: list[Path] = []
    first = _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)

    def fail_if_rerun(**_kwargs: Any) -> Path:
        pytest.fail("matching evaluation provenance should be reused")

    monkeypatch.setattr(iterative, "_run_newton_eval", fail_if_rerun)
    second = iterative._run_il_evaluation_stage(
        eval_output=tmp_path / "train",
        specs=[(5_000, "step_005000", checkpoint)],
        expected_episodes=100,
        batch_size=16,
        inference_steps=10,
        policy_device="cuda",
        env_device="cuda:0",
        env_type="moya_newton",
        seed=101_000,
        log_root=tmp_path / "logs",
    )

    provenance = json.loads(
        (tmp_path / "train" / "eval" / "step_005000" / "eval_provenance.json").read_text(encoding="utf-8")
    )
    assert calls == [checkpoint]
    assert second == first
    assert provenance == {
        "batch_size": 16,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": iterative._digest_tree(checkpoint),
        "env_device": "cuda:0",
        "env_type": "moya_newton",
        "episodes": 100,
        "inference_steps": 10,
        "policy_device": "cuda",
        "schema_version": 1,
        "seed": 101_000,
    }


def test_trainer_written_provenance_is_reused_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_output = tmp_path / "train"
    checkpoint_dir = train_output / "checkpoints" / "005000"
    checkpoint = _checkpoint(checkpoint_dir / "pretrained_model")
    eval_dir = train_output / "eval" / "step_005000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_info.json").write_text(
        '{"overall": {"pc_success": 64, "n_episodes": 100}}', encoding="utf-8"
    )
    cfg = SimpleNamespace(
        is_reward_model_training=False,
        policy=SimpleNamespace(device="cuda", num_inference_steps=10),
        env=SimpleNamespace(device="cuda:0", type="moya_newton"),
        eval=SimpleNamespace(n_episodes=100, batch_size=16),
        seed=101_000,
    )
    sidecar = _save_periodic_eval_provenance(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        eval_dir=eval_dir,
    )

    monkeypatch.setattr(
        iterative,
        "_run_newton_eval",
        lambda **_kwargs: pytest.fail("trainer evaluation should be reused without subprocess"),
    )
    selected = iterative._run_il_evaluation_stage(
        eval_output=train_output,
        specs=[(5_000, "step_005000", checkpoint)],
        expected_episodes=100,
        batch_size=16,
        inference_steps=10,
        policy_device="cuda",
        env_device="cuda:0",
        env_type="moya_newton",
        seed=101_000,
        log_root=tmp_path / "logs",
    )

    assert sidecar == (eval_dir / "eval_provenance.json").resolve()
    assert selected.checkpoint == checkpoint
    assert selected.success_rate == 64.0


def test_trainer_writes_no_provenance_without_same_step_checkpoint(tmp_path: Path) -> None:
    eval_dir = tmp_path / "train" / "eval" / "step_002500"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_info.json").write_text(
        '{"overall": {"pc_success": 64, "n_episodes": 100}}', encoding="utf-8"
    )
    cfg = SimpleNamespace(
        is_reward_model_training=False,
        policy=SimpleNamespace(device="cuda", num_inference_steps=10),
        env=SimpleNamespace(device="cuda:0", type="moya_newton"),
        eval=SimpleNamespace(n_episodes=100, batch_size=16),
        seed=101_000,
    )

    sidecar = _save_periodic_eval_provenance(
        cfg=cfg,
        checkpoint_dir=None,
        eval_dir=eval_dir,
    )

    assert sidecar is None
    assert not (eval_dir / "eval_provenance.json").exists()


@pytest.mark.parametrize("stale_sidecar", [None, "not-json", '{"episodes": 1}'])
def test_il_evaluation_reruns_missing_or_invalid_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_sidecar: str | None,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    eval_dir = tmp_path / "train" / "eval" / "step_005000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_info.json").write_text(
        '{"overall": {"pc_success": 99, "n_episodes": 100}}', encoding="utf-8"
    )
    if stale_sidecar is not None:
        (eval_dir / "eval_provenance.json").write_text(stale_sidecar, encoding="utf-8")
    calls: list[Path] = []

    selected = _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)

    assert calls == [checkpoint]
    assert selected.success_rate == 61.0
    assert json.loads((eval_dir / "eval_provenance.json").read_text(encoding="utf-8"))["episodes"] == 100


def test_il_evaluation_checkpoint_mutation_invalidates_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    calls: list[Path] = []
    _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)
    old_provenance = json.loads(
        (tmp_path / "train" / "eval" / "step_005000" / "eval_provenance.json").read_text(encoding="utf-8")
    )
    (checkpoint / "model.safetensors").write_bytes(b"mutated-model")

    _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)

    new_provenance = json.loads(
        (tmp_path / "train" / "eval" / "step_005000" / "eval_provenance.json").read_text(encoding="utf-8")
    )
    assert calls == [checkpoint, checkpoint]
    assert new_provenance["checkpoint_sha256"] != old_provenance["checkpoint_sha256"]


def test_il_evaluation_recovers_incomplete_point_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    eval_dir = tmp_path / "train" / "eval" / "step_005000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "partial.tmp").write_text("interrupted", encoding="utf-8")
    calls: list[Path] = []

    _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)

    assert calls == [checkpoint]
    assert not (eval_dir / "partial.tmp").exists()
    assert (eval_dir / "eval_info.json").is_file()
    assert (eval_dir / "eval_provenance.json").is_file()


def test_il_evaluation_unlinks_stale_point_symlink_without_deleting_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    eval_dir = tmp_path / "train" / "eval" / "step_005000"
    eval_dir.parent.mkdir(parents=True)
    eval_dir.symlink_to(external, target_is_directory=True)
    calls: list[Path] = []

    _run_il_eval_for_test(tmp_path, monkeypatch, checkpoint=checkpoint, calls=calls)

    assert calls == [checkpoint]
    assert marker.read_text(encoding="utf-8") == "keep"
    assert eval_dir.is_dir()
    assert not eval_dir.is_symlink()


def test_il_evaluation_writes_no_provenance_before_newton_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    eval_dir = tmp_path / "train" / "eval" / "step_005000"

    def interrupted_eval(**kwargs: Any) -> Path:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "eval_info.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(iterative, "_run_newton_eval", interrupted_eval)

    with pytest.raises(RuntimeError, match="interrupted"):
        iterative._run_il_evaluation_stage(
            eval_output=tmp_path / "train",
            specs=[(5_000, "step_005000", checkpoint)],
            expected_episodes=100,
            batch_size=16,
            inference_steps=10,
            policy_device="cuda",
            env_device="cuda:0",
            env_type="moya_newton",
            seed=101_000,
            log_root=tmp_path / "logs",
        )

    assert not (eval_dir / "eval_provenance.json").exists()


def test_best_checkpoint_requires_complete_evaluation(tmp_path: Path) -> None:
    (tmp_path / "eval" / "sync_005").mkdir(parents=True)
    (tmp_path / "eval" / "sync_005" / "eval_info.json").write_text(
        '{"overall": {"pc_success": 99, "n_episodes": 100}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no complete"):
        select_best_checkpoint(tmp_path)


def test_parser_defaults_dynamic_selection_and_resume() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "outputs/rounds",
            "--base-dataset-root",
            "base",
            "--base-repo-id",
            "local/base",
            "--base-summary",
            "base-summary.json",
            "--il-checkpoint",
            "il",
            "--source-offline-run",
            "offline",
        ]
    )
    assert args.rounds == 1
    assert args.source_eval_episodes == 100
    assert args.inference_steps == 10
    assert args.final_rollout_merge is False
    assert args.offline_sync_target == 50
    assert args.offline_use_il_final is False
    assert args.offline_eval_every_syncs == 5
    assert args.resume is True
    assert args.offline_amq_enabled is True
    assert args.offline_debug is False


def test_selection_payload_detects_source_artifact_mutation(tmp_path: Path) -> None:
    _evaluated_checkpoint(tmp_path, "sync_005", 65.0)
    selected = select_best_checkpoint(tmp_path)
    before = iterative._selection_payload(selected)

    (selected.checkpoint / "model.safetensors").write_bytes(b"updated-model")
    after_model_change = iterative._selection_payload(selected)
    assert after_model_change["checkpoint_sha256"] != before["checkpoint_sha256"]

    selected.eval_info.write_text('{"overall": {"pc_success": 66, "n_episodes": 100}}', encoding="utf-8")
    after_eval_change = iterative._selection_payload(selected)
    assert after_eval_change["eval_info_sha256"] != before["eval_info_sha256"]


def test_smoke_offline_command_uses_direct_sync_instead_of_amq(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(tmp_path / "base"),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(tmp_path / "base-summary.json"),
            "--il-checkpoint",
            str(tmp_path / "il"),
            "--source-offline-run",
            str(tmp_path / "offline"),
            "--swanlab-mode",
            "disabled",
            "--smoke",
        ]
    )
    command = iterative._offline_command(
        args,
        tmp_path / "checkpoint",
        tmp_path / "dataset",
        "local/merged",
        tmp_path / "summary.json",
        tmp_path / "output",
        seed=1,
        run_name="smoke",
    )

    assert "--smoke" in command
    assert "--amq-enabled" not in command
    assert "--old-policy-sync-interval=1" in command
    assert "--debug" not in command
    collection = iterative._collection_command(
        args,
        BestCheckpoint(
            checkpoint=tmp_path / "checkpoint",
            label="sync_005",
            success_rate=65.0,
            episodes=100,
            eval_info=tmp_path / "eval.json",
        ),
        tmp_path / "collection",
        seed=1,
        repo_id="local/rollout",
    )
    assert "--inference-steps=10" in collection


def test_offline_validation_rejects_corrupt_final_manifest(tmp_path: Path) -> None:
    final = tmp_path / "checkpoints" / "final"
    checkpoint = final / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    (final / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is invalid"):
        iterative._offline_valid(
            tmp_path,
            sync_target=1,
            eval_every_syncs=1,
            eval_episodes=100,
        )


def test_dry_run_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source-offline"
    _evaluated_checkpoint(source, "sync_005", 65.0)
    base = tmp_path / "base"
    base.mkdir()
    base_summary = tmp_path / "base-summary.json"
    base_summary.write_text('{"task": "grasp"}', encoding="utf-8")
    il_checkpoint = tmp_path / "il"
    il_checkpoint.mkdir()
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(base),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(base_summary),
            "--il-checkpoint",
            str(il_checkpoint),
            "--source-offline-run",
            str(source),
            "--swanlab-mode",
            "disabled",
            "--dry-run",
        ]
    )

    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )
    manifest_path = tmp_path / "rounds" / "round_001" / "round_manifest.json"
    first = manifest_path.read_bytes()

    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )

    assert manifest_path.read_bytes() == first


def test_dry_run_after_failed_stage_does_not_grow_history(tmp_path: Path) -> None:
    source = tmp_path / "source-offline"
    _evaluated_checkpoint(source, "sync_005", 65.0)
    base = tmp_path / "base"
    base.mkdir()
    base_summary = tmp_path / "base-summary.json"
    base_summary.write_text('{"task": "grasp"}', encoding="utf-8")
    il_checkpoint = tmp_path / "il"
    il_checkpoint.mkdir()
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(base),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(base_summary),
            "--il-checkpoint",
            str(il_checkpoint),
            "--source-offline-run",
            str(source),
            "--swanlab-mode",
            "disabled",
            "--dry-run",
        ]
    )
    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )
    manifest_path = tmp_path / "rounds" / "round_001" / "round_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["rollout"]["status"] = "failed"
    manifest["stages"]["rollout"]["error"] = "simulated failure"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args.dry_run = True
    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )
    first_planned = manifest_path.read_bytes()
    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )

    assert manifest_path.read_bytes() == first_planned


def test_complete_round_rechecks_current_source_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = BestCheckpoint(
        checkpoint=tmp_path / "current",
        label="sync_010",
        success_rate=70.0,
        episodes=100,
        eval_info=tmp_path / "current-eval.json",
    )
    monkeypatch.setattr(iterative, "select_best_checkpoint", lambda *args, **kwargs: current)
    monkeypatch.setattr(iterative, "_selection_payload", lambda selected: {"label": selected.label})
    manifest = {
        "inputs": {},
        "outputs": {},
        "selection": {"label": "sync_005"},
        "stages": {},
    }

    with pytest.raises(ValueError, match="selection changed"):
        iterative._validate_complete_round(
            manifest,
            source_offline_run=tmp_path,
            source_eval_episodes=100,
            collection_episodes=100,
            sync_target=50,
            eval_every_syncs=5,
            eval_episodes=100,
        )


def test_complete_round_resume_runs_no_stage_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-offline"
    _evaluated_checkpoint(source, "sync_005", 65.0)
    base = tmp_path / "base"
    base.mkdir()
    base_summary = tmp_path / "base-summary.json"
    base_summary.write_text('{"task": "grasp"}', encoding="utf-8")
    il_checkpoint = tmp_path / "il"
    il_checkpoint.mkdir()
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(base),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(base_summary),
            "--il-checkpoint",
            str(il_checkpoint),
            "--source-offline-run",
            str(source),
            "--swanlab-mode",
            "disabled",
            "--dry-run",
        ]
    )
    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )
    manifest_path = tmp_path / "rounds" / "round_001" / "round_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["outputs"] = {"offline_run": str(tmp_path / "complete")}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args.dry_run = False
    monkeypatch.setattr(iterative, "_validate_complete_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        iterative,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("resume executed a stage command"),
    )
    monkeypatch.setattr(
        iterative,
        "merge_lerobot_v3_datasets",
        lambda *args, **kwargs: pytest.fail("resume reran dataset merge"),
    )

    resumed = _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
    )

    assert resumed["outputs"]["offline_run"] == str(tmp_path / "complete")


def test_run_adds_collection_merge_tail_after_full_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(tmp_path / "base"),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(tmp_path / "base-summary.json"),
            "--il-checkpoint",
            str(tmp_path / "initial-il"),
            "--source-offline-run",
            str(tmp_path / "initial-offline"),
            "--final-rollout-merge",
        ]
    )
    calls: list[dict[str, object]] = []
    initial_il = tmp_path / "initial-il"
    round_il = tmp_path / "round-il"
    initial_offline = tmp_path / "initial-offline"
    initial_il.mkdir()
    round_il.mkdir()
    initial_offline.mkdir()

    monkeypatch.setattr(iterative, "_validate_args", lambda args: None)
    monkeypatch.setattr(iterative, "_checkpoint_valid", lambda path: Path(path).resolve())

    def run_round(_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        if kwargs.get("stop_after_merge") is True:
            return {
                "outputs": {
                    "dataset_root": str(tmp_path / "merged-300"),
                    "repo_id": "local/merged-300",
                    "summary": str(tmp_path / "merged-300-summary.json"),
                }
            }
        return {
            "outputs": {
                "dataset_root": str(tmp_path / "merged-200"),
                "repo_id": "local/merged-200",
                "summary": str(tmp_path / "merged-200-summary.json"),
                "il_checkpoint": str(round_il),
                "offline_run": str(tmp_path / "offline-round"),
            }
        }

    monkeypatch.setattr(iterative, "_run_one_round", run_round)

    result = iterative.run(args)

    assert result == (tmp_path / "merged-300").resolve()
    assert len(calls) == 2
    assert calls[0]["round_index"] == 1
    assert calls[0].get("stop_after_merge") is None
    assert calls[1]["round_index"] == 2
    assert calls[1]["base_dataset_root"] == (tmp_path / "merged-200").resolve()
    assert calls[1]["source_offline_run"] == (tmp_path / "offline-round").resolve()
    assert calls[1]["source_eval_episodes"] == 100
    assert calls[1]["stop_after_merge"] is True


def test_collection_merge_round_returns_before_il_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-offline"
    _evaluated_checkpoint(source, "sync_005", 65.0)
    base = tmp_path / "base"
    base.mkdir()
    base_summary = tmp_path / "base-summary.json"
    base_summary.write_text('{"task": "grasp"}', encoding="utf-8")
    il_checkpoint = tmp_path / "il"
    il_checkpoint.mkdir()
    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "rounds"),
            "--base-dataset-root",
            str(base),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(base_summary),
            "--il-checkpoint",
            str(il_checkpoint),
            "--source-offline-run",
            str(source),
            "--swanlab-mode",
            "disabled",
        ]
    )
    monkeypatch.setattr(iterative, "_digest_tree", lambda path: f"tree:{Path(path).resolve()}")
    monkeypatch.setattr(iterative, "_digest_file", lambda path: f"file:{Path(path).resolve()}")

    def run_logged(command: list[str], *, log_path: Path, env=None) -> None:
        del log_path, env
        assert "RL.cli.collect_moya_il" in command

    def collection_valid(output: Path, *, repo_id: str, expected_episodes: int):
        del repo_id, expected_episodes
        dataset = output / "dataset"
        summary = output / "collection_summary.json"
        dataset.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return dataset, summary

    def merge_datasets(output: Path, **_: object) -> Path:
        (output / "dataset").mkdir(parents=True)
        (output / "collection_summary.json").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(iterative, "_run_logged", run_logged)
    monkeypatch.setattr(iterative, "_collection_valid", collection_valid)
    monkeypatch.setattr(iterative, "merge_lerobot_v3_datasets", merge_datasets)
    monkeypatch.setattr(iterative, "_canonical_dataset_valid", lambda *args, **kwargs: None)
    completed = _run_one_round(
        args,
        round_index=2,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=il_checkpoint,
        source_offline_run=source,
        stop_after_merge=True,
    )

    assert completed["status"] == "complete"
    assert set(completed["stages"]) == {"rollout", "merge"}
    assert completed["outputs"]["source_label"] == "sync_005"
    assert completed["outputs"]["dataset_root"].endswith("/merge/attempt_001/merged/dataset")


def test_replaced_rollout_rebuilds_all_downstream_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-offline"
    _evaluated_checkpoint(source, "sync_005", 65.0)
    base = tmp_path / "base"
    base.mkdir()
    base_summary = tmp_path / "base-summary.json"
    base_summary.write_text('{"task": "grasp"}', encoding="utf-8")
    source_il = tmp_path / "source-il"
    source_il.mkdir()
    output_root = tmp_path / "rounds"
    args = _parser().parse_args(
        [
            "--output-root",
            str(output_root),
            "--base-dataset-root",
            str(base),
            "--base-repo-id",
            "local/base",
            "--base-summary",
            str(base_summary),
            "--il-checkpoint",
            str(source_il),
            "--source-offline-run",
            str(source),
            "--swanlab-mode",
            "disabled",
            "--dry-run",
        ]
    )
    monkeypatch.setattr(iterative, "_digest_tree", lambda path: f"tree:{Path(path).resolve()}")
    monkeypatch.setattr(iterative, "_digest_file", lambda path: f"file:{Path(path).resolve()}")
    _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=source_il,
        source_offline_run=source,
    )

    round_root = output_root / "round_001"
    manifest_path = round_root / "round_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_best_checkpoint(source)
    old_rollout = tmp_path / "old-rollout"
    old_rollout_dataset = old_rollout / "dataset"
    old_rollout_summary = old_rollout / "collection_summary.json"
    old_rollout_dataset.mkdir(parents=True)
    old_rollout_summary.write_text("{}", encoding="utf-8")
    old_merge = tmp_path / "old-merge"
    old_merge_dataset = old_merge / "dataset"
    old_merge_summary = old_merge / "collection_summary.json"
    old_merge_dataset.mkdir(parents=True)
    old_merge_summary.write_text("{}", encoding="utf-8")
    old_il = tmp_path / "old-il"
    old_il.mkdir()
    old_offline = tmp_path / "old-offline"
    old_offline.mkdir()
    rollout_repo = "local/moya-rl100-round-001-rollout"
    merged_repo = "local/moya-rl100-round-001-merged"
    manifest["stages"] = {
        "rollout": {
            "status": "complete",
            "inputs": iterative._rollout_stage_inputs(selected),
            "output_dir": str(old_rollout),
            "repo_id": rollout_repo,
        },
        "merge": {
            "status": "complete",
            "inputs": iterative._merge_stage_inputs(
                base_dataset_root=base,
                base_repo_id="local/base",
                base_summary=base_summary,
                rollout_dataset=old_rollout_dataset,
                rollout_repo_id=rollout_repo,
                rollout_summary=old_rollout_summary,
            ),
            "output_dir": str(old_merge),
            "dataset_root": str(old_merge_dataset),
            "summary": str(old_merge_summary),
            "repo_id": merged_repo,
        },
        "il": {
            "status": "complete",
            "inputs": iterative._il_stage_inputs(
                source_checkpoint=source_il,
                dataset_root=old_merge_dataset,
                repo_id=merged_repo,
            ),
            "output_dir": str(tmp_path / "old-il-run"),
            "checkpoint": str(old_il),
        },
        "offline": {
            "status": "complete",
            "inputs": iterative._offline_stage_inputs(
                checkpoint=old_il,
                dataset_root=old_merge_dataset,
                repo_id=merged_repo,
                summary=old_merge_summary,
            ),
            "output_dir": str(old_offline),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls = {"collect": 0, "merge": 0, "il": 0, "offline": 0}

    def collection_valid(output: Path, *, repo_id: str, expected_episodes: int):
        assert repo_id == rollout_repo
        assert expected_episodes == 100
        if output == old_rollout:
            raise ValueError("old rollout is corrupt")
        calls["collect"] += 1
        dataset = output / "dataset"
        summary = output / "collection_summary.json"
        dataset.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return dataset, summary

    def merge_datasets(output: Path, **_: object) -> Path:
        calls["merge"] += 1
        (output / "dataset").mkdir(parents=True)
        (output / "collection_summary.json").write_text("{}", encoding="utf-8")
        return output

    new_il = tmp_path / "new-il"

    def run_logged(command: list[str], *, log_path: Path, env=None) -> None:
        del log_path, env
        if "RL.cli.collect_moya_il" in command:
            return
        if "RL.cli.train_il_warmstart" in command:
            calls["il"] += 1
            new_il.mkdir(exist_ok=True)
            output_arg = next(value for value in command if value.startswith("--output-dir="))
            stage_manifest = Path(output_arg.split("=", 1)[1]).parent / "il_stage.json"
            stage_manifest.parent.mkdir(parents=True, exist_ok=True)
            stage_manifest.write_text(json.dumps({"checkpoint": str(new_il)}), encoding="utf-8")
            return
        if "RL.cli.train_offline" in command:
            calls["offline"] += 1
            return
        raise AssertionError(command)

    new_best = BestCheckpoint(
        checkpoint=tmp_path / "new-best",
        label="sync_005",
        success_rate=70.0,
        episodes=100,
        eval_info=tmp_path / "new-eval.json",
    )
    monkeypatch.setattr(iterative, "_collection_valid", collection_valid)
    monkeypatch.setattr(iterative, "merge_lerobot_v3_datasets", merge_datasets)
    monkeypatch.setattr(iterative, "_canonical_dataset_valid", lambda *args, **kwargs: None)
    monkeypatch.setattr(iterative, "_run_logged", run_logged)
    monkeypatch.setattr(iterative, "_checkpoint_valid", lambda path: Path(path).resolve())
    monkeypatch.setattr(
        iterative,
        "_il_stage_valid",
        lambda record, **kwargs: Path(str(record["checkpoint"])).resolve(),
    )
    monkeypatch.setattr(iterative, "_offline_valid", lambda *args, **kwargs: new_best)
    monkeypatch.setattr(
        iterative,
        "_il_checkpoint_specs",
        lambda *args, **kwargs: [(50_000, "step_050000", new_il)],
    )
    monkeypatch.setattr(
        iterative,
        "_il_eval_inputs",
        lambda *args, **kwargs: {"mock": True},
    )

    def run_il_eval(**kwargs: object) -> iterative.ILEvaluation:
        del kwargs
        (new_il / "model.safetensors").write_bytes(b"new-il")
        eval_info = tmp_path / "new-il-eval.json"
        eval_info.write_text(
            '{"overall": {"pc_success": 70, "n_episodes": 100}}',
            encoding="utf-8",
        )
        return iterative.ILEvaluation(
            checkpoint=new_il,
            label="step_050000",
            step=50_000,
            success_rate=70.0,
            episodes=100,
            eval_info=eval_info,
        )

    monkeypatch.setattr(iterative, "_run_il_evaluation_stage", run_il_eval)
    args.dry_run = False

    completed = _run_one_round(
        args,
        round_index=1,
        base_dataset_root=base,
        base_repo_id="local/base",
        base_summary=base_summary,
        il_checkpoint=source_il,
        source_offline_run=source,
    )

    assert calls == {"collect": 1, "merge": 1, "il": 1, "offline": 1}
    assert completed["outputs"]["dataset_root"] != str(old_merge_dataset)
    assert completed["outputs"]["il_checkpoint"] == str(new_il.resolve())
