# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Resumable RL-100-style rollout, IL, and offline-RL outer loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.eval_provenance import (
    EVAL_PROVENANCE_FILENAME,
    build_eval_provenance,
    sha256_file as _digest_file,
    sha256_tree as _digest_tree,
    write_eval_provenance,
)
from RL.checkpointing import _load_manifest as _load_rl_manifest
from RL.cli.train_il_warmstart import processor_fingerprint
from RL.datasets.merge_lerobot_v3 import LeRobotV3MergeSource, merge_lerobot_v3_datasets

_MANIFEST_VERSION = 3
_SYNC_RE = re.compile(r"^sync_(\d+)$")
_IL_STEP_RE = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class BestCheckpoint:
    """A real-environment-evaluated policy checkpoint."""

    checkpoint: Path
    label: str
    success_rate: float
    episodes: int
    eval_info: Path


@dataclass(frozen=True)
class ILEvaluation:
    """A Diffusion Policy IL checkpoint evaluated in the real Newton env."""

    checkpoint: Path
    label: str
    step: int
    success_rate: float
    episodes: int
    eval_info: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _eval_aggregate(payload: Mapping[str, Any], *, path: Path) -> tuple[float, int]:
    aggregate: Mapping[str, Any] | None = None
    for key in ("overall", "aggregated"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            aggregate = candidate
            break
    if aggregate is None:
        raise ValueError(f"evaluation info has no overall/aggregated metrics: {path}")
    success = _finite_number(aggregate.get("pc_success"), name="pc_success")
    if not 0.0 <= success <= 100.0:
        raise ValueError(f"evaluation pc_success must be within [0, 100]: {path}")
    episodes_value = _finite_number(aggregate.get("n_episodes"), name="n_episodes")
    episodes = int(episodes_value)
    if episodes <= 0 or episodes_value != episodes:
        raise ValueError(f"evaluation n_episodes must be a positive integer: {path}")
    return success, episodes


def _checkpoint_from_label(offline_run: Path, label: str) -> Path:
    checkpoint = offline_run / "checkpoints" / label / "pretrained_model"
    checkpoint = checkpoint.resolve(strict=True)
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"evaluated checkpoint is missing {name}: {checkpoint}")
    return checkpoint


def _label_order(label: str) -> tuple[int, int, str]:
    match = _SYNC_RE.match(label)
    if match:
        return (0, int(match.group(1)), label)
    actor_match = re.match(r"^actor_(\d+)$", label)
    if actor_match:
        return (1, int(actor_match.group(1)), label)
    return (2, 0, label)


def select_best_checkpoint(
    offline_run: Path | str,
    *,
    required_episodes: int | None = None,
    sync_only: bool = False,
) -> BestCheckpoint:
    """Select the highest real Newton success checkpoint, with deterministic ties."""

    if required_episodes is not None and required_episodes <= 0:
        raise ValueError("required_episodes must be positive when provided")
    root = Path(offline_run).resolve(strict=True)
    eval_root = root / "eval"
    if not eval_root.is_dir():
        raise ValueError(f"offline run has no eval directory: {eval_root}")
    candidates: list[BestCheckpoint] = []
    for eval_info in sorted(eval_root.glob("*/eval_info.json")):
        label = eval_info.parent.name
        if sync_only and _SYNC_RE.match(label) is None:
            continue
        payload = _read_json(eval_info)
        success_rate, episodes = _eval_aggregate(payload, path=eval_info)
        if required_episodes is not None and episodes != required_episodes:
            continue
        try:
            checkpoint = _checkpoint_from_label(root, label)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append(
            BestCheckpoint(
                checkpoint=checkpoint,
                label=label,
                success_rate=success_rate,
                episodes=episodes,
                eval_info=eval_info.resolve(),
            )
        )
    if not candidates:
        raise ValueError(f"no complete evaluated checkpoints found under {eval_root}")
    candidates.sort(key=lambda item: (-item.success_rate, _label_order(item.label)))
    return candidates[0]


def _selection_payload(selected: BestCheckpoint) -> dict[str, Any]:
    return {
        "checkpoint": str(selected.checkpoint),
        "checkpoint_sha256": _digest_tree(selected.checkpoint),
        "eval_info": str(selected.eval_info),
        "eval_info_sha256": _digest_file(selected.eval_info),
        "episodes": selected.episodes,
        "label": selected.label,
        "success_rate": selected.success_rate,
    }


def _il_label(step: int) -> str:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"IL checkpoint step must be a positive integer, got {step!r}")
    return f"step_{step:06d}"


def _il_checkpoint_specs(
    train_output: Path | str,
    *,
    steps: int,
    save_freq: int,
    eval_every_steps: int | None = None,
) -> list[tuple[int, str, Path]]:
    """Return the checkpoints that must be evaluated for one IL stage.

    LeRobot writes numeric checkpoint directories at ``save_freq`` intervals.
    If the requested total is not divisible by ``save_freq``, the final
    ``checkpoints/last`` bundle is included as the final evaluation point.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("IL evaluation steps must be a positive integer")
    if isinstance(save_freq, bool) or not isinstance(save_freq, int) or save_freq <= 0:
        raise ValueError("IL evaluation save_freq must be a positive integer")
    if eval_every_steps is None:
        eval_every_steps = save_freq
    if isinstance(eval_every_steps, bool) or not isinstance(eval_every_steps, int) or eval_every_steps <= 0:
        raise ValueError("IL eval_every_steps must be a positive integer")
    if eval_every_steps % save_freq != 0:
        raise ValueError("IL eval_every_steps must be a multiple of il_save_freq")
    root = Path(train_output).resolve(strict=True)
    checkpoints_root = root / "checkpoints"
    if not checkpoints_root.is_dir():
        raise ValueError(f"IL output has no checkpoints directory: {checkpoints_root}")
    saved_steps = set(range(save_freq, steps + 1, save_freq))
    expected_steps = list(range(eval_every_steps, steps + 1, eval_every_steps))
    if not expected_steps or expected_steps[-1] != steps:
        expected_steps.append(steps)
    specs: list[tuple[int, str, Path]] = []
    for step in expected_steps:
        if step not in saved_steps and step != steps:
            raise ValueError(
                f"IL checkpoint for evaluation step {step} is not saved; "
                f"use an eval interval that lands on il_save_freq={save_freq}"
            )
        numeric = checkpoints_root / f"{step:06d}" / "pretrained_model"
        if numeric.exists():
            checkpoint = _checkpoint_valid(numeric)
        elif step == steps:
            checkpoint = _checkpoint_valid(checkpoints_root / "last")
        else:
            raise ValueError(f"IL checkpoint for step {step} is missing under {checkpoints_root}")
        specs.append((step, _il_label(step), checkpoint))
    return specs


def _il_eval_inputs(
    *,
    train_output: Path,
    source_checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    steps: int,
    il_save_freq: int,
    eval_every_steps: int,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
    specs: Sequence[tuple[int, str, Path]],
) -> dict[str, Any]:
    expected_processor_fingerprint = processor_fingerprint(source_checkpoint)
    checkpoint_records = []
    for step, label, checkpoint in specs:
        checkpoint_fingerprint = processor_fingerprint(checkpoint)
        if checkpoint_fingerprint != expected_processor_fingerprint:
            raise ValueError(
                f"IL checkpoint {label} changed the fixed normalizer: "
                f"expected={expected_processor_fingerprint} got={checkpoint_fingerprint}"
            )
        checkpoint_records.append(
            {
                "step": step,
                "label": label,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _digest_tree(checkpoint),
                "processor_fingerprint": checkpoint_fingerprint,
            }
        )
    return {
        "train_output": str(train_output.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _digest_tree(source_checkpoint),
        "processor_fingerprint": expected_processor_fingerprint,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_sha256": _digest_tree(dataset_root),
        "repo_id": repo_id,
        "steps": steps,
        "il_save_freq": il_save_freq,
        "eval_every_steps": eval_every_steps,
        "episodes": episodes,
        "batch_size": batch_size,
        "inference_steps": inference_steps,
        "policy_device": policy_device,
        "env_device": env_device,
        "env_type": env_type,
        "seed": seed,
        "checkpoints": checkpoint_records,
    }


def _il_eval_aggregate(path: Path) -> tuple[float, int]:
    payload = _read_json(path)
    return _eval_aggregate(payload, path=path)


def _il_eval_provenance(
    checkpoint: Path,
    *,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
) -> dict[str, Any]:
    resolved_checkpoint = _checkpoint_valid(checkpoint)
    return build_eval_provenance(
        resolved_checkpoint,
        episodes=episodes,
        batch_size=batch_size,
        inference_steps=inference_steps,
        policy_device=policy_device,
        env_device=env_device,
        env_type=env_type,
        seed=seed,
    )


def _il_eval_provenances(
    specs: Sequence[tuple[int, str, Path]],
    *,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    provenances: dict[str, dict[str, Any]] = {}
    for step, label, checkpoint in specs:
        expected_label = _il_label(step)
        if label != expected_label:
            raise ValueError(
                f"IL evaluation label disagrees with step {step}: {label!r} != {expected_label!r}"
            )
        if label in provenances:
            raise ValueError(f"duplicate IL evaluation label: {label}")
        provenances[label] = _il_eval_provenance(
            checkpoint,
            episodes=episodes,
            batch_size=batch_size,
            inference_steps=inference_steps,
            policy_device=policy_device,
            env_device=env_device,
            env_type=env_type,
            seed=seed,
        )
    return provenances


def _il_eval_point_dir(eval_output: Path, *, step: int, label: str) -> Path:
    expected_label = _il_label(step)
    if label != expected_label:
        raise ValueError(f"IL evaluation label disagrees with step {step}: {label!r} != {expected_label!r}")
    output_root = eval_output.resolve()
    eval_root = output_root / "eval"
    if eval_root.is_symlink():
        raise ValueError(f"refusing to use symlinked IL evaluation root: {eval_root}")
    point = eval_root / label
    if point.parent != eval_root:
        raise ValueError(f"IL evaluation point escapes its evaluation root: {point}")
    return point


def _remove_stale_il_eval_point(eval_output: Path, *, step: int, label: str) -> None:
    point = _il_eval_point_dir(eval_output, step=step, label=label)
    eval_root = point.parent
    if point.is_symlink():
        point.unlink()
        return
    if not point.exists():
        return
    resolved_point = point.resolve(strict=True)
    resolved_eval_root = eval_root.resolve(strict=True)
    if resolved_point.parent != resolved_eval_root:
        raise ValueError(
            f"refusing to remove IL evaluation point outside {resolved_eval_root}: {resolved_point}"
        )
    if point.is_dir():
        shutil.rmtree(point)
    else:
        point.unlink()


def _complete_il_eval_point(
    eval_dir: Path,
    *,
    expected_provenance: Mapping[str, Any],
) -> tuple[float, int, Path]:
    if eval_dir.is_symlink() or not eval_dir.is_dir():
        raise ValueError(f"IL evaluation point must be a local directory: {eval_dir}")
    eval_info = eval_dir / "eval_info.json"
    provenance_path = eval_dir / EVAL_PROVENANCE_FILENAME
    if eval_info.is_symlink() or provenance_path.is_symlink():
        raise ValueError(f"IL evaluation artifacts must not be symlinks: {eval_dir}")
    provenance = _read_json(provenance_path)
    if provenance != _jsonable(expected_provenance):
        raise ValueError(f"IL evaluation provenance does not match current inputs: {provenance_path}")
    success_rate, episodes = _il_eval_aggregate(eval_info)
    if episodes != expected_provenance["episodes"]:
        raise ValueError(
            f"IL evaluation used {episodes} episodes, expected {expected_provenance['episodes']}: {eval_info}"
        )
    return success_rate, episodes, eval_info.resolve()


def _run_newton_eval(
    *,
    policy_path: Path,
    output_dir: Path,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
    log_path: Path,
) -> Path:
    """Run one headless Newton evaluation and return its eval_info path."""

    if output_dir.exists():
        raise FileExistsError(f"IL evaluation output already exists: {output_dir}")
    checkpoint = _checkpoint_valid(policy_path)
    if episodes <= 0 or batch_size <= 0 or inference_steps <= 0:
        raise ValueError("IL evaluation episodes, batch_size, and inference_steps must be positive")
    if not policy_device or not env_device or not env_type:
        raise ValueError("IL evaluation devices and env_type must be nonempty")
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={checkpoint}",
        f"--policy.device={policy_device}",
        f"--policy.num_inference_steps={inference_steps}",
        f"--env.type={env_type}",
        f"--env.device={env_device}",
        f"--eval.n_episodes={episodes}",
        f"--eval.batch_size={batch_size}",
        "--eval.use_async_envs=false",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Newton IL evaluation failed with exit code {completed.returncode}; see {log_path}"
        )
    eval_info = output_dir / "eval_info.json"
    if not eval_info.is_file():
        raise RuntimeError(f"Newton IL evaluation did not write eval_info.json: {output_dir}")
    _il_eval_aggregate(eval_info)
    return eval_info.resolve()


def _collect_il_evaluations(
    *,
    eval_output: Path,
    specs: Sequence[tuple[int, str, Path]],
    expected_provenances: Mapping[str, Mapping[str, Any]],
) -> list[ILEvaluation]:
    evaluations: list[ILEvaluation] = []
    for step, label, checkpoint in specs:
        expected_provenance = expected_provenances.get(label)
        if expected_provenance is None:
            continue
        eval_dir = _il_eval_point_dir(eval_output, step=step, label=label)
        try:
            success_rate, episodes, eval_info = _complete_il_eval_point(
                eval_dir,
                expected_provenance=expected_provenance,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        evaluations.append(
            ILEvaluation(
                checkpoint=checkpoint.resolve(),
                label=label,
                step=step,
                success_rate=success_rate,
                episodes=episodes,
                eval_info=eval_info.resolve(),
            )
        )
    return evaluations


def _select_best_il_checkpoint(evaluations: Sequence[ILEvaluation]) -> ILEvaluation:
    if not evaluations:
        raise ValueError("no complete IL checkpoint evaluations found")
    return sorted(evaluations, key=lambda item: (-item.success_rate, item.step))[0]


def _select_offline_il_checkpoint(
    best_il: ILEvaluation,
    il_final_checkpoint: Path,
    *,
    use_final: bool,
) -> Path:
    """Choose the IL checkpoint that seeds offline RL."""

    return _checkpoint_valid(il_final_checkpoint if use_final else best_il.checkpoint)


def _il_eval_payload(selected: ILEvaluation) -> dict[str, Any]:
    return {
        "checkpoint": str(selected.checkpoint),
        "checkpoint_sha256": _digest_tree(selected.checkpoint),
        "eval_info": str(selected.eval_info),
        "eval_info_sha256": _digest_file(selected.eval_info),
        "episodes": selected.episodes,
        "label": selected.label,
        "step": selected.step,
        "success_rate": selected.success_rate,
    }


def _il_eval_valid(
    record: Mapping[str, Any],
    *,
    expected_inputs: Mapping[str, Any],
    specs: Sequence[tuple[int, str, Path]],
    expected_episodes: int,
) -> ILEvaluation:
    if record.get("status") != "complete":
        raise ValueError("IL evaluation stage is not complete")
    if record.get("inputs") != _jsonable(expected_inputs):
        raise ValueError("IL evaluation stage inputs do not match current dependencies")
    eval_output = Path(str(record.get("output_dir", ""))).resolve(strict=True)
    if expected_inputs.get("episodes") != expected_episodes:
        raise ValueError("IL evaluation expected episode count disagrees with stage inputs")
    expected_provenances = _il_eval_provenances(
        specs,
        episodes=expected_episodes,
        batch_size=expected_inputs["batch_size"],
        inference_steps=expected_inputs["inference_steps"],
        policy_device=expected_inputs["policy_device"],
        env_device=expected_inputs["env_device"],
        env_type=expected_inputs["env_type"],
        seed=expected_inputs["seed"],
    )
    evaluations = _collect_il_evaluations(
        eval_output=eval_output,
        specs=specs,
        expected_provenances=expected_provenances,
    )
    if len(evaluations) != len(specs):
        raise ValueError(f"IL evaluation stage is missing checkpoints: {len(evaluations)}/{len(specs)}")
    selected = _select_best_il_checkpoint(evaluations)
    expected_best = record.get("best")
    if expected_best != _il_eval_payload(selected):
        raise ValueError("IL evaluation best checkpoint disagrees with evaluation artifacts")
    return selected


def _run_il_evaluation_stage(
    *,
    eval_output: Path,
    specs: Sequence[tuple[int, str, Path]],
    expected_episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
    log_root: Path,
) -> ILEvaluation:
    """Evaluate every saved IL checkpoint, reusing completed points on resume."""

    eval_output.mkdir(parents=True, exist_ok=True)
    expected_provenances = _il_eval_provenances(
        specs,
        episodes=expected_episodes,
        batch_size=batch_size,
        inference_steps=inference_steps,
        policy_device=policy_device,
        env_device=env_device,
        env_type=env_type,
        seed=seed,
    )
    for step, label, checkpoint in specs:
        eval_dir = _il_eval_point_dir(eval_output, step=step, label=label)
        expected_provenance = expected_provenances[label]
        try:
            _complete_il_eval_point(eval_dir, expected_provenance=expected_provenance)
            continue
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if eval_dir.exists() or eval_dir.is_symlink():
                print(f"[IL eval] {label}: stale or incomplete ({exc}); rerunning")
                _remove_stale_il_eval_point(eval_output, step=step, label=label)
        eval_info = _run_newton_eval(
            policy_path=checkpoint,
            output_dir=eval_dir,
            episodes=expected_episodes,
            batch_size=batch_size,
            inference_steps=inference_steps,
            policy_device=policy_device,
            env_device=env_device,
            env_type=env_type,
            seed=seed,
            log_path=log_root / f"{label}.log",
        )
        expected_eval_info = (eval_dir / "eval_info.json").resolve(strict=True)
        if eval_info.resolve(strict=True) != expected_eval_info:
            raise RuntimeError(
                f"Newton IL evaluation {label} returned an unexpected eval_info path: {eval_info}"
            )
        success_rate, episodes = _il_eval_aggregate(eval_info)
        if episodes != expected_episodes:
            raise RuntimeError(
                f"Newton IL evaluation {label} used {episodes} episodes, expected {expected_episodes}"
            )
        if not math.isfinite(success_rate):
            raise RuntimeError(f"Newton IL evaluation {label} returned a non-finite success rate")
        write_eval_provenance(eval_dir, expected_provenance)
    evaluations = _collect_il_evaluations(
        eval_output=eval_output,
        specs=specs,
        expected_provenances=expected_provenances,
    )
    if len(evaluations) != len(specs):
        raise RuntimeError(f"IL evaluation is incomplete: {len(evaluations)}/{len(specs)} checkpoints")
    metrics_path = eval_output / "il_eval_metrics.jsonl"
    temporary = metrics_path.with_name(f".{metrics_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for evaluation in sorted(evaluations, key=lambda item: item.step):
            handle.write(
                json.dumps(
                    {
                        "step": evaluation.step,
                        "label": evaluation.label,
                        "eval/pc_success": evaluation.success_rate,
                        "eval/n_episodes": evaluation.episodes,
                        "checkpoint": str(evaluation.checkpoint),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"[IL eval] {evaluation.label}: pc_success={evaluation.success_rate:.2f}% "
                f"({evaluation.episodes} episodes)"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, metrics_path)
    return _select_best_il_checkpoint(evaluations)


def _rollout_stage_inputs(selected: BestCheckpoint) -> dict[str, Any]:
    return _selection_payload(selected)


def _merge_stage_inputs(
    *,
    base_dataset_root: Path,
    base_repo_id: str,
    base_summary: Path,
    rollout_dataset: Path,
    rollout_repo_id: str,
    rollout_summary: Path,
) -> dict[str, Any]:
    return {
        "base_dataset_root": str(base_dataset_root.resolve()),
        "base_dataset_sha256": _digest_tree(base_dataset_root),
        "base_repo_id": base_repo_id,
        "base_summary": str(base_summary.resolve()),
        "base_summary_sha256": _digest_file(base_summary.resolve(strict=True)),
        "rollout_dataset_root": str(rollout_dataset.resolve()),
        "rollout_dataset_sha256": _digest_tree(rollout_dataset),
        "rollout_repo_id": rollout_repo_id,
        "rollout_summary": str(rollout_summary.resolve()),
        "rollout_summary_sha256": _digest_file(rollout_summary.resolve(strict=True)),
    }


def _il_stage_inputs(*, source_checkpoint: Path, dataset_root: Path, repo_id: str) -> dict[str, Any]:
    return {
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _digest_tree(source_checkpoint),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_sha256": _digest_tree(dataset_root),
        "repo_id": repo_id,
    }


def _offline_stage_inputs(
    *, checkpoint: Path, dataset_root: Path, repo_id: str, summary: Path
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _digest_tree(checkpoint),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_sha256": _digest_tree(dataset_root),
        "repo_id": repo_id,
        "summary": str(summary.resolve()),
        "summary_sha256": _digest_file(summary.resolve(strict=True)),
    }


def _checkpoint_valid(path: Path | str) -> Path:
    root = Path(path).resolve(strict=True)
    if root.is_symlink():
        root = root.resolve(strict=True)
    if (root / "pretrained_model").is_dir():
        root = (root / "pretrained_model").resolve(strict=True)
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (root / name).is_file():
            raise ValueError(f"checkpoint is missing {name}: {root}")
    return root


def _canonical_dataset_valid(dataset_root: Path, summary_path: Path, repo_id: str) -> None:
    summary = _read_json(summary_path)
    if summary.get("complete") is not True:
        raise ValueError(f"dataset summary is not complete: {summary_path}")
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    allowed_fps = {60}
    if getattr(dataset.meta, "robot_type", None) == "franka_duo":
        allowed_fps.add(15)
    if dataset.fps not in allowed_fps:
        raise ValueError(
            "canonical RL dataset has unsupported fps for "
            f"robot_type={getattr(dataset.meta, 'robot_type', None)!r}; "
            f"expected one of {sorted(allowed_fps)}, got {dataset.fps}"
        )
    expected = summary.get("episodes_saved")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected != dataset.num_episodes:
        raise ValueError(
            f"dataset/summary episode count mismatch: dataset={dataset.num_episodes} summary={expected}"
        )
    if not all(key in dataset.features for key in ("next.reward", "next.done", "next.truncated")):
        raise ValueError(f"canonical RL fields are missing from {dataset_root}")


def _collection_valid(output_dir: Path, *, repo_id: str, expected_episodes: int) -> tuple[Path, Path]:
    dataset_root = (output_dir / "dataset").resolve(strict=True)
    summary_path = (output_dir / "collection_summary.json").resolve(strict=True)
    summary = _read_json(summary_path)
    if summary.get("complete") is not True:
        raise ValueError(f"rollout collection is incomplete: {summary_path}")
    if summary.get("repo_id") != repo_id:
        raise ValueError(f"rollout repo_id disagrees: expected={repo_id!r}")
    if summary.get("episodes_saved") != expected_episodes:
        raise ValueError(
            "rollout episode count disagrees: "
            f"expected={expected_episodes}, got={summary.get('episodes_saved')!r}"
        )
    _canonical_dataset_valid(dataset_root, summary_path, repo_id)
    return dataset_root, summary_path


def _next_attempt(stage_root: Path) -> tuple[int, Path]:
    stage_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in stage_root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    number = max(numbers, default=0) + 1
    return number, stage_root / f"attempt_{number:03d}"


def _command_for_module(module: str, *, swanlab: bool) -> list[str]:
    if swanlab:
        return ["uv", "run", "--with", "swanlab==0.9.4", "python", "-m", module]
    return [sys.executable, "-m", module]


def _run_logged(command: Sequence[str], *, log_path: Path, env: Mapping[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=Path.cwd(),
            env=merged_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"stage command failed with exit code {completed.returncode}; see {log_path}")


def _base_manifest(index: int, *, inputs: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _MANIFEST_VERSION,
        "round_index": index,
        "status": "running",
        "inputs": dict(inputs),
        "settings": dict(settings),
        "selection": None,
        "stages": {},
        "outputs": None,
    }


def _manifest_path(round_root: Path) -> Path:
    return round_root / "round_manifest.json"


def _load_or_create_manifest(
    round_root: Path,
    index: int,
    *,
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = _manifest_path(round_root)
    if path.exists():
        if not resume:
            raise FileExistsError(f"round already exists and resume is disabled: {round_root}")
        manifest = _read_json(path)
        if manifest.get("schema_version") != _MANIFEST_VERSION or manifest.get("round_index") != index:
            raise ValueError(f"round manifest schema/index mismatch: {path}")
        if manifest.get("inputs") != _jsonable(inputs):
            raise ValueError(f"round inputs differ from the existing manifest: {path}")
        existing_settings = manifest.get("settings")
        expected_settings = _jsonable(settings)
        # Manifests created before ``offline_use_il_final`` was introduced are
        # equivalent to the default best-periodic selection mode.
        if isinstance(existing_settings, Mapping) and "offline_use_il_final" not in existing_settings:
            existing_settings = {**existing_settings, "offline_use_il_final": False}
        if existing_settings != expected_settings:
            raise ValueError(f"round settings differ from the existing manifest: {path}")
        if manifest.get("status") != "complete":
            manifest["status"] = "running"
            _write_json(path, manifest)
        return manifest
    if round_root.exists() and any(round_root.iterdir()):
        raise FileExistsError(f"round directory contains untracked files: {round_root}")
    manifest = _base_manifest(index, inputs=inputs, settings=settings)
    _write_json(path, manifest)
    return manifest


def _record(manifest: dict[str, Any], round_root: Path, name: str, record: Mapping[str, Any]) -> None:
    stages = manifest.setdefault("stages", {})
    previous = stages.get(name)
    history: list[dict[str, Any]] = []
    if isinstance(previous, Mapping):
        previous_history = previous.get("history")
        if isinstance(previous_history, list):
            history.extend(item for item in previous_history if isinstance(item, dict))
        history.append({key: value for key, value in previous.items() if key != "history"})
    current = dict(record)
    if history:
        current["history"] = history
    stages[name] = current
    _write_json(_manifest_path(round_root), manifest)


def _record_planned(manifest: dict[str, Any], round_root: Path, name: str, record: Mapping[str, Any]) -> None:
    stages = manifest.get("stages")
    current = stages.get(name) if isinstance(stages, Mapping) else None
    comparable_current = (
        {key: value for key, value in current.items() if key != "history"}
        if isinstance(current, Mapping)
        else None
    )
    if comparable_current == dict(record):
        return
    _record(manifest, round_root, name, record)


def _stage_is_complete(manifest: Mapping[str, Any], name: str) -> bool:
    record = manifest.get("stages", {}).get(name) if isinstance(manifest.get("stages"), Mapping) else None
    return isinstance(record, Mapping) and record.get("status") == "complete"


def _complete_stage_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("round manifest has no stage records")
    record = stages.get(name)
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        raise ValueError(f"round stage is not complete: {name}")
    return record


def _require_stage_inputs(record: Mapping[str, Any], expected: Mapping[str, Any], *, stage: str) -> None:
    if record.get("inputs") != _jsonable(expected):
        raise ValueError(f"{stage} stage inputs do not match its current dependencies")


def _as_path(record: Mapping[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"stage record is missing {key}")
    return Path(value)


def _round_settings(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "output_root",
        "base_dataset_root",
        "base_repo_id",
        "base_summary",
        "il_checkpoint",
        "source_offline_run",
        "rounds",
        "resume",
        "dry_run",
    }
    return {key: _jsonable(value) for key, value in vars(args).items() if key not in excluded}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--output_root", type=Path, required=True)
    parser.add_argument("--base-dataset-root", "--base_dataset_root", type=Path, required=True)
    parser.add_argument("--base-repo-id", "--base_repo_id", required=True)
    parser.add_argument("--base-summary", "--base_summary", type=Path, required=True)
    parser.add_argument("--il-checkpoint", "--il_checkpoint", type=Path, required=True)
    parser.add_argument("--source-offline-run", "--source_offline_run", type=Path, required=True)
    parser.add_argument("--source-eval-episodes", "--source_eval_episodes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--final-rollout-merge",
        "--final_rollout_merge",
        action="store_true",
        help=(
            "After the requested full rounds, collect once from the new best sync checkpoint "
            "and publish the next cumulative dataset without starting another IL/offline stage."
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=16)
    parser.add_argument("--episode-length", "--episode_length", type=int, default=930)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-device", "--sim_device", default="cuda:0")
    parser.add_argument(
        "--inference-steps",
        "--inference_steps",
        type=int,
        default=10,
        help="Diffusion steps for collection; keep this equal to the evaluated offline policy.",
    )
    parser.add_argument("--collection-seed", "--collection_seed", type=int, default=2000)
    parser.add_argument("--canonical-task", "--canonical_task")
    parser.add_argument("--il-steps", "--il_steps", type=int, default=50_000)
    parser.add_argument("--il-batch-size", "--il_batch_size", type=int, default=256)
    parser.add_argument("--il-num-workers", "--il_num_workers", type=int, default=4)
    parser.add_argument("--il-save-freq", "--il_save_freq", type=int, default=5_000)
    parser.add_argument("--il-learning-rate", "--il_learning_rate", type=float, default=1e-5)
    parser.add_argument("--il-seed", "--il_seed", type=int, default=1000)
    parser.add_argument(
        "--il-eval-every-steps",
        "--il_eval_every_steps",
        type=int,
        default=5_000,
        help="Evaluate every saved IL checkpoint at this step interval in Newton.",
    )
    parser.add_argument("--il-eval-episodes", "--il_eval_episodes", type=int, default=100)
    parser.add_argument("--il-eval-batch-size", "--il_eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--il-eval-inference-steps",
        "--il_eval_inference_steps",
        type=int,
        default=None,
        help="Diffusion inference steps for IL evaluation (defaults to --inference-steps).",
    )
    parser.add_argument("--il-eval-env-type", "--il_eval_env_type", default="moya_newton")
    parser.add_argument("--offline-iql-steps", "--offline_iql_steps", type=int, default=100_000)
    parser.add_argument("--offline-dynamics-steps", "--offline_dynamics_steps", type=int, default=10_000)
    parser.add_argument("--offline-batch-size", "--offline_batch_size", type=int, default=32)
    parser.add_argument("--offline-actor-lr", "--offline_actor_lr", type=float, default=1e-6)
    parser.add_argument("--offline-ppo-epochs", "--offline_ppo_epochs", type=int, default=1)
    parser.add_argument("--offline-inference-steps", "--offline_inference_steps", type=int, default=10)
    parser.add_argument("--offline-sync-target", "--offline_sync_target", type=int, default=50)
    parser.add_argument(
        "--offline-use-il-final",
        "--offline_use_il_final",
        action="store_true",
        help="Seed offline RL from the final IL checkpoint instead of the best periodic IL evaluation.",
    )
    parser.add_argument("--offline-eval-every-syncs", "--offline_eval_every_syncs", type=int, default=5)
    parser.add_argument("--offline-eval-episodes", "--offline_eval_episodes", type=int, default=100)
    parser.add_argument("--offline-eval-batch-size", "--offline_eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--offline-probability-sigma-min", "--offline_probability_sigma_min", type=float, default=0.1
    )
    parser.add_argument("--offline-seed", "--offline_seed", type=int, default=0)
    parser.add_argument("--offline-amq-eval-interval", "--offline_amq_eval_interval", type=int, default=50)
    parser.add_argument(
        "--offline-amq-rollout-horizon", "--offline_amq_rollout_horizon", type=int, default=20
    )
    parser.add_argument(
        "--offline-amq-relative-margin", "--offline_amq_relative_margin", type=float, default=0.05
    )
    parser.add_argument(
        "--offline-debug",
        "--offline_debug",
        action="store_true",
        help="Include verbose per-denoising-step diagnostics in offline RL metrics.",
    )
    parser.add_argument("--no-offline-amq", dest="offline_amq_enabled", action="store_false")
    parser.set_defaults(offline_amq_enabled=True)
    parser.add_argument("--swanlab-project", "--swanlab_project", default="moya-rl100")
    parser.add_argument(
        "--swanlab-mode",
        "--swanlab_mode",
        choices=("online", "offline", "local", "disabled"),
        default="online",
    )
    parser.add_argument("--swanlab-strict", "--swanlab_strict", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "rounds",
        "episodes",
        "num_envs",
        "episode_length",
        "inference_steps",
        "source_eval_episodes",
        "il_steps",
        "il_batch_size",
        "il_num_workers",
        "il_save_freq",
        "offline_iql_steps",
        "il_eval_every_steps",
        "il_eval_episodes",
        "il_eval_batch_size",
        "offline_dynamics_steps",
        "offline_batch_size",
        "offline_ppo_epochs",
        "offline_inference_steps",
        "offline_sync_target",
        "offline_eval_every_syncs",
        "offline_eval_episodes",
        "offline_eval_batch_size",
        "offline_amq_eval_interval",
        "offline_amq_rollout_horizon",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "il_learning_rate",
        "offline_actor_lr",
        "offline_probability_sigma_min",
        "offline_amq_relative_margin",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    if args.il_eval_inference_steps is not None and (
        isinstance(args.il_eval_inference_steps, bool)
        or not isinstance(args.il_eval_inference_steps, int)
        or args.il_eval_inference_steps <= 0
    ):
        raise ValueError("il_eval_inference_steps must be positive when provided")
    if args.il_eval_every_steps % args.il_save_freq != 0:
        raise ValueError("il_eval_every_steps must be a multiple of il_save_freq")
    if not isinstance(args.il_eval_env_type, str) or not args.il_eval_env_type.strip():
        raise ValueError("il_eval_env_type must be nonempty")
    for name in ("base_repo_id",):
        if not isinstance(getattr(args, name), str) or not getattr(args, name).strip():
            raise ValueError(f"{name} must be nonempty")
    if args.swanlab_mode != "disabled" and (
        not isinstance(args.swanlab_project, str) or not args.swanlab_project.strip()
    ):
        raise ValueError("swanlab_project must be nonempty when tracking is enabled")
    if args.offline_sync_target % args.offline_eval_every_syncs != 0:
        raise ValueError("offline_sync_target must be divisible by offline_eval_every_syncs")
    if not Path(args.base_dataset_root).is_dir() or not Path(args.base_summary).is_file():
        raise ValueError("base dataset and summary must exist")
    if args.canonical_task is None:
        task = _read_json(Path(args.base_summary)).get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("canonical_task is required when the base summary has no task")
        args.canonical_task = task
    elif not isinstance(args.canonical_task, str) or not args.canonical_task.strip():
        raise ValueError("canonical_task must be nonempty")
    _checkpoint_valid(Path(args.il_checkpoint))
    Path(args.source_offline_run).resolve(strict=True)


def _collection_command(
    args: argparse.Namespace, checkpoint: BestCheckpoint, output: Path, seed: int, repo_id: str
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "RL.cli.collect_moya_il",
        f"--checkpoint={checkpoint.checkpoint}",
        f"--output-dir={output}",
        f"--repo-id={repo_id}",
        f"--episodes={args.episodes}",
        f"--num-envs={args.num_envs}",
        f"--episode-length={args.episode_length}",
        f"--device={args.device}",
        f"--sim-device={args.sim_device}",
        f"--inference-steps={args.inference_steps}",
        f"--seed={seed}",
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def _il_command(
    args: argparse.Namespace,
    source_il: Path,
    dataset_root: Path,
    repo_id: str,
    output: Path,
    *,
    seed: int,
    run_name: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "RL.cli.train_il_warmstart",
        f"--checkpoint={source_il}",
        f"--dataset-root={dataset_root}",
        f"--repo-id={repo_id}",
        f"--output-dir={output}",
        f"--steps={args.il_steps}",
        f"--batch-size={args.il_batch_size}",
        f"--num-workers={args.il_num_workers}",
        f"--save-freq={args.il_save_freq}",
        f"--learning-rate={args.il_learning_rate}",
        f"--swanlab-project={args.swanlab_project}",
        f"--swanlab-mode={args.swanlab_mode}",
        f"--swanlab-run-name={run_name}",
        f"--seed={seed}",
        f"--device={args.device}",
        f"--eval-every-steps={args.il_eval_every_steps}",
        f"--eval-episodes={args.il_eval_episodes}",
        f"--eval-batch-size={args.il_eval_batch_size}",
        f"--eval-inference-steps={args.il_eval_inference_steps or args.inference_steps}",
        f"--eval-env-type={args.il_eval_env_type}",
        f"--eval-env-device={args.sim_device}",
    ]
    if args.smoke:
        command[command.index(f"--steps={args.il_steps}")] = "--steps=2"
        command[command.index(f"--batch-size={args.il_batch_size}")] = "--batch-size=2"
    return command


def _offline_command(
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    summary: Path,
    output: Path,
    *,
    seed: int,
    run_name: str,
) -> list[str]:
    command = _command_for_module("RL.cli.train_offline", swanlab=args.swanlab_mode != "disabled")
    amq_enabled = args.offline_amq_enabled and not args.smoke
    sync_interval = 0 if amq_enabled else 1
    command.extend(
        [
            f"--checkpoint={checkpoint}",
            f"--dataset-root={dataset_root}",
            f"--repo-id={repo_id}",
            f"--summary={summary}",
            f"--output-dir={output}",
            f"--device={args.device}",
            f"--iql-steps={args.offline_iql_steps}",
            "--actor-steps=0",
            f"--inference-steps={args.offline_inference_steps}",
            f"--batch-size={args.offline_batch_size}",
            f"--actor-lr={args.offline_actor_lr}",
            f"--ppo-epochs={args.offline_ppo_epochs}",
            f"--probability-sigma-min={args.offline_probability_sigma_min}",
            f"--old-policy-sync-interval={sync_interval}",
            f"--old-policy-sync-target={args.offline_sync_target}",
            f"--eval-every-old-policy-syncs={args.offline_eval_every_syncs}",
            f"--eval-episodes={args.offline_eval_episodes}",
            f"--eval-batch-size={args.offline_eval_batch_size}",
            f"--eval-env-device={args.sim_device}",
            f"--seed={seed}",
            f"--swanlab-project={args.swanlab_project}",
            f"--swanlab-mode={args.swanlab_mode}",
            f"--swanlab-run-name={run_name}",
        ]
    )
    if amq_enabled:
        command.extend(
            (
                "--amq-enabled",
                f"--dynamics-steps={args.offline_dynamics_steps}",
                f"--amq-min-dynamics-updates={args.offline_dynamics_steps}",
                f"--amq-eval-interval={args.offline_amq_eval_interval}",
                f"--amq-rollout-horizon={args.offline_amq_rollout_horizon}",
                f"--amq-relative-margin={args.offline_amq_relative_margin}",
            )
        )
    if args.swanlab_strict:
        command.append("--swanlab-strict")
    if args.offline_debug:
        command.append("--debug")
    if args.smoke:
        command.append("--smoke")
    return command


def _offline_valid(
    output: Path,
    *,
    sync_target: int,
    eval_every_syncs: int,
    eval_episodes: int,
    expected_inputs: Mapping[str, Any] | None = None,
) -> BestCheckpoint:
    final_root = output / "checkpoints" / "final"
    _checkpoint_valid(final_root / "pretrained_model")
    try:
        _load_rl_manifest(final_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"offline final checkpoint manifest is invalid: {output}: {exc}") from exc
    if not (output / "metrics.jsonl").is_file():
        raise ValueError(f"offline run is missing metrics.jsonl: {output}")
    if expected_inputs is not None:
        provenance_path = output / "checkpoints" / "final" / "provenance.json"
        provenance = _read_json(provenance_path)
        for provenance_key, input_key in (
            ("input_checkpoint", "checkpoint"),
            ("dataset_root", "dataset_root"),
            ("dataset_repo_id", "repo_id"),
            ("dataset_summary_path", "summary"),
        ):
            if provenance.get(provenance_key) != expected_inputs.get(input_key):
                raise ValueError(f"offline provenance {provenance_key} disagrees with stage inputs")
        if provenance.get("dataset_summary_hash") != expected_inputs.get("summary_sha256"):
            raise ValueError("offline provenance dataset summary hash disagrees")
    expected_syncs = range(eval_every_syncs, sync_target + 1, eval_every_syncs)
    for sync_count in expected_syncs:
        label = f"sync_{sync_count:03d}"
        if not (output / "eval" / label / "eval_info.json").is_file():
            raise ValueError(f"offline run is missing periodic evaluation {label}: {output}")
        _checkpoint_from_label(output.resolve(), label)
        _, episodes = _eval_aggregate(
            _read_json(output / "eval" / label / "eval_info.json"),
            path=output / "eval" / label / "eval_info.json",
        )
        if episodes != eval_episodes:
            raise ValueError(f"offline evaluation {label} used {episodes} episodes, expected {eval_episodes}")
    return select_best_checkpoint(output, required_episodes=eval_episodes, sync_only=True)


def _il_stage_valid(
    record: Mapping[str, Any], *, source_checkpoint: Path, dataset_root: Path, repo_id: str
) -> Path:
    checkpoint = _checkpoint_valid(Path(str(record.get("checkpoint", ""))))
    stage_manifest_value = record.get("stage_manifest")
    if not isinstance(stage_manifest_value, str) or not stage_manifest_value:
        raise ValueError("IL stage record is missing stage_manifest")
    stage_manifest = _read_json(Path(stage_manifest_value))
    if stage_manifest.get("complete") is not True:
        raise ValueError("IL stage manifest is not complete")
    if Path(str(stage_manifest.get("checkpoint"))).resolve() != checkpoint:
        raise ValueError("IL stage manifest checkpoint disagrees with the round manifest")
    if Path(str(stage_manifest.get("source_checkpoint"))).resolve() != source_checkpoint.resolve():
        raise ValueError("IL stage manifest source checkpoint disagrees")
    if Path(str(stage_manifest.get("dataset_root"))).resolve() != dataset_root.resolve():
        raise ValueError("IL stage manifest dataset root disagrees")
    if stage_manifest.get("repo_id") != repo_id:
        raise ValueError("IL stage manifest repo_id disagrees")
    expected_fingerprint = processor_fingerprint(source_checkpoint)
    if processor_fingerprint(checkpoint) != expected_fingerprint:
        raise ValueError("IL checkpoint did not preserve the source normalizer")
    if stage_manifest.get("processor_fingerprint") != expected_fingerprint:
        raise ValueError("IL stage processor fingerprint disagrees with checkpoint artifacts")
    return checkpoint


def _claim_completed_il_output(
    record: Mapping[str, Any],
    *,
    source_checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    expected_steps: int,
) -> tuple[dict[str, Any], Path]:
    """Recover a finished IL process whose parent stage record was not written.

    A process can finish checkpointing and then be interrupted before the
    orchestrator persists ``il_stage.json``.  We only claim such an output when
    every durable contract is present and matches the current stage inputs.
    """

    _require_stage_inputs(
        record,
        _il_stage_inputs(
            source_checkpoint=source_checkpoint,
            dataset_root=dataset_root,
            repo_id=repo_id,
        ),
        stage="il",
    )
    output_dir = Path(str(record.get("output_dir", ""))).resolve(strict=True)
    checkpoint = _checkpoint_valid(output_dir / "checkpoints" / "last")
    step_path = output_dir / "checkpoints" / "last" / "training_state" / "training_step.json"
    step_payload = _read_json(step_path.resolve(strict=True))
    step = step_payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_steps:
        raise ValueError(f"completed IL output has step={step!r}, expected {expected_steps}: {output_dir}")
    train_config = _read_json((checkpoint / "train_config.json").resolve(strict=True))
    if Path(str(train_config.get("output_dir", ""))).resolve() != output_dir:
        raise ValueError("completed IL train_config output_dir disagrees")
    if train_config.get("steps") != expected_steps:
        raise ValueError("completed IL train_config steps disagree")
    if train_config.get("preserve_pretrained_processor_stats") is not True:
        raise ValueError("completed IL output did not enable fixed processor statistics")
    if train_config.get("resume") is not False:
        raise ValueError("completed IL output unexpectedly used resume=true")
    dataset_config = train_config.get("dataset")
    if not isinstance(dataset_config, Mapping):
        raise ValueError("completed IL train_config is missing dataset config")
    if dataset_config.get("repo_id") != repo_id:
        raise ValueError("completed IL train_config repo_id disagrees")
    if Path(str(dataset_config.get("root", ""))).resolve() != dataset_root.resolve():
        raise ValueError("completed IL train_config dataset root disagrees")
    policy_config = train_config.get("policy")
    if not isinstance(policy_config, Mapping):
        raise ValueError("completed IL train_config is missing policy config")
    if Path(str(policy_config.get("pretrained_path", ""))).resolve() != source_checkpoint.resolve():
        raise ValueError("completed IL train_config source checkpoint disagrees")
    expected_fingerprint = processor_fingerprint(source_checkpoint)
    if processor_fingerprint(checkpoint) != expected_fingerprint:
        raise ValueError("completed IL output did not preserve the source normalizer")

    stage_manifest_value = record.get("stage_manifest")
    if not isinstance(stage_manifest_value, str) or not stage_manifest_value:
        raise ValueError("IL stage record is missing stage_manifest")
    stage_manifest = Path(stage_manifest_value)
    if stage_manifest.exists():
        existing = _read_json(stage_manifest)
        if existing.get("complete") is not True:
            raise ValueError("existing IL stage manifest is incomplete")
        if Path(str(existing.get("checkpoint", ""))).resolve() != checkpoint:
            raise ValueError("existing IL stage manifest checkpoint disagrees")
        if existing.get("processor_fingerprint") != expected_fingerprint:
            raise ValueError("existing IL stage manifest fingerprint disagrees")
        existing_record = {
            **dict(record),
            "status": "complete",
            "checkpoint": str(checkpoint),
        }
        _il_stage_valid(
            existing_record,
            source_checkpoint=source_checkpoint,
            dataset_root=dataset_root,
            repo_id=repo_id,
        )
        return existing_record, checkpoint
    _write_json(
        stage_manifest,
        {
            "complete": True,
            "checkpoint": str(checkpoint),
            "command": list(record.get("command", [])),
            "dataset_root": str(dataset_root.resolve()),
            "processor_fingerprint": expected_fingerprint,
            "repo_id": repo_id,
            "source_checkpoint": str(source_checkpoint.resolve()),
        },
    )
    completed_record = {
        **dict(record),
        "status": "complete",
        "checkpoint": str(checkpoint),
    }
    _il_stage_valid(
        completed_record,
        source_checkpoint=source_checkpoint,
        dataset_root=dataset_root,
        repo_id=repo_id,
    )
    return completed_record, checkpoint


def _invalidate_stage(manifest: dict[str, Any], round_root: Path, name: str, error: Exception) -> None:
    current = manifest.get("stages", {}).get(name)
    record = dict(current) if isinstance(current, Mapping) else {}
    record.update({"status": "invalid", "error": str(error)})
    _record(manifest, round_root, name, record)


def _validate_complete_round(
    manifest: Mapping[str, Any],
    *,
    source_offline_run: Path,
    source_eval_episodes: int,
    collection_episodes: int,
    sync_target: int,
    eval_every_syncs: int,
    eval_episodes: int,
    il_eval_every_steps: int = 5_000,
    il_eval_episodes: int = 100,
    il_eval_batch_size: int = 16,
    il_eval_inference_steps: int = 10,
    il_eval_env_type: str = "moya_newton",
    il_eval_seed: int = 101_000,
    il_save_freq: int = 5_000,
    il_steps: int = 50_000,
    device: str = "cuda",
    sim_device: str = "cuda:0",
    offline_use_il_final: bool = False,
) -> None:
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, Mapping):
        raise ValueError("complete round manifest is missing inputs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("complete round manifest is missing outputs")

    selected = select_best_checkpoint(
        source_offline_run,
        required_episodes=source_eval_episodes,
        sync_only=True,
    )
    if manifest.get("selection") != _selection_payload(selected):
        raise ValueError("complete round source checkpoint selection changed")

    rollout_record = _complete_stage_record(manifest, "rollout")
    _require_stage_inputs(rollout_record, _rollout_stage_inputs(selected), stage="rollout")
    rollout_repo_id = str(rollout_record.get("repo_id", ""))
    rollout_dataset, rollout_summary = _collection_valid(
        Path(str(rollout_record.get("output_dir", ""))),
        repo_id=rollout_repo_id,
        expected_episodes=collection_episodes,
    )

    merge_record = _complete_stage_record(manifest, "merge")
    expected_merge_inputs = _merge_stage_inputs(
        base_dataset_root=Path(str(manifest_inputs.get("base_dataset_root", ""))),
        base_repo_id=str(manifest_inputs.get("base_repo_id", "")),
        base_summary=Path(str(manifest_inputs.get("base_summary", ""))),
        rollout_dataset=rollout_dataset,
        rollout_repo_id=rollout_repo_id,
        rollout_summary=rollout_summary,
    )
    _require_stage_inputs(merge_record, expected_merge_inputs, stage="merge")
    repo_id = str(merge_record.get("repo_id", ""))
    dataset_root = Path(str(merge_record.get("dataset_root", "")))
    summary = Path(str(merge_record.get("summary", "")))
    _canonical_dataset_valid(dataset_root, summary, repo_id)

    il_record = _complete_stage_record(manifest, "il")
    source_il = Path(str(manifest_inputs.get("il_checkpoint", "")))
    _require_stage_inputs(
        il_record,
        _il_stage_inputs(
            source_checkpoint=source_il,
            dataset_root=dataset_root,
            repo_id=repo_id,
        ),
        stage="il",
    )
    il_final_checkpoint = _il_stage_valid(
        il_record,
        source_checkpoint=source_il,
        dataset_root=dataset_root,
        repo_id=repo_id,
    )

    il_eval_record = _complete_stage_record(manifest, "il_eval")
    il_specs = _il_checkpoint_specs(
        Path(str(il_record.get("output_dir", ""))),
        steps=il_steps,
        save_freq=il_save_freq,
        eval_every_steps=il_eval_every_steps,
    )
    il_eval_inputs = _il_eval_inputs(
        train_output=Path(str(il_record.get("output_dir", ""))),
        source_checkpoint=source_il,
        dataset_root=dataset_root,
        repo_id=repo_id,
        steps=il_steps,
        il_save_freq=il_save_freq,
        eval_every_steps=il_eval_every_steps,
        episodes=il_eval_episodes,
        batch_size=il_eval_batch_size,
        inference_steps=il_eval_inference_steps,
        policy_device=device,
        env_device=sim_device,
        env_type=il_eval_env_type,
        seed=il_eval_seed,
        specs=il_specs,
    )
    best_il = _il_eval_valid(
        il_eval_record,
        expected_inputs=il_eval_inputs,
        specs=il_specs,
        expected_episodes=il_eval_episodes,
    )
    offline_il_checkpoint = _select_offline_il_checkpoint(
        best_il,
        il_final_checkpoint,
        use_final=offline_use_il_final,
    )

    offline_record = _complete_stage_record(manifest, "offline")
    _require_stage_inputs(
        offline_record,
        _offline_stage_inputs(
            checkpoint=offline_il_checkpoint,
            dataset_root=dataset_root,
            repo_id=repo_id,
            summary=summary,
        ),
        stage="offline",
    )
    offline_run = Path(str(offline_record.get("output_dir", "")))
    best = _offline_valid(
        offline_run,
        sync_target=sync_target,
        eval_every_syncs=eval_every_syncs,
        eval_episodes=eval_episodes,
        expected_inputs=_offline_stage_inputs(
            checkpoint=offline_il_checkpoint,
            dataset_root=dataset_root,
            repo_id=repo_id,
            summary=summary,
        ),
    )
    expected_outputs = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "summary": str(summary),
        "il_checkpoint": str(offline_il_checkpoint),
        "il_final_checkpoint": str(il_final_checkpoint),
        "il_best_label": best_il.label,
        "il_best_success_rate": best_il.success_rate,
        "offline_run": str(offline_run.resolve()),
        "best_checkpoint": str(best.checkpoint),
        "best_label": best.label,
        "best_success_rate": best.success_rate,
    }
    if "offline_init_selection" in outputs or offline_use_il_final:
        expected_outputs["offline_init_selection"] = "il_final" if offline_use_il_final else "il_best_eval"
    if outputs != expected_outputs:
        raise ValueError("complete round outputs do not match its completed stages")


def _validate_collection_merge_round(
    manifest: Mapping[str, Any],
    *,
    source_offline_run: Path,
    source_eval_episodes: int,
    collection_episodes: int,
) -> None:
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, Mapping):
        raise ValueError("complete collection/merge round manifest is missing inputs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("complete collection/merge round manifest is missing outputs")

    selected = select_best_checkpoint(
        source_offline_run,
        required_episodes=source_eval_episodes,
        sync_only=True,
    )
    if manifest.get("selection") != _selection_payload(selected):
        raise ValueError("complete collection/merge source checkpoint selection changed")

    rollout_record = _complete_stage_record(manifest, "rollout")
    _require_stage_inputs(rollout_record, _rollout_stage_inputs(selected), stage="rollout")
    rollout_repo_id = str(rollout_record.get("repo_id", ""))
    rollout_dataset, rollout_summary = _collection_valid(
        Path(str(rollout_record.get("output_dir", ""))),
        repo_id=rollout_repo_id,
        expected_episodes=collection_episodes,
    )

    merge_record = _complete_stage_record(manifest, "merge")
    _require_stage_inputs(
        merge_record,
        _merge_stage_inputs(
            base_dataset_root=Path(str(manifest_inputs.get("base_dataset_root", ""))),
            base_repo_id=str(manifest_inputs.get("base_repo_id", "")),
            base_summary=Path(str(manifest_inputs.get("base_summary", ""))),
            rollout_dataset=rollout_dataset,
            rollout_repo_id=rollout_repo_id,
            rollout_summary=rollout_summary,
        ),
        stage="merge",
    )
    repo_id = str(merge_record.get("repo_id", ""))
    dataset_root = Path(str(merge_record.get("dataset_root", "")))
    summary = Path(str(merge_record.get("summary", "")))
    _canonical_dataset_valid(dataset_root, summary, repo_id)
    expected_outputs = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "summary": str(summary),
        "source_checkpoint": str(selected.checkpoint),
        "source_label": selected.label,
        "source_success_rate": selected.success_rate,
    }
    if outputs != expected_outputs:
        raise ValueError("collection/merge outputs do not match its completed stages")


def _run_one_round(
    args: argparse.Namespace,
    *,
    round_index: int,
    base_dataset_root: Path,
    base_repo_id: str,
    base_summary: Path,
    il_checkpoint: Path,
    source_offline_run: Path,
    source_eval_episodes: int | None = None,
    stop_after_merge: bool = False,
) -> dict[str, Any]:
    if source_eval_episodes is None:
        source_eval_episodes = args.source_eval_episodes
    round_root = Path(args.output_root).resolve() / f"round_{round_index:03d}"
    inputs = {
        "base_dataset_root": str(base_dataset_root.resolve()),
        "base_repo_id": base_repo_id,
        "base_summary": str(base_summary.resolve()),
        "il_checkpoint": str(il_checkpoint.resolve()),
        "source_offline_run": str(source_offline_run.resolve()),
        "source_eval_episodes": source_eval_episodes,
        "stop_after_merge": stop_after_merge,
    }
    manifest = _load_or_create_manifest(
        round_root,
        round_index,
        inputs=inputs,
        settings=_round_settings(args),
        resume=args.resume,
    )
    if manifest.get("status") == "complete":
        try:
            if stop_after_merge:
                _validate_collection_merge_round(
                    manifest,
                    source_offline_run=source_offline_run,
                    source_eval_episodes=source_eval_episodes,
                    collection_episodes=args.episodes,
                )
            else:
                _validate_complete_round(
                    manifest,
                    source_offline_run=source_offline_run,
                    source_eval_episodes=source_eval_episodes,
                    collection_episodes=args.episodes,
                    sync_target=args.offline_sync_target,
                    eval_every_syncs=args.offline_eval_every_syncs,
                    eval_episodes=args.offline_eval_episodes,
                    il_eval_every_steps=args.il_eval_every_steps,
                    il_eval_episodes=args.il_eval_episodes,
                    il_eval_batch_size=args.il_eval_batch_size,
                    il_eval_inference_steps=args.il_eval_inference_steps or args.inference_steps,
                    il_eval_env_type=args.il_eval_env_type,
                    il_eval_seed=args.il_seed + round_index * 100_000,
                    il_save_freq=args.il_save_freq,
                    il_steps=args.il_steps,
                    device=args.device,
                    sim_device=args.sim_device,
                    offline_use_il_final=args.offline_use_il_final,
                )
            return manifest
        except Exception as exc:
            manifest["status"] = "running"
            manifest["outputs"] = None
            manifest["completion_validation_error"] = str(exc)
            _write_json(_manifest_path(round_root), manifest)

    selected = select_best_checkpoint(
        source_offline_run,
        required_episodes=source_eval_episodes,
        sync_only=True,
    )
    selection_payload = _selection_payload(selected)
    if manifest.get("selection") is None:
        manifest["selection"] = selection_payload
        _write_json(_manifest_path(round_root), manifest)
    elif manifest.get("selection") != selection_payload:
        raise ValueError(
            f"source offline evaluation results changed after this round was created: {source_offline_run}"
        )

    # Stage 1: rollout collection.
    rollout_repo = f"local/moya-rl100-round-{round_index:03d}-rollout"
    rollout_inputs = _rollout_stage_inputs(selected)
    rollout_record = manifest.get("stages", {}).get("rollout")
    rollout_valid = False
    if _stage_is_complete(manifest, "rollout") and isinstance(rollout_record, Mapping):
        try:
            _require_stage_inputs(rollout_record, rollout_inputs, stage="rollout")
            rollout_output = Path(str(rollout_record["output_dir"]))
            rollout_repo = str(rollout_record["repo_id"])
            rollout_dataset, rollout_summary = _collection_valid(
                rollout_output,
                repo_id=rollout_repo,
                expected_episodes=args.episodes,
            )
            rollout_valid = True
        except Exception as exc:
            _invalidate_stage(manifest, round_root, "rollout", exc)
    if not rollout_valid:
        attempt_no, attempt = _next_attempt(round_root / "rollout")
        rollout_output = attempt / "collection"
        command = _collection_command(
            args, selected, rollout_output, args.collection_seed + round_index * 100_000, rollout_repo
        )
        planned = {
            "status": "planned",
            "attempt": attempt_no,
            "command": command,
            "inputs": rollout_inputs,
            "output_dir": str(rollout_output),
            "repo_id": rollout_repo,
        }
        if args.dry_run:
            _record_planned(manifest, round_root, "rollout", planned)
            return manifest
        try:
            _run_logged(command, log_path=attempt / "collect.log")
            rollout_dataset, rollout_summary = _collection_valid(
                rollout_output,
                repo_id=rollout_repo,
                expected_episodes=args.episodes,
            )
        except BaseException as exc:
            _record(manifest, round_root, "rollout", {**planned, "status": "failed", "error": str(exc)})
            raise
        rollout_record = {
            **planned,
            "status": "complete",
            "dataset_root": str(rollout_dataset),
            "summary": str(rollout_summary),
        }
        _record(manifest, round_root, "rollout", rollout_record)

    # Stage 2: canonical merge.
    merged_repo = f"local/moya-rl100-round-{round_index:03d}-merged"
    merge_inputs = _merge_stage_inputs(
        base_dataset_root=base_dataset_root,
        base_repo_id=base_repo_id,
        base_summary=base_summary,
        rollout_dataset=rollout_dataset,
        rollout_repo_id=rollout_repo,
        rollout_summary=rollout_summary,
    )
    merge_record = manifest.get("stages", {}).get("merge")
    merge_valid = False
    if _stage_is_complete(manifest, "merge") and isinstance(merge_record, Mapping):
        try:
            _require_stage_inputs(merge_record, merge_inputs, stage="merge")
            merge_output = Path(str(merge_record["output_dir"]))
            merged_dataset = Path(str(merge_record["dataset_root"]))
            merged_summary = Path(str(merge_record["summary"]))
            merged_repo = str(merge_record["repo_id"])
            _canonical_dataset_valid(merged_dataset, merged_summary, merged_repo)
            merge_valid = True
        except Exception as exc:
            _invalidate_stage(manifest, round_root, "merge", exc)
    if not merge_valid:
        attempt_no, attempt = _next_attempt(round_root / "merge")
        merge_output = attempt / "merged"
        planned = {
            "status": "planned",
            "attempt": attempt_no,
            "inputs": merge_inputs,
            "output_dir": str(merge_output),
            "repo_id": merged_repo,
        }
        if args.dry_run:
            _record_planned(manifest, round_root, "merge", planned)
            return manifest
        try:
            merge_lerobot_v3_datasets(
                merge_output,
                output_repo_id=merged_repo,
                base=LeRobotV3MergeSource(base_dataset_root, base_repo_id, base_summary),
                rollout=LeRobotV3MergeSource(rollout_dataset, rollout_repo, rollout_summary),
                canonical_task=args.canonical_task,
            )
            merged_dataset = (merge_output / "dataset").resolve(strict=True)
            merged_summary = (merge_output / "collection_summary.json").resolve(strict=True)
            _canonical_dataset_valid(merged_dataset, merged_summary, merged_repo)
        except BaseException as exc:
            _record(manifest, round_root, "merge", {**planned, "status": "failed", "error": str(exc)})
            raise
        merge_record = {
            **planned,
            "status": "complete",
            "dataset_root": str(merged_dataset),
            "summary": str(merged_summary),
        }
        _record(manifest, round_root, "merge", merge_record)

    if stop_after_merge:
        manifest["outputs"] = {
            "dataset_root": str(merged_dataset),
            "repo_id": merged_repo,
            "summary": str(merged_summary),
            "source_checkpoint": str(selected.checkpoint),
            "source_label": selected.label,
            "source_success_rate": selected.success_rate,
        }
        manifest["status"] = "complete"
        _write_json(_manifest_path(round_root), manifest)
        return manifest

    # Stage 3: fixed-normalizer IL warm-start.
    il_inputs = _il_stage_inputs(
        source_checkpoint=il_checkpoint,
        dataset_root=merged_dataset,
        repo_id=merged_repo,
    )
    il_record = manifest.get("stages", {}).get("il")
    il_valid = False
    if _stage_is_complete(manifest, "il") and isinstance(il_record, Mapping):
        try:
            _require_stage_inputs(il_record, il_inputs, stage="il")
            il_output = Path(str(il_record["output_dir"]))
            next_il = _il_stage_valid(
                il_record,
                source_checkpoint=il_checkpoint,
                dataset_root=merged_dataset,
                repo_id=merged_repo,
            )
            il_valid = True
        except Exception as exc:
            _invalidate_stage(manifest, round_root, "il", exc)
    if (
        not il_valid
        and isinstance(il_record, Mapping)
        and il_record.get("status") in {"failed", "planned", "invalid"}
    ):
        try:
            claimed_record, next_il = _claim_completed_il_output(
                il_record,
                source_checkpoint=il_checkpoint,
                dataset_root=merged_dataset,
                repo_id=merged_repo,
                expected_steps=args.il_steps,
            )
            _record(manifest, round_root, "il", claimed_record)
            il_record = claimed_record
            il_valid = True
        except Exception:
            # A partial or mismatched output is not claimable; create a fresh
            # attempt through the normal failure/retry path below.
            pass
    if not il_valid:
        attempt_no, attempt = _next_attempt(round_root / "il")
        il_output = attempt / "train"
        stage_manifest = attempt / "il_stage.json"
        command = _il_command(
            args,
            il_checkpoint,
            merged_dataset,
            merged_repo,
            il_output,
            seed=args.il_seed + round_index * 100_000,
            run_name=f"rl100-round-{round_index:03d}-il",
        )
        planned = {
            "status": "planned",
            "attempt": attempt_no,
            "command": command,
            "inputs": il_inputs,
            "output_dir": str(il_output),
            "stage_manifest": str(stage_manifest),
        }
        if args.dry_run:
            _record_planned(manifest, round_root, "il", planned)
            return manifest
        try:
            _run_logged(command, log_path=attempt / "orchestrator.log")
            stage_payload = _read_json(stage_manifest)
            next_il = _checkpoint_valid(Path(str(stage_payload["checkpoint"])))
            completed_record = {**planned, "status": "complete", "checkpoint": str(next_il)}
            _il_stage_valid(
                completed_record,
                source_checkpoint=il_checkpoint,
                dataset_root=merged_dataset,
                repo_id=merged_repo,
            )
        except BaseException as exc:
            _record(manifest, round_root, "il", {**planned, "status": "failed", "error": str(exc)})
            raise
        il_record = completed_record
        _record(manifest, round_root, "il", il_record)

    # Stage 3b: evaluate every IL checkpoint before choosing the offline-RL
    # initialization.  This is deliberately separate from the IL trainer so
    # a failed Newton evaluation can be resumed without rerunning 50k IL
    # optimizer steps.
    il_output = Path(str(il_record["output_dir"])).resolve()
    il_specs = _il_checkpoint_specs(
        il_output,
        steps=args.il_steps,
        save_freq=args.il_save_freq,
        eval_every_steps=args.il_eval_every_steps,
    )
    il_eval_inputs = _il_eval_inputs(
        train_output=il_output,
        source_checkpoint=il_checkpoint,
        dataset_root=merged_dataset,
        repo_id=merged_repo,
        steps=args.il_steps,
        il_save_freq=args.il_save_freq,
        eval_every_steps=args.il_eval_every_steps,
        episodes=args.il_eval_episodes,
        batch_size=args.il_eval_batch_size,
        inference_steps=args.il_eval_inference_steps or args.inference_steps,
        policy_device=args.device,
        env_device=args.sim_device,
        env_type=args.il_eval_env_type,
        seed=args.il_seed + round_index * 100_000,
        specs=il_specs,
    )
    il_eval_record = manifest.get("stages", {}).get("il_eval")
    best_il: ILEvaluation | None = None
    if _stage_is_complete(manifest, "il_eval") and isinstance(il_eval_record, Mapping):
        try:
            best_il = _il_eval_valid(
                il_eval_record,
                expected_inputs=il_eval_inputs,
                specs=il_specs,
                expected_episodes=args.il_eval_episodes,
            )
        except Exception as exc:
            _invalidate_stage(manifest, round_root, "il_eval", exc)
    if best_il is None:
        attempt_no, attempt = _next_attempt(round_root / "il_eval")
        eval_output = il_output
        planned = {
            "status": "planned",
            "attempt": attempt_no,
            "inputs": il_eval_inputs,
            "output_dir": str(eval_output),
            "log_dir": str(attempt / "logs"),
        }
        if args.dry_run:
            _record_planned(manifest, round_root, "il_eval", planned)
            return manifest
        try:
            best_il = _run_il_evaluation_stage(
                eval_output=eval_output,
                specs=il_specs,
                expected_episodes=args.il_eval_episodes,
                batch_size=args.il_eval_batch_size,
                inference_steps=args.il_eval_inference_steps or args.inference_steps,
                policy_device=args.device,
                env_device=args.sim_device,
                env_type=args.il_eval_env_type,
                seed=args.il_seed + round_index * 100_000,
                log_root=attempt / "logs",
            )
        except BaseException as exc:
            _record(manifest, round_root, "il_eval", {**planned, "status": "failed", "error": str(exc)})
            raise
        il_eval_record = {
            **planned,
            "status": "complete",
            "best": _il_eval_payload(best_il),
        }
        _record(manifest, round_root, "il_eval", il_eval_record)
    assert best_il is not None
    offline_il_checkpoint = _select_offline_il_checkpoint(
        best_il,
        next_il,
        use_final=args.offline_use_il_final,
    )

    # Stage 4: offline RL.
    offline_inputs = _offline_stage_inputs(
        checkpoint=offline_il_checkpoint,
        dataset_root=merged_dataset,
        repo_id=merged_repo,
        summary=merged_summary,
    )
    offline_record = manifest.get("stages", {}).get("offline")
    offline_valid = False
    if _stage_is_complete(manifest, "offline") and isinstance(offline_record, Mapping):
        try:
            _require_stage_inputs(offline_record, offline_inputs, stage="offline")
            offline_output = Path(str(offline_record["output_dir"]))
            next_best = _offline_valid(
                offline_output,
                sync_target=args.offline_sync_target,
                eval_every_syncs=args.offline_eval_every_syncs,
                eval_episodes=args.offline_eval_episodes,
                expected_inputs=offline_inputs,
            )
            offline_valid = True
        except Exception as exc:
            _invalidate_stage(manifest, round_root, "offline", exc)
    if not offline_valid:
        attempt_no, attempt = _next_attempt(round_root / "offline")
        offline_output = attempt / "run"
        command = _offline_command(
            args,
            offline_il_checkpoint,
            merged_dataset,
            merged_repo,
            merged_summary,
            offline_output,
            seed=args.offline_seed + round_index * 100_000,
            run_name=f"rl100-round-{round_index:03d}-offline",
        )
        planned = {
            "status": "planned",
            "attempt": attempt_no,
            "command": command,
            "inputs": offline_inputs,
            "output_dir": str(offline_output),
        }
        if args.dry_run:
            _record_planned(manifest, round_root, "offline", planned)
            return manifest
        try:
            _run_logged(command, log_path=attempt / "offline.log")
            next_best = _offline_valid(
                offline_output,
                sync_target=args.offline_sync_target,
                eval_every_syncs=args.offline_eval_every_syncs,
                eval_episodes=args.offline_eval_episodes,
                expected_inputs=offline_inputs,
            )
        except BaseException as exc:
            _record(manifest, round_root, "offline", {**planned, "status": "failed", "error": str(exc)})
            raise
        offline_record = {**planned, "status": "complete"}
        _record(manifest, round_root, "offline", offline_record)

    manifest["outputs"] = {
        "dataset_root": str(merged_dataset),
        "repo_id": merged_repo,
        "summary": str(merged_summary),
        "il_checkpoint": str(offline_il_checkpoint),
        "il_final_checkpoint": str(next_il),
        "il_best_label": best_il.label,
        "il_best_success_rate": best_il.success_rate,
        "offline_init_selection": "il_final" if args.offline_use_il_final else "il_best_eval",
        "offline_run": str(offline_output.resolve()),
        "best_checkpoint": str(next_best.checkpoint),
        "best_label": next_best.label,
        "best_success_rate": next_best.success_rate,
    }
    manifest["status"] = "complete"
    _write_json(_manifest_path(round_root), manifest)
    return manifest


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if args.smoke:
        args.episodes = min(args.episodes, 2)
        args.num_envs = min(args.num_envs, 2)
        args.rounds = min(args.rounds, 1)
        args.il_steps = min(args.il_steps, 2)
        args.il_save_freq = min(args.il_save_freq, args.il_steps)
        args.il_eval_every_steps = min(args.il_eval_every_steps, args.il_steps)
        args.il_eval_episodes = min(args.il_eval_episodes, 1)
        args.offline_iql_steps = min(args.offline_iql_steps, 1)
        args.offline_dynamics_steps = min(args.offline_dynamics_steps, 1)
        args.offline_sync_target = min(args.offline_sync_target, 1)
        args.offline_eval_every_syncs = 1
        args.offline_eval_episodes = min(args.offline_eval_episodes, 1)
    current_dataset = Path(args.base_dataset_root).resolve()
    current_repo = args.base_repo_id
    current_summary = Path(args.base_summary).resolve()
    current_il = _checkpoint_valid(Path(args.il_checkpoint))
    current_offline = Path(args.source_offline_run).resolve(strict=True)
    last_manifest: dict[str, Any] | None = None
    for round_index in range(1, args.rounds + 1):
        source_eval_episodes = args.source_eval_episodes if round_index == 1 else args.offline_eval_episodes
        last_manifest = _run_one_round(
            args,
            round_index=round_index,
            base_dataset_root=current_dataset,
            base_repo_id=current_repo,
            base_summary=current_summary,
            il_checkpoint=current_il,
            source_offline_run=current_offline,
            source_eval_episodes=source_eval_episodes,
        )
        outputs = last_manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            return Path(args.output_root).resolve()
        current_dataset = Path(str(outputs["dataset_root"])).resolve()
        current_repo = str(outputs["repo_id"])
        current_summary = Path(str(outputs["summary"])).resolve()
        current_il = _checkpoint_valid(Path(str(outputs["il_checkpoint"])))
        current_offline = Path(str(outputs["offline_run"])).resolve()
    if args.final_rollout_merge and last_manifest is not None:
        final_manifest = _run_one_round(
            args,
            round_index=args.rounds + 1,
            base_dataset_root=current_dataset,
            base_repo_id=current_repo,
            base_summary=current_summary,
            il_checkpoint=current_il,
            source_offline_run=current_offline,
            source_eval_episodes=args.offline_eval_episodes,
            stop_after_merge=True,
        )
        final_outputs = final_manifest.get("outputs")
        if isinstance(final_outputs, Mapping):
            return Path(str(final_outputs["dataset_root"])).resolve()
        return Path(args.output_root).resolve()
    return (
        Path(str(last_manifest["outputs"]["offline_run"])).resolve()
        if last_manifest and isinstance(last_manifest.get("outputs"), Mapping)
        else Path(args.output_root).resolve()
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
