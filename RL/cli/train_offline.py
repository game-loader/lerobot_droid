# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Command line entry point for a bounded offline diffusion-RL run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.lerobot_v3 import LeRobotV3DecisionDataset, collate_decision_batches
from RL.algorithms.iql import IQL
from RL.checkpointing import RLProvenance
from RL.config import RLConfig, TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.tracking import SwanLabConfig, create_swanlab_tracker
from RL.trainers.offline import OfflineTrainer
from RL.types import DecisionBatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iql-steps", type=int, default=1000)
    parser.add_argument("--actor-steps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "offline", "local", "disabled"),
        default="disabled",
    )
    parser.add_argument("--swanlab-strict", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _batches(dataset: LeRobotV3DecisionDataset, *, batch_size: int) -> Iterator[DecisionBatch]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_decision_batches,
        drop_last=False,
    )
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("offline dataset produced no decision batches")


def _build_provenance(
    *,
    adapter: CheckpointAdapter,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    summary: Path,
    config: RLConfig,
) -> RLProvenance:
    model_hash = _sha256(checkpoint / "model.safetensors")
    policy_config_hash = _sha256(checkpoint / "config.json")
    policy_config = adapter.policy.config
    return RLProvenance(
        stage="offline",
        root_base_path=str(checkpoint.resolve()),
        root_base_hash=model_hash,
        input_checkpoint=str(checkpoint.resolve()),
        input_checkpoint_hash=model_hash,
        processor_fingerprint=adapter.processor_fingerprint(),
        policy_type=type(policy_config).__name__,
        policy_config_hash=policy_config_hash,
        dataset_root=str(dataset_root.resolve()),
        dataset_repo_id=repo_id,
        dataset_summary_path=str(summary.resolve()),
        dataset_summary_hash=_sha256(summary),
        feature_keys=tuple(sorted(policy_config.input_features)),
        state_key=config.state_key,
        state_dim=config.state_dim,
        chunk_size=config.chunk_size,
        action_dim=config.action_dim,
        active_action_mask=tuple(bool(value) for value in adapter.active_action_mask.tolist()),
        lerobot_commit=_git_commit(),
        rl100_url="https://github.com/Lei-Kun/RL-100",
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def _validate_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"output_dir must be a directory: {output_dir}")
    if next(output_dir.iterdir(), None) is not None:
        raise FileExistsError(f"output_dir must be empty: {output_dir}")


def _tracking_run_config(
    args: argparse.Namespace,
    *,
    dataset_summary: dict[str, object],
) -> dict[str, object]:
    resolved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "arguments": resolved_args,
        "dataset": dict(dataset_summary),
        "hashes": {
            "checkpoint_model_sha256": _sha256(args.checkpoint / "model.safetensors"),
            "checkpoint_config_sha256": _sha256(args.checkpoint / "config.json"),
            "dataset_summary_sha256": _sha256(args.summary),
        },
    }


def run(args: argparse.Namespace) -> Path:
    _validate_output_dir(args.output_dir)
    if args.smoke:
        args.iql_steps = 1
        args.actor_steps = 1
        args.batch_size = 2
        args.inference_steps = 2
    for name in ("iql_steps", "actor_steps", "inference_steps", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    torch.manual_seed(args.seed)

    current_checkpoint = CheckpointAdapter.load(args.checkpoint, device=args.device)
    old_checkpoint = CheckpointAdapter.load(args.checkpoint, device=args.device)
    trace_config = TraceConfig(num_inference_steps=args.inference_steps)
    current = DiffusionRLAdapter(current_checkpoint, trace_config)
    old = DiffusionRLAdapter(old_checkpoint, trace_config)
    policy_config = current.policy.config
    state_dim = policy_config.robot_state_feature.shape[0]
    action_dim = policy_config.action_feature.shape[0]
    rl_config = RLConfig(
        state_key="observation.state",
        n_obs_steps=policy_config.n_obs_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=policy_config.n_action_steps,
    )
    dataset = LeRobotV3DecisionDataset.from_root(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        summary_path=args.summary,
        config=rl_config,
    )
    dataset_summary = dataset.inspection_summary()
    print(json.dumps(dataset_summary, sort_keys=True))
    batches = _batches(dataset, batch_size=args.batch_size)
    encoder = StateFeatureEncoder(
        state_dim=state_dim,
        n_obs_steps=policy_config.n_obs_steps,
        hidden_dims=(256, 256),
        output_dim=256,
    )
    iql = IQL(
        feature_encoder=encoder,
        action_dim=action_dim,
        chunk_size=policy_config.n_action_steps,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(256, 256),
        expectile=0.7,
        tau=0.005,
        q_lr=3e-4,
        v_lr=3e-4,
    ).to(next(current.policy.parameters()).device)
    actor_optimizer = torch.optim.Adam(current.policy.parameters(), lr=1e-5)
    tracker = create_swanlab_tracker(
        SwanLabConfig(
            project=args.swanlab_project or "",
            run_name=args.swanlab_run_name,
            mode=args.swanlab_mode,
            log_dir=args.output_dir / "swanlog",
            strict=args.swanlab_strict,
        ),
        _tracking_run_config(args, dataset_summary=dataset_summary),
    )
    try:
        trainer = OfflineTrainer(
            current_policy=current,
            old_policy=old,
            iql=iql,
            actor_optimizer=actor_optimizer,
            metrics_path=args.output_dir / "metrics.jsonl",
            actor_clip_ratio=0.2,
            tracker=tracker,
        )
        for _ in range(args.iql_steps):
            trainer.record_metrics(trainer.train_iql_step(next(batches)), 0)
        for index in range(args.actor_steps):
            metrics = trainer.train_actor_step(
                next(batches), generator=current.make_generator(args.seed + index + 1)
            )
            trainer.record_metrics(metrics, 1)
        provenance = _build_provenance(
            adapter=current_checkpoint,
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            summary=args.summary,
            config=rl_config,
        )
        destination = args.output_dir / "checkpoints" / "final"
        saved = trainer.save_checkpoint(destination, provenance=provenance, rl_config=rl_config)
        print(f"saved={saved}")
        return saved
    finally:
        if tracker is not None:
            tracker.finish()


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
