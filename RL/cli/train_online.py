# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

"""Run bounded online diffusion PPO in the fused Moya Newton environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.moya_newton import create_moya_env
from RL.checkpointing import (
    LoadedRLCheckpoint,
    RLProvenance,
    load_rl_checkpoint,
    restore_rl_state,
)
from RL.config import RLConfig, TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.trainers.online import OnlineTrainer


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-decisions", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--episode-length", type=int, default=930)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _checkpoint_root(checkpoint: Path) -> Path:
    root = checkpoint.resolve(strict=True)
    final = root / "checkpoints" / "final"
    if (final / "manifest.json").is_file() and (final / "rl_state.pt").is_file():
        return final
    return root


def _resolve_checkpoint(
    checkpoint: Path, *, device: str
) -> tuple[CheckpointAdapter, LoadedRLCheckpoint | None, Path]:
    root = _checkpoint_root(checkpoint)
    if (root / "manifest.json").is_file() and (root / "rl_state.pt").is_file():
        loaded = load_rl_checkpoint(root, device=device)
        return loaded.current, loaded, root
    if (root / "pretrained_model").is_dir() and not (root / "model.safetensors").is_file():
        root = root / "pretrained_model"
    return CheckpointAdapter.load(root, device=device), None, root


def _build_provenance(
    *,
    adapter: CheckpointAdapter,
    checkpoint: Path,
    loaded: LoadedRLCheckpoint | None,
    config: RLConfig,
) -> RLProvenance:
    policy_root = adapter.source_path
    model_hash = _sha256(policy_root / "model.safetensors")
    input_hash_path = checkpoint / "manifest.json" if (checkpoint / "manifest.json").is_file() else policy_root / "model.safetensors"
    previous = loaded.provenance if loaded is not None else None
    policy_config = adapter.policy.config
    return RLProvenance(
        stage="online",
        root_base_path=previous.root_base_path if previous else str(policy_root),
        root_base_hash=previous.root_base_hash if previous else model_hash,
        input_checkpoint=str(checkpoint.resolve()),
        input_checkpoint_hash=_sha256(input_hash_path),
        processor_fingerprint=adapter.processor_fingerprint(),
        policy_type=type(policy_config).__name__,
        policy_config_hash=_sha256(policy_root / "config.json"),
        dataset_root=previous.dataset_root if previous else "",
        dataset_repo_id=previous.dataset_repo_id if previous else "",
        dataset_summary_path=previous.dataset_summary_path if previous else "",
        dataset_summary_hash=previous.dataset_summary_hash if previous else "",
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


def _swanlab_logger(args: argparse.Namespace) -> tuple[Callable[[Mapping[str, float]], None] | None, Any]:
    if not args.swanlab_project:
        return None, None
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab logging was requested but swanlab is not installed in this uv environment"
        ) from exc
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name,
        config=config,
        logdir=str(args.output_dir / "swanlog"),
    )
    step = 0

    def log(metrics: Mapping[str, float]) -> None:
        nonlocal step
        swanlab.log(dict(metrics), step=step)
        step += 1

    return log, run


def _restore_online_state(loaded: LoadedRLCheckpoint, trainer: OnlineTrainer) -> None:
    if loaded.state.stage != "online":
        return
    value_state = loaded.state.trainer_state.get("value_network")
    if not isinstance(value_state, Mapping):
        raise ValueError("online checkpoint is missing trainer.value_network state")
    expected_value = trainer.value_network.state_dict()
    if set(value_state) != set(expected_value):
        raise ValueError("online checkpoint value network keys disagree")
    for key, expected in expected_value.items():
        actual = value_state[key]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            raise ValueError(f"online checkpoint value network tensor mismatch for {key!r}")
        if (actual.is_floating_point() or actual.is_complex()) and not torch.isfinite(actual).all().item():
            raise ValueError(f"online checkpoint value network tensor {key!r} is non-finite")
    sampler_state = loaded.state.sampler_state
    if not isinstance(sampler_state, Mapping) or not isinstance(
        sampler_state.get("generator"), torch.Tensor
    ) or sampler_state["generator"].dtype != torch.uint8:
        raise ValueError("online checkpoint is missing a valid sampler generator state")
    sampler_probe = torch.Generator(device=trainer._generator.device)
    try:
        sampler_probe.set_state(
            sampler_state["generator"].to(device=trainer._generator.device)
        )
    except RuntimeError as exc:
        raise ValueError("online checkpoint sampler generator state is malformed") from exc

    counters = restore_rl_state(
        loaded,
        old_policy=trainer.old_policy.policy,
        actor_optimizer=trainer.actor_optimizer,
        optimizers={"value": trainer.value_optimizer},
    )
    trainer.value_network.load_state_dict(dict(value_state), strict=True)
    trainer.restore_sampler_state(sampler_state)
    trainer.counters = counters


def run(args: argparse.Namespace) -> Path:
    if args.smoke:
        args.num_envs = 16
        args.rollout_decisions = 1
        args.ppo_epochs = 1
        args.updates = 1
        args.inference_steps = 2
        args.minibatch_size = 16
    for name in (
        "num_envs",
        "rollout_decisions",
        "ppo_epochs",
        "updates",
        "inference_steps",
        "episode_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("actor_lr", "value_lr"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    torch.manual_seed(args.seed)

    checkpoint = args.checkpoint.resolve(strict=True)
    current_checkpoint, loaded, resolved_checkpoint = _resolve_checkpoint(
        checkpoint, device=args.device
    )
    old_checkpoint = CheckpointAdapter.load(current_checkpoint.source_path, device=args.device)
    trace_config = TraceConfig(num_inference_steps=args.inference_steps)
    current = DiffusionRLAdapter(current_checkpoint, trace_config)
    old = DiffusionRLAdapter(old_checkpoint, trace_config)
    policy_config = current.policy.config
    state_feature = policy_config.robot_state_feature
    action_feature = policy_config.action_feature
    if state_feature is None or tuple(state_feature.shape) != (39,):
        raise ValueError(f"Moya online training requires a 39D state feature, got {state_feature!r}")
    if action_feature is None or tuple(action_feature.shape) != (14,):
        raise ValueError(f"Moya online training requires a 14D action feature, got {action_feature!r}")
    rl_config = RLConfig(
        trace=trace_config,
        state_key="observation.state",
        n_obs_steps=policy_config.n_obs_steps,
        state_dim=39,
        action_dim=14,
        chunk_size=policy_config.n_action_steps,
        gamma=args.gamma,
    )
    if loaded is not None and loaded.state.stage == "online" and loaded.config != rl_config:
        raise ValueError(
            "online checkpoint RLConfig is immutable; pass matching gamma, inference steps, "
            "state history, and action chunk settings before resuming"
        )
    logger, swanlab_run = _swanlab_logger(args)
    env = None
    try:
        metrics_path = args.output_dir / "metrics.jsonl"
        if loaded is not None and loaded.state.stage == "online":
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if metrics_path.exists():
                raise FileExistsError(f"resume metrics destination already exists: {metrics_path}")
            shutil.copyfile(loaded.metrics_path, metrics_path)
        elif metrics_path.exists():
            raise FileExistsError(f"metrics destination already exists: {metrics_path}")
        trainer = OnlineTrainer(
            current_policy=current,
            old_policy=old,
            actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=args.actor_lr),
            metrics_path=metrics_path,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            seed=args.seed,
            logger=logger,
        )
        for group in trainer.value_optimizer.param_groups:
            group["lr"] = args.value_lr
        if loaded is not None:
            _restore_online_state(loaded, trainer)

        env = create_moya_env(
            num_envs=args.num_envs,
            device=args.sim_device,
            episode_length=args.episode_length,
            headless=True,
        )
        for update in range(args.updates):
            rollout = trainer.collect(
                env,
                decisions=args.rollout_decisions,
                seed=args.seed + trainer.counters.global_updates,
            )
            metrics = trainer.update(rollout)
            print(json.dumps({"update": update + 1, **metrics}, sort_keys=True))
        provenance = _build_provenance(
            adapter=current_checkpoint,
            checkpoint=resolved_checkpoint,
            loaded=loaded,
            config=rl_config,
        )
        destination = args.output_dir / "checkpoints" / "final"
        saved = trainer.save_checkpoint(destination, provenance=provenance, rl_config=rl_config)
        print(f"saved={saved}")
        return saved
    finally:
        if env is not None:
            try:
                env.close()
            finally:
                if swanlab_run is not None and hasattr(swanlab_run, "finish"):
                    swanlab_run.finish()
        elif swanlab_run is not None and hasattr(swanlab_run, "finish"):
            swanlab_run.finish()


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
