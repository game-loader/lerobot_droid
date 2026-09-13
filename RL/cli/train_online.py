# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

"""Run bounded online diffusion PPO in a user-provided real-robot environment.

The command never creates a simulator implicitly.  ``--env-factory`` must
resolve to a callable returning a Gymnasium-compatible vector environment with
``num_envs``, ``reset()`` and ``step(action)``.  A one-robot adapter should
expose ``num_envs=1`` and may return unbatched observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.dp3.configuration_dp3 import DP3Config
from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.real_robot import validate_vector_env
from RL.checkpointing import (
    LoadedRLCheckpoint,
    RLProvenance,
    load_rl_checkpoint,
    restore_rl_state,
)
from RL.config import RLConfig, TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import DP3FeatureEncoder
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
    parser.add_argument(
        "--env-device",
        "--sim-device",
        dest="env_device",
        default=None,
        help=(
            "Optional device hint forwarded to the caller-provided environment "
            "factory. This command never creates or configures a simulator."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of real-robot worlds exposed by the environment (default: 1).",
    )
    # A decision emits one policy action chunk.  The environment factory may
    # use ``episode_length`` to impose a task-specific truncation horizon.
    parser.add_argument("--rollout-decisions", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--probability-sigma-min", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument(
        "--env-factory",
        default=None,
        help=(
            "Environment factory as module:callable. The callable must return a "
            "Gymnasium-compatible vector environment with num_envs, reset(), and step()."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("terminal_success", "environment"),
        default="terminal_success",
        help=(
            "terminal_success preserves the sparse RL-100 reward contract; "
            "environment accumulates rewards returned by env.step()."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--debug", action="store_true")
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
    input_hash_path = (
        checkpoint / "manifest.json"
        if (checkpoint / "manifest.json").is_file()
        else policy_root / "model.safetensors"
    )
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
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
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


def _resolve_callable(spec: str) -> Callable[..., Any]:
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"env-factory must use module:callable syntax, got {spec!r}")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError(f"env-factory must use module:callable syntax, got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"env-factory target is not callable: {spec!r}")
    return factory


def _create_env(args: argparse.Namespace) -> Any:
    factory = _resolve_callable(args.env_factory)
    # Forward only generic hints explicitly supplied by the caller.  Do not
    # force simulation-only ``headless`` kwargs onto a real-robot factory.
    candidate_kwargs: dict[str, Any] = {"num_envs": args.num_envs}
    if args.env_device is not None:
        candidate_kwargs["device"] = args.env_device
    if args.episode_length is not None:
        candidate_kwargs["episode_length"] = args.episode_length
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None
    if signature is None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    ):
        return factory(**candidate_kwargs)
    accepted = {name: value for name, value in candidate_kwargs.items() if name in signature.parameters}
    return factory(**accepted)


def _restore_online_state(loaded: LoadedRLCheckpoint, trainer: OnlineTrainer) -> None:
    if loaded.state.stage != "online":
        return
    value_state = loaded.state.trainer_state.get("value_network")
    if not isinstance(value_state, Mapping):
        raise ValueError("online checkpoint is missing trainer.value_network state")
    saved_contract = loaded.state.trainer_state.get("online_contract")
    if not isinstance(saved_contract, Mapping) or dict(saved_contract) != trainer.training_contract():
        raise ValueError("online checkpoint training contract does not match the requested resume")
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
    if (
        not isinstance(sampler_state, Mapping)
        or not isinstance(sampler_state.get("generator"), torch.Tensor)
        or sampler_state["generator"].dtype != torch.uint8
    ):
        raise ValueError("online checkpoint is missing a valid sampler generator state")
    sampler_probe = torch.Generator(device=trainer._generator.device)
    try:
        sampler_probe.set_state(sampler_state["generator"].detach().cpu().contiguous())
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
    historical_metrics: list[dict[str, float]] = []
    for line in loaded.metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("online checkpoint metric rows must be JSON objects")
        converted = {str(key): float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in converted.values()):
            raise ValueError("online checkpoint metric rows must be finite")
        historical_metrics.append(converted)
    if len(historical_metrics) != counters.metrics_rows:
        raise ValueError("online checkpoint metric history disagrees with its counters")
    trainer._metrics_snapshot = historical_metrics
    trainer.counters = counters


def run(args: argparse.Namespace) -> Path:
    if args.smoke:
        args.rollout_decisions = 1
        args.ppo_epochs = 1
        args.updates = 1
        args.inference_steps = 2
        args.minibatch_size = args.num_envs
    for name in (
        "num_envs",
        "rollout_decisions",
        "ppo_epochs",
        "updates",
        "inference_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.episode_length is not None and args.episode_length <= 0:
        raise ValueError("episode_length must be positive when supplied")
    if not args.env_factory:
        raise ValueError(
            "--env-factory is required for online RL. This command never creates a "
            "simulator implicitly; provide a real-robot vector environment factory."
        )
    for name in ("actor_lr", "value_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(args.probability_sigma_min) or args.probability_sigma_min <= 0:
        raise ValueError("probability_sigma_min must be positive and finite")
    torch.manual_seed(args.seed)

    checkpoint = args.checkpoint.resolve(strict=True)
    current_checkpoint, loaded, resolved_checkpoint = _resolve_checkpoint(checkpoint, device=args.device)
    old_checkpoint = CheckpointAdapter.load(current_checkpoint.source_path, device=args.device)
    trace_config = TraceConfig(
        num_inference_steps=args.inference_steps,
        probability_sigma_min=args.probability_sigma_min,
    )
    current = DiffusionRLAdapter(current_checkpoint, trace_config)
    old = DiffusionRLAdapter(old_checkpoint, trace_config)
    policy_config = current.policy.config
    state_feature = policy_config.robot_state_feature
    action_feature = policy_config.action_feature
    if state_feature is None or len(state_feature.shape) != 1:
        raise ValueError(f"online training requires a vector state feature, got {state_feature!r}")
    if action_feature is None or len(action_feature.shape) != 1:
        raise ValueError(f"online training requires a vector action feature, got {action_feature!r}")
    state_dim = state_feature.shape[0]
    action_dim = action_feature.shape[0]
    rl_config = RLConfig(
        trace=trace_config,
        state_key="observation.state",
        n_obs_steps=policy_config.n_obs_steps,
        state_dim=state_dim,
        action_dim=action_dim,
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
            # RL-100 3D uses obs2latent for the critic as well as the actor.
            # Keep the historical state-only critic for standard Diffusion
            # checkpoints, but make DP3's value function multimodal by
            # default so online PPO can use the point cloud and both wrists.
            value_encoder=(
                DP3FeatureEncoder(current.policy) if isinstance(policy_config, DP3Config) else None
            ),
            metrics_path=metrics_path,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            seed=args.seed,
            logger=logger,
            debug=args.debug,
            reward_mode=args.reward_mode,
        )
        for group in trainer.value_optimizer.param_groups:
            group["lr"] = args.value_lr
        trainer.set_runtime_contract(
            {
                "num_envs": args.num_envs,
                "rollout_decisions": args.rollout_decisions,
                "episode_length": args.episode_length,
                "env_device": args.env_device,
                "policy_device": args.device,
                "actor_lr": args.actor_lr,
                "value_lr": args.value_lr,
                "env_factory": args.env_factory,
                "reward_mode": args.reward_mode,
            }
        )
        if loaded is not None:
            _restore_online_state(loaded, trainer)

        env = _create_env(args)
        validate_vector_env(env, expected_num_envs=args.num_envs)
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
