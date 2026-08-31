# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Command line entry point for a bounded offline diffusion-RL run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.policies.dp3.configuration_dp3 import DP3Config
from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.lerobot_v3 import LeRobotV3DecisionDataset, collate_decision_batches
from RL.algorithms.amq import AMQEvaluator
from RL.algorithms.dynamics import (
    DP3FeatureDynamicsEnsemble,
    PolicyPromotionGate,
    StateDynamicsEnsemble,
)
from RL.algorithms.iql import IQL
from RL.checkpointing import RLProvenance
from RL.config import AMQConfig, RLConfig, TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import DP3FeatureEncoder, StateFeatureEncoder
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


def _save_eval_policy_snapshot(checkpoint: CheckpointAdapter, destination: str | Path) -> Path:
    """Persist a loadable LeRobot policy bundle for one evaluation point."""

    if not isinstance(checkpoint, CheckpointAdapter):
        raise ValueError("checkpoint must be a CheckpointAdapter")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"evaluation policy snapshot already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    checkpoint.policy.save_pretrained(target)
    # Save the live processor objects rather than copying files from the input
    # checkpoint.  This keeps the snapshot self-contained and preserves any
    # state that may have been updated while training.
    checkpoint.preprocessor.save_pretrained(target, config_filename="policy_preprocessor.json")
    checkpoint.postprocessor.save_pretrained(target, config_filename="policy_postprocessor.json")
    return target


def _parse_newton_eval_info(
    path: str | Path,
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, float]:
    """Extract finite aggregate metrics from a LeRobot ``eval_info.json``."""

    info_path = Path(path)
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read evaluation info {info_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation info must contain a JSON object: {info_path}")
    aggregate = None
    for key in ("overall", "aggregated"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            aggregate = candidate
            break
    if aggregate is None:
        raise ValueError(f"evaluation info has no overall/aggregated metrics: {info_path}")

    def _number(name: str, *, required: bool = False) -> float | None:
        value = aggregate.get(name)
        if value is None:
            if required:
                raise ValueError(f"evaluation info is missing required metric {name!r}")
            return None
        if isinstance(value, bool):
            raise ValueError(f"evaluation metric {name!r} must be numeric")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"evaluation metric {name!r} must be numeric") from exc
        if not math.isfinite(converted):
            raise ValueError(f"evaluation metric {name!r} is non-finite: {value!r}")
        return converted

    pc_success_value = _number("pc_success", required=True)
    n_episodes_value = _number("n_episodes", required=True)
    # The required=True calls above raise when absent; explicit narrowing keeps
    # this helper friendly to static type checkers as well.
    if pc_success_value is None or n_episodes_value is None:
        raise ValueError("evaluation info is missing required aggregate metrics")
    pc_success = pc_success_value
    n_episodes = n_episodes_value
    if n_episodes <= 0:
        raise ValueError(f"evaluation metric 'n_episodes' must be positive, got {n_episodes}")
    metrics: dict[str, float] = {
        "eval/pc_success": pc_success,
        # LeRobot reports percentage success, so recover the count for a
        # directly interpretable scalar without requiring per-episode output.
        "eval/successes": n_episodes * pc_success / 100.0,
        "eval/n_episodes": n_episodes,
    }
    for source_name, metric_name in (
        ("avg_sum_reward", "eval/avg_sum_reward"),
        ("avg_max_reward", "eval/avg_max_reward"),
    ):
        value = _number(source_name)
        if value is not None:
            metrics[metric_name] = value

    eval_seconds = elapsed_seconds
    if eval_seconds is None:
        eval_seconds = _number("eval_s")
    if eval_seconds is not None:
        if not math.isfinite(eval_seconds) or eval_seconds < 0:
            raise ValueError("evaluation elapsed seconds must be finite and nonnegative")
        metrics["eval/eval_seconds"] = eval_seconds
        # Keep the LeRobot evaluator's historical key as an alias for
        # dashboards built against the standard eval schema.
        metrics["eval/eval_s"] = eval_seconds
    return metrics


def _run_newton_eval(
    *,
    policy_path: str | Path,
    output_dir: str | Path,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    seed: int,
    env_type: str = "moya_newton",
    executable: str | None = None,
) -> dict[str, float]:
    """Run standard LeRobot evaluation and return prefixed scalar metrics."""

    if episodes <= 0 or batch_size <= 0 or inference_steps <= 0:
        raise ValueError("evaluation episodes, batch_size, and inference_steps must be positive")
    if not env_type:
        raise ValueError("evaluation env_type must be nonempty")
    if executable is not None and not executable:
        raise ValueError("evaluation executable must be nonempty when provided")
    if not policy_device or not env_device:
        raise ValueError("evaluation policy_device and env_device must be nonempty")
    policy_path = Path(policy_path).resolve()
    result_dir = Path(output_dir).resolve()
    if not policy_path.is_dir():
        raise ValueError(f"evaluation policy snapshot must be a directory: {policy_path}")
    if result_dir.exists():
        raise FileExistsError(f"evaluation output directory already exists: {result_dir}")
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    command = (
        [executable] if executable is not None else [sys.executable, "-m", "lerobot.scripts.lerobot_eval"]
    ) + [
        f"--policy.path={policy_path}",
        f"--policy.device={policy_device}",
        f"--policy.num_inference_steps={inference_steps}",
        f"--env.type={env_type}",
        f"--env.device={env_device}",
        f"--eval.n_episodes={episodes}",
        f"--eval.batch_size={batch_size}",
        "--eval.use_async_envs=false",
        f"--seed={seed}",
        f"--output_dir={result_dir}",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        executable_name = executable or f"{sys.executable} -m lerobot.scripts.lerobot_eval"
        raise RuntimeError(
            f"could not execute {executable_name!r}; install the LeRobot evaluation entrypoint"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(text for text in (getattr(exc, "stdout", ""), getattr(exc, "stderr", "")) if text)
        tail = output[-4000:] if output else "<no subprocess output>"
        raise RuntimeError(
            f"LeRobot Newton evaluation failed with exit code {exc.returncode}; output tail:\n{tail}"
        ) from exc
    # Keep a short success trace visible when running interactively. Test
    # doubles may return an object without a ``stderr`` attribute.
    stderr = getattr(completed, "stderr", None)
    if stderr:
        print(str(stderr).rstrip())
    return _parse_newton_eval_info(result_dir / "eval_info.json", elapsed_seconds=time.monotonic() - started)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iql-steps", type=int, default=1000)
    parser.add_argument(
        "--actor-steps",
        type=int,
        default=1000,
        help="Maximum actor optimizer updates (0 means no cap when a sync target is set).",
    )
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=16,
        help="DataLoader worker processes used for LeRobot video decoding.",
    )
    parser.add_argument(
        "--prefetch-factor",
        "--prefetch_factor",
        type=int,
        default=4,
        help="Batches prefetched by each DataLoader worker.",
    )
    parser.add_argument(
        "--camera-cache",
        choices=("none", "ram"),
        default="none",
        help="Predecode required decision-level camera frames into RAM before offline RL.",
    )
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--probability-sigma-min", type=float, default=0.1)
    parser.add_argument(
        "--old-policy-sync-interval",
        type=int,
        default=1,
        help="Synchronize the behavior snapshot every N actor updates; 0 keeps it fixed.",
    )
    parser.add_argument(
        "--old-policy-sync-target",
        type=int,
        default=0,
        help="Stop after this many behavior-policy synchronizations; 0 uses --actor-steps.",
    )
    parser.add_argument(
        "--amq-enabled",
        action="store_true",
        help="Train state dynamics and gate behavior promotion with paired AM-Q rollouts.",
    )
    parser.add_argument("--dynamics-steps", type=int, default=0)
    parser.add_argument("--amq-rollout-horizon", type=int, default=20)
    parser.add_argument("--amq-eval-interval", type=int, default=50)
    parser.add_argument("--amq-min-dynamics-updates", type=int, default=1)
    parser.add_argument("--amq-relative-margin", type=float, default=0.05)
    parser.add_argument("--amq-max-validation-loss", type=float, default=1.0)
    parser.add_argument("--amq-max-disagreement", type=float, default=None)
    parser.add_argument("--amq-ensemble-size", type=int, default=5)
    parser.add_argument("--amq-use-critic-reference", action="store_true")
    parser.add_argument("--amq-discounted", action="store_true")
    parser.add_argument(
        "--eval-every-actor-updates",
        "--eval-every",
        type=int,
        default=0,
        help="Run a headless Newton evaluation every N actor optimizer updates; 0 disables it.",
    )
    parser.add_argument(
        "--eval-every-old-policy-syncs",
        "--eval-every-syncs",
        type=int,
        default=0,
        help="Run a headless Newton evaluation every N old-policy syncs; 0 disables it.",
    )
    parser.add_argument(
        "--checkpoint-every-old-policy-syncs",
        "--checkpoint-every-syncs",
        type=int,
        default=0,
        help="Persist a standard LeRobot policy bundle every N old-policy syncs; "
        "independent of evaluation (0 disables periodic snapshots).",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-inference-steps",
        type=int,
        default=None,
        help="Inference steps for periodic eval (defaults to --inference-steps).",
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="Policy device for periodic Newton evaluation (defaults to --device).",
    )
    parser.add_argument("--eval-env-device", default="cuda:0")
    parser.add_argument("--eval-env-type", default="moya_newton")
    parser.add_argument(
        "--eval-executable",
        default=None,
        help="Optional evaluation executable; by default invoke lerobot_eval via this Python.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "offline", "local", "disabled"),
        default="disabled",
    )
    parser.add_argument("--swanlab-strict", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include verbose per-denoising-step actor diagnostics in info/ metrics.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def _batches(
    dataset: LeRobotV3DecisionDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> Iterator[DecisionBatch]:
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate_decision_batches,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )
    loader = DataLoader(dataset, **loader_kwargs)
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
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
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
    for name in ("iql_steps", "inference_steps", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if args.actor_steps < 0:
        raise ValueError("actor_steps must be nonnegative")
    if args.old_policy_sync_target < 0:
        raise ValueError("old_policy_sync_target must be nonnegative")
    if not math.isfinite(args.actor_lr) or args.actor_lr <= 0:
        raise ValueError("actor_lr must be positive and finite")
    if not math.isfinite(args.probability_sigma_min) or args.probability_sigma_min <= 0:
        raise ValueError("probability_sigma_min must be positive and finite")
    if args.old_policy_sync_interval < 0:
        raise ValueError("old_policy_sync_interval must be nonnegative")
    if args.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if args.dynamics_steps < 0:
        raise ValueError("dynamics_steps must be nonnegative")
    for name in (
        "amq_rollout_horizon",
        "amq_eval_interval",
        "amq_min_dynamics_updates",
        "amq_ensemble_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("amq_relative_margin", "amq_max_validation_loss"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.amq_max_disagreement is not None and (
        not math.isfinite(args.amq_max_disagreement) or args.amq_max_disagreement < 0
    ):
        raise ValueError("amq_max_disagreement must be finite and nonnegative")
    if args.eval_every_actor_updates < 0:
        raise ValueError("eval_every_actor_updates must be nonnegative")
    if args.eval_every_old_policy_syncs < 0:
        raise ValueError("eval_every_old_policy_syncs must be nonnegative")
    if args.checkpoint_every_old_policy_syncs < 0:
        raise ValueError("checkpoint_every_old_policy_syncs must be nonnegative")
    for name in ("eval_episodes", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_inference_steps is not None and args.eval_inference_steps <= 0:
        raise ValueError("eval_inference_steps must be positive when provided")
    if not args.eval_env_type:
        raise ValueError("eval_env_type must be nonempty")
    if args.old_policy_sync_target == 0 and args.actor_steps == 0:
        raise ValueError("at least one of old_policy_sync_target or actor_steps must be positive")
    torch.manual_seed(args.seed)

    current_checkpoint = CheckpointAdapter.load(args.checkpoint, device=args.device)
    old_checkpoint = CheckpointAdapter.load(args.checkpoint, device=args.device)
    trace_config = TraceConfig(
        num_inference_steps=args.inference_steps,
        probability_sigma_min=args.probability_sigma_min,
    )
    current = DiffusionRLAdapter(current_checkpoint, trace_config)
    old = DiffusionRLAdapter(old_checkpoint, trace_config)
    policy_config = current.policy.config
    state_dim = policy_config.robot_state_feature.shape[0]
    action_dim = policy_config.action_feature.shape[0]
    rl_config = RLConfig(
        amq=AMQConfig(
            enabled=args.amq_enabled,
            dynamics_steps=args.dynamics_steps,
            rollout_horizon=args.amq_rollout_horizon,
            eval_interval=args.amq_eval_interval,
            min_dynamics_updates=args.amq_min_dynamics_updates,
            relative_margin=args.amq_relative_margin,
            max_validation_loss=args.amq_max_validation_loss,
            max_disagreement=args.amq_max_disagreement,
            discounted=args.amq_discounted,
            ensemble_size=args.amq_ensemble_size,
            use_critic_reference=args.amq_use_critic_reference,
        ),
        trace=trace_config,
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
    if args.camera_cache == "ram":
        cache_started = time.monotonic()
        cache_summary = dataset.preload_camera_frame_cache()
        cache_summary = dict(cache_summary)
        cache_summary["camera_cache_seconds"] = time.monotonic() - cache_started
        print(json.dumps(cache_summary, sort_keys=True))
    dataset_summary = dataset.inspection_summary()
    print(json.dumps(dataset_summary, sort_keys=True))
    if args.amq_enabled:
        train_dataset, validation_dataset = dataset.split_by_episode(validation_fraction=0.1)
    else:
        train_dataset, validation_dataset = dataset, None
    batches = _batches(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    validation_batches = (
        _batches(
            validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if validation_dataset is not None
        else None
    )
    is_dp3 = isinstance(policy_config, DP3Config)
    if is_dp3:
        # Match RL-100 3D: IQL and dynamics operate on the DP3 latent emitted
        # by the multimodal point-cloud/RGB/state encoder.
        encoder = DP3FeatureEncoder(current.policy)
    else:
        encoder = StateFeatureEncoder(
            state_dim=state_dim,
            n_obs_steps=policy_config.n_obs_steps,
            hidden_dims=(256, 256),
            output_dim=256,
            ignore_extra_features=True,
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
    dynamics = None
    amq_evaluator = None
    promotion_gate = None
    if args.amq_enabled:
        if is_dp3:
            dynamics = DP3FeatureDynamicsEnsemble(
                feature_encoder=encoder,
                n_obs_steps=policy_config.n_obs_steps,
                action_dim=action_dim,
                chunk_size=policy_config.n_action_steps,
                active_action_mask=current.checkpoint.active_action_mask,
                hidden_dims=(256, 256),
                ensemble_size=rl_config.amq.ensemble_size,
                learning_rate=3e-4,
                prediction_mode="last",
            ).to(next(current.policy.parameters()).device)
        else:
            dynamics = StateDynamicsEnsemble(
                feature_encoder=encoder,
                state_dim=state_dim,
                n_obs_steps=policy_config.n_obs_steps,
                action_dim=action_dim,
                chunk_size=policy_config.n_action_steps,
                active_action_mask=current.checkpoint.active_action_mask,
                hidden_dims=(256, 256),
                ensemble_size=rl_config.amq.ensemble_size,
                learning_rate=3e-4,
                state_key=rl_config.state_key,
            ).to(next(current.policy.parameters()).device)
        amq_evaluator = AMQEvaluator(
            dynamics=dynamics,
            iql=iql,
            candidate_policy=current,
            behavior_policy=old,
            gamma=rl_config.gamma,
            rollout_horizon=args.amq_rollout_horizon,
            state_key=rl_config.state_key,
            discounted=rl_config.amq.discounted,
        )
        promotion_gate = PolicyPromotionGate(
            relative_margin=args.amq_relative_margin,
            max_validation_loss=args.amq_max_validation_loss,
            max_rollout_disagreement=args.amq_max_disagreement,
            use_critic_reference=rl_config.amq.use_critic_reference,
            inclusive_margin=rl_config.amq.inclusive_margin,
        )
    actor_optimizer = torch.optim.Adam(current.policy.parameters(), lr=args.actor_lr)
    provenance = _build_provenance(
        adapter=current_checkpoint,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        summary=args.summary,
        config=rl_config,
    )
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
            old_policy_sync_interval=0 if args.amq_enabled else args.old_policy_sync_interval,
            ppo_epochs=args.ppo_epochs,
            dynamics=dynamics,
            amq_evaluator=amq_evaluator,
            promotion_gate=promotion_gate,
            tracker=tracker,
            debug=args.debug,
        )
        for _ in range(args.iql_steps):
            trainer.record_metrics(trainer.train_iql_step(next(batches)), 0)
        if dynamics is not None:
            for _ in range(args.dynamics_steps):
                trainer.record_metrics(trainer.train_dynamics_step(next(batches)), 0)
        actor_iteration = 0
        last_eval_sync_count = -1
        last_checkpoint_sync_count = -1
        last_amq_eval_bucket = -1
        while (
            args.old_policy_sync_target > 0
            and trainer.counters.old_policy_syncs < args.old_policy_sync_target
        ) or (
            args.old_policy_sync_target == 0
            and (args.actor_steps == 0 or trainer.counters.actor_updates < args.actor_steps)
        ):
            if args.actor_steps > 0 and trainer.counters.actor_updates >= args.actor_steps:
                raise RuntimeError(
                    "actor update cap reached before old-policy sync target: "
                    f"{trainer.counters.actor_updates}/{args.actor_steps} updates, "
                    f"{trainer.counters.old_policy_syncs}/{args.old_policy_sync_target} syncs"
                )
            actor_batch = next(batches)
            actor_epochs = args.ppo_epochs
            if args.actor_steps > 0:
                actor_epochs = min(actor_epochs, args.actor_steps - trainer.counters.actor_updates)
            metrics = trainer.train_actor_step(
                actor_batch,
                generator=current.make_generator(args.seed + actor_iteration + 1),
                ppo_epochs=actor_epochs,
            )
            if dynamics is not None:
                metrics.update(trainer.train_dynamics_step(actor_batch))
                # AM-Q evaluations are scheduled by actor-update count.  Use
                # crossed interval buckets rather than an exact modulo test:
                # with PPO epochs > 1, actor_updates can jump over an
                # interval boundary (e.g. 48 -> 52 for interval 50), which
                # would otherwise prevent any promotion and leave a
                # sync-targeted run spinning forever.
                amq_bucket = (
                    trainer.counters.actor_updates // args.amq_eval_interval
                    if args.amq_eval_interval > 0
                    else -1
                )
                if (
                    trainer.counters.dynamics_updates >= args.amq_min_dynamics_updates
                    and amq_bucket > last_amq_eval_bucket
                ):
                    if validation_batches is None:
                        raise RuntimeError("AM-Q validation batches were not initialized")
                    metrics.update(
                        trainer.evaluate_amq(
                            next(validation_batches),
                            seed=args.seed + 100_000 + actor_iteration,
                        )
                    )
                    last_amq_eval_bucket = amq_bucket
            actor_updates = trainer.counters.actor_updates
            sync_count = trainer.counters.old_policy_syncs
            sync_eval_due = (
                args.eval_every_old_policy_syncs > 0
                and sync_count > 0
                and sync_count % args.eval_every_old_policy_syncs == 0
                and sync_count != last_eval_sync_count
            )
            actor_eval_due = (
                args.eval_every_actor_updates > 0
                and actor_updates > 0
                and actor_updates % args.eval_every_actor_updates == 0
            )
            sync_checkpoint_due = (
                args.checkpoint_every_old_policy_syncs > 0
                and sync_count > 0
                and sync_count % args.checkpoint_every_old_policy_syncs == 0
                and sync_count != last_checkpoint_sync_count
            )
            if sync_checkpoint_due or sync_eval_due or actor_eval_due:
                if sync_checkpoint_due or sync_eval_due:
                    checkpoint_label = f"sync_{sync_count:03d}"
                    if sync_eval_due:
                        last_eval_sync_count = sync_count
                    if sync_checkpoint_due:
                        last_checkpoint_sync_count = sync_count
                else:
                    checkpoint_label = f"actor_{actor_updates:06d}"
                # Persist a standard LeRobot policy checkpoint for the
                # evaluator. The full RL state is persisted once at the end;
                # these sync checkpoints intentionally contain only weights
                # and processor artifacts so periodic saves stay inexpensive.
                snapshot_root = args.output_dir / "checkpoints" / checkpoint_label / "pretrained_model"
                snapshot_path = _save_eval_policy_snapshot(current_checkpoint, snapshot_root)
                if sync_eval_due or actor_eval_due:
                    eval_root = args.output_dir / "eval" / checkpoint_label
                    eval_metrics = _run_newton_eval(
                        policy_path=snapshot_path,
                        output_dir=eval_root,
                        episodes=args.eval_episodes,
                        batch_size=args.eval_batch_size,
                        inference_steps=args.eval_inference_steps or args.inference_steps,
                        policy_device=args.eval_device or args.device,
                        env_device=args.eval_env_device,
                        seed=args.seed,
                        env_type=args.eval_env_type,
                        executable=args.eval_executable,
                    )
                    metrics.update(eval_metrics)
                    metrics["eval/actor_updates"] = float(actor_updates)
                    metrics["eval/old_policy_sync_count"] = float(sync_count)
                metrics["checkpoint/old_policy_sync_count"] = float(sync_count)
            trainer.record_metrics(metrics, 1)
            actor_iteration += 1
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
