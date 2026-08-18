# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect state-only Diffusion Policy rollouts in the headless Moya simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.moya_newton import create_moya_env
from RL.collectors.moya_il import (
    ACTION_DIM,
    STATE_DIM,
    CollectionResult,
    collect_rollouts,
    publish_collection,
)


def _sha256(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""

    if not path.is_file():
        raise ValueError(f"checkpoint artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--repo-id", "--repo_id", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=930)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _resolve_policy_checkpoint(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"checkpoint must not be a symlink, got {path}")
    root = path.resolve(strict=True)
    if root.is_file():
        raise ValueError(f"checkpoint must be a directory, got {path}")
    if (root / "model.safetensors").is_file():
        return root
    nested = root / "pretrained_model"
    if not nested.is_symlink() and (nested / "model.safetensors").is_file():
        return nested
    final = root / "checkpoints" / "final" / "pretrained_model"
    if not final.is_symlink() and (final / "model.safetensors").is_file():
        return final
    raise ValueError(
        "checkpoint must contain model.safetensors (or a pretrained_model subdirectory): "
        f"{root}"
    )


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in ("episodes", "num_envs", "episode_length", "inference_steps"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError(f"seed must be an integer, got {args.seed!r}")
    if not isinstance(args.repo_id, str) or not args.repo_id.strip():
        raise ValueError("repo-id must be a nonempty string")


class CheckpointPolicyRunner:
    """Run a saved LeRobot Diffusion Policy on raw Moya state batches."""

    def __init__(
        self, checkpoint: CheckpointAdapter, *, inference_steps: int | None = None
    ) -> None:
        if not isinstance(checkpoint, CheckpointAdapter):
            raise ValueError("checkpoint must be a CheckpointAdapter")
        config = checkpoint.policy.config
        state_feature = config.robot_state_feature
        action_feature = config.action_feature
        if state_feature is None or tuple(state_feature.shape) != (STATE_DIM,):
            raise ValueError(
                f"Moya IL collection requires a {STATE_DIM}D robot state, got {state_feature!r}"
            )
        if action_feature is None or tuple(action_feature.shape) != (ACTION_DIM,):
            raise ValueError(
                f"Moya IL collection requires a {ACTION_DIM}D action, got {action_feature!r}"
            )
        image_features = tuple(config.image_features)
        if image_features:
            raise ValueError(
                "Moya IL collection is state-only; checkpoint declares image features "
                f"{image_features!r}"
            )
        self.checkpoint = checkpoint
        self.policy = checkpoint.policy
        try:
            self.device = next(self.policy.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        preferred_state_keys = (
            "observation.state",
            "observation.environment_state",
            "environment_state",
            "state",
        )
        state_key = next(
            (
                key
                for key in preferred_state_keys
                if key in config.input_features
                and config.input_features[key] == state_feature
            ),
            None,
        )
        if state_key is None:
            candidates = [
                key
                for key, feature in config.input_features.items()
                if tuple(feature.shape) == (STATE_DIM,)
                and str(getattr(feature.type, "value", feature.type)).upper() == "STATE"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "could not identify the checkpoint state input feature; "
                    f"candidates={candidates!r}"
                )
            state_key = candidates[0]
        self.state_key = state_key
        self.n_obs_steps = int(config.n_obs_steps)
        if self.n_obs_steps <= 0:
            raise ValueError("policy n_obs_steps must be positive")
        if inference_steps is not None:
            if (
                isinstance(inference_steps, bool)
                or not isinstance(inference_steps, int)
                or inference_steps <= 0
            ):
                raise ValueError("inference_steps must be a positive integer")
            self.policy.config.num_inference_steps = inference_steps
            self.policy.diffusion.num_inference_steps = inference_steps

    def reset(self) -> None:
        for processor in (self.checkpoint.preprocessor, self.checkpoint.postprocessor):
            reset_processor = getattr(processor, "reset", None)
            if callable(reset_processor):
                reset_processor()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(self, raw_states: np.ndarray) -> np.ndarray:
        states = np.asarray(raw_states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1:] != (STATE_DIM,):
            raise ValueError(f"raw state must have shape [batch,{STATE_DIM}], got {states.shape}")
        if not np.all(np.isfinite(states)):
            raise ValueError("raw state must contain only finite values")
        observation = {
            self.state_key: torch.as_tensor(states, device=self.device)
        }
        with torch.inference_mode():
            processed = self.checkpoint.preprocessor(observation)
            action = self.policy.select_action(processed)
            action = self.checkpoint.postprocessor(action)
        if not isinstance(action, torch.Tensor):
            raise ValueError("postprocessor must return a torch.Tensor action")
        expected = (states.shape[0], ACTION_DIM)
        if tuple(action.shape) != expected:
            raise ValueError(f"postprocessed action must have shape {expected}, got {tuple(action.shape)}")
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(action).all().item():
            raise ValueError("postprocessed action must contain only finite values")
        return action.numpy()


def _contract_from_env(env: Any) -> dict[str, Any]:
    getter = getattr(env, "environment_contract", None)
    if not callable(getter):
        raise RuntimeError("Moya environment does not expose environment_contract()")
    contract = getter()
    if not isinstance(contract, Mapping):
        raise RuntimeError("Moya environment contract must be a mapping")
    schema_version = contract.get("schema_version")
    payload = contract.get("payload")
    digest = contract.get("sha256")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise RuntimeError("Moya environment contract schema_version is malformed")
    if not isinstance(payload, Mapping) or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("Moya environment contract payload/hash is malformed")
    try:
        int(digest, 16)
        return json.loads(json.dumps(contract, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Moya environment contract must be strict JSON with a hex SHA-256") from exc


def _summary(
    *,
    checkpoint: Path,
    adapter: CheckpointAdapter,
    environment_contract: Mapping[str, Any],
    args: argparse.Namespace,
    result: CollectionResult,
) -> dict[str, Any]:
    model_path = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint is missing config.json: {config_path}")
    return {
        "collector": "RL.cli.collect_moya_il",
        "task": "moya_charger_grasp",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": _sha256(model_path),
        "checkpoint_config_sha256": _sha256(config_path),
        "processor_fingerprint": adapter.processor_fingerprint(),
        "active_action_mask": [
            bool(value) for value in adapter.active_action_mask.detach().cpu().tolist()
        ],
        "environment_contract": dict(environment_contract),
        "environment_contract_schema_version": int(
            environment_contract["schema_version"]
        ),
        "environment_contract_sha256": str(environment_contract["sha256"]),
        "episodes_requested": int(args.episodes),
        "num_envs": int(args.num_envs),
        "episode_length": int(args.episode_length),
        "inference_steps": int(args.inference_steps),
        "base_seed": int(args.seed),
        "device": str(args.device),
        "sim_device": str(args.sim_device),
        "batches": list(result.batches),
        "success_count": sum(bool(episode.metadata.get("success")) for episode in result.episodes),
        "failure_count": sum(not bool(episode.metadata.get("success")) for episode in result.episodes),
    }


def run(args: argparse.Namespace) -> Path:
    """Run collection and atomically publish the resulting v3 dataset."""

    if getattr(args, "smoke", False):
        args.episodes = 2
        args.num_envs = 2
        args.inference_steps = 2
    _validate_positive_args(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"final output path already exists: {output_dir}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA collection requested but torch.cuda.is_available() is false")
    checkpoint_root = _resolve_policy_checkpoint(Path(args.checkpoint))
    adapter = CheckpointAdapter.load(checkpoint_root, device=args.device)
    runner = CheckpointPolicyRunner(adapter, inference_steps=args.inference_steps)
    env = create_moya_env(
        num_envs=args.num_envs,
        device=args.sim_device,
        episode_length=args.episode_length,
        headless=True,
    )
    try:
        environment_contract = _contract_from_env(env)
        result = collect_rollouts(
            env,
            runner,
            target_episodes=args.episodes,
            episode_length=args.episode_length,
            base_seed=args.seed,
        )
        summary = _summary(
            checkpoint=checkpoint_root,
            adapter=adapter,
            environment_contract=environment_contract,
            args=args,
            result=result,
        )
        return publish_collection(
            output_dir,
            repo_id=args.repo_id,
            episodes=result.episodes,
            summary=summary,
            fps=60,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
