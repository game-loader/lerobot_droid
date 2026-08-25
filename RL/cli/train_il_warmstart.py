# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Warm-start Diffusion Policy IL while preserving pretrained normalizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

_PROCESSOR_CONFIGS = ("policy_preprocessor.json", "policy_postprocessor.json")
_RUNTIME_STATS = {
    "MEAN_STD": ("mean", "std"),
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}
_SWANLAB_BOOTSTRAP = """
import os
import swanlab

swanlab.sync_wandb(
    mode=os.environ["RL100_SWANLAB_MODE"],
    wandb_run=False,
    log_dir=os.environ["RL100_SWANLAB_LOG_DIR"],
)
from lerobot.scripts.lerobot_train import main

main()
""".strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, required=True)
    parser.add_argument("--repo-id", "--repo_id", required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--save-freq", "--save_freq", type=int, default=5_000)
    parser.add_argument("--log-freq", "--log_freq", type=int, default=100)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every-steps", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-inference-steps", type=int, default=10)
    parser.add_argument("--eval-env-type", default="moya_newton")
    parser.add_argument("--eval-env-device", default="cuda:0")
    parser.add_argument("--swanlab-project", default="lerobot-moya-diffusion")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "offline", "local", "disabled"),
        default="online",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _resolve_checkpoint(path: Path) -> Path:
    root = Path(path).resolve(strict=True)
    candidates = (root, root / "pretrained_model")
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file() and (candidate / "config.json").is_file():
            return candidate
    raise ValueError(f"checkpoint does not contain a pretrained policy bundle: {path}")


def _canonical_tensor(value: torch.Tensor, *, feature_type: str | None = None) -> bytes:
    """Serialize a runtime statistic independently of safetensors shape metadata."""

    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    # ``NormalizerProcessorStep`` reshapes flat visual statistics to ``(C, 1, 1)``
    # during deserialization.  Canonicalize that representation before hashing.
    if feature_type == "VISUAL" and tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1, 1)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("normalization statistics must contain only finite values")
    metadata = json.dumps(
        {"shape": list(tensor.shape), "dtype": "float32"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return metadata + b"\0" + tensor.numpy().tobytes()


def processor_fingerprint(checkpoint: Path) -> str:
    """Hash the normalization contract consumed at runtime.

    Auxiliary dataset statistics (for example ``count`` and episode indices) are
    deliberately excluded.  LeRobot may serialize those as ``(1,)`` or ``()``
    without changing normalization behavior.
    """

    root = Path(checkpoint).resolve(strict=True)
    entries: list[tuple[str, bytes]] = []
    normalization_steps = 0
    for name in _PROCESSOR_CONFIGS:
        config_path = root / name
        if not config_path.is_file():
            raise ValueError(f"checkpoint is missing processor config: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid processor config {config_path}: {exc}") from exc
        steps = payload.get("steps") if isinstance(payload, Mapping) else None
        if not isinstance(steps, list):
            raise ValueError(f"processor config must contain a steps list: {config_path}")
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                continue
            registry_name = step.get("registry_name")
            if registry_name not in ("normalizer_processor", "unnormalizer_processor"):
                continue
            normalization_steps += 1
            config = step.get("config")
            if not isinstance(config, Mapping):
                raise ValueError(f"invalid normalization config in {config_path}")
            features = config.get("features")
            norm_map = config.get("norm_map")
            if not isinstance(features, Mapping) or not isinstance(norm_map, Mapping):
                raise ValueError(f"normalization config is missing features/norm_map: {config_path}")
            semantic_config = json.dumps(
                {"registry_name": registry_name, "config": dict(config)},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            prefix = f"{name}:step_{step_index}:{registry_name}"
            entries.append((f"{prefix}:config", semantic_config))
            state_name = step.get("state_file")
            if not isinstance(state_name, str) or not state_name:
                raise ValueError(f"normalization step is missing state_file in {config_path}")
            state_path = (root / state_name).resolve()
            if not state_path.is_relative_to(root) or not state_path.is_file():
                raise ValueError(f"invalid processor state file: {state_path}")
            try:
                state = load_file(str(state_path), device="cpu")
            except Exception as exc:
                raise ValueError(f"invalid processor state file {state_path}: {exc}") from exc
            for feature_key, feature in sorted(features.items(), key=lambda item: str(item[0])):
                if not isinstance(feature_key, str) or not isinstance(feature, Mapping):
                    raise ValueError(f"invalid normalization feature in {config_path}")
                feature_type = feature.get("type")
                mode = norm_map.get(feature_type, "IDENTITY")
                mode_name = str(mode)
                if mode_name != "IDENTITY" and mode_name not in _RUNTIME_STATS:
                    raise ValueError(f"unsupported normalization mode {mode_name!r} in {config_path}")
                stat_names = _RUNTIME_STATS.get(mode_name, ())
                for stat_name in stat_names:
                    state_key = f"{feature_key}.{stat_name}"
                    value = state.get(state_key)
                    if value is None:
                        raise ValueError(f"normalization state is missing required statistic {state_key}")
                    entries.append(
                        (
                            f"{prefix}:stat:{state_key}",
                            _canonical_tensor(value, feature_type=str(feature_type)),
                        )
                    )
    if normalization_steps == 0:
        raise ValueError(f"checkpoint contains no normalization processor steps: {root}")
    digest = hashlib.sha256()
    for name, content in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def build_train_command(args: argparse.Namespace) -> list[str]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir)
    eval_every_steps = getattr(args, "eval_every_steps", 5_000)
    eval_episodes = getattr(args, "eval_episodes", 100)
    eval_batch_size = getattr(args, "eval_batch_size", 16)
    eval_inference_steps = getattr(args, "eval_inference_steps", 10)
    eval_env_type = getattr(args, "eval_env_type", "moya_newton")
    eval_env_device = getattr(args, "eval_env_device", "cuda:0")
    train_args = [
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={Path(args.dataset_root).resolve(strict=True)}",
        f"--policy.path={checkpoint}",
        f"--policy.device={args.device}",
        f"--policy.optimizer_lr={args.learning_rate}",
        "--policy.push_to_hub=false",
        "--preserve_pretrained_processor_stats=true",
        "--resume=false",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={eval_every_steps}",
        f"--eval.n_episodes={eval_episodes}",
        f"--eval.batch_size={eval_batch_size}",
        "--eval.use_async_envs=false",
        f"--env.type={eval_env_type}",
        f"--env.device={eval_env_device}",
        "--env.headless=true",
        f"--policy.num_inference_steps={eval_inference_steps}",
        f"--seed={args.seed}",
        "--save_checkpoint=true",
        f"--output_dir={output_dir}",
    ]
    if args.swanlab_mode == "disabled":
        return [sys.executable, "-m", "lerobot.scripts.lerobot_train", *train_args]
    train_args.extend(
        (
            "--wandb.enable=true",
            f"--wandb.project={args.swanlab_project}",
            "--wandb.disable_artifact=true",
            "--wandb.mode=online" if args.swanlab_mode == "online" else "--wandb.mode=offline",
        )
    )
    if args.swanlab_run_name:
        train_args.append(f"--job_name={args.swanlab_run_name}")
    return [
        "uv",
        "run",
        "--with",
        "swanlab==0.9.4",
        "python",
        "-c",
        _SWANLAB_BOOTSTRAP,
        *train_args,
    ]


def _resolve_final_checkpoint(output_dir: Path) -> Path:
    last = output_dir / "checkpoints" / "last"
    if not last.is_symlink():
        raise RuntimeError(f"IL training did not create checkpoints/last: {last}")
    checkpoint = (last.resolve(strict=True) / "pretrained_model").resolve(strict=True)
    for name in ("config.json", "model.safetensors", *_PROCESSOR_CONFIGS):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"IL checkpoint is missing {name}: {checkpoint}")
    return checkpoint


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path | list[str]:
    for name in (
        "steps",
        "batch_size",
        "num_workers",
        "save_freq",
        "log_freq",
        "eval_every_steps",
        "eval_episodes",
        "eval_batch_size",
        "eval_inference_steps",
    ):
        _positive_int(name, getattr(args, name))
    if not isinstance(args.repo_id, str) or not args.repo_id.strip():
        raise ValueError("repo-id must be nonempty")
    if not isinstance(args.learning_rate, float) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.eval_every_steps % args.save_freq != 0:
        raise ValueError("eval-every-steps must be a multiple of save-freq")
    for name in ("device", "eval_env_type", "eval_env_device"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"IL output directory already exists: {output_dir}")
    source = _resolve_checkpoint(args.checkpoint)
    source_fingerprint = processor_fingerprint(source)
    command = build_train_command(args)
    if args.dry_run:
        return command

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / "il_train.log"
    env = os.environ.copy()
    env["RL100_SWANLAB_MODE"] = args.swanlab_mode
    env["RL100_SWANLAB_LOG_DIR"] = str(output_dir.parent / "swanlog")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"IL training failed with exit code {completed.returncode}; see {log_path}")
    checkpoint = _resolve_final_checkpoint(output_dir)
    final_fingerprint = processor_fingerprint(checkpoint)
    if final_fingerprint != source_fingerprint:
        raise RuntimeError(
            "IL warm-start changed the pretrained processor fingerprint: "
            f"source={source_fingerprint} final={final_fingerprint}"
        )
    _write_json(
        output_dir.parent / "il_stage.json",
        {
            "complete": True,
            "checkpoint": str(checkpoint),
            "command": command,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "processor_fingerprint": final_fingerprint,
            "repo_id": args.repo_id,
            "source_checkpoint": str(source),
        },
    )
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    if isinstance(result, list):
        print(json.dumps(result))
    else:
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
