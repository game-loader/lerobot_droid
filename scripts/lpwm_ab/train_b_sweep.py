"""Authorized B-only LPWM-FM sweep; immutable exports and synchronous rollout gates.

Production defaults are the fixed 30k recipe. No overwrite is supported; only the full-task adapter may enable verified resume.
The parent supplies an eval runner (wrapping evaluate.py plus its own SwanLab run).
Its JSON must use evaluate.py's status/num_episodes/checkpoint.step/per_task schema.
Only explicit --preflight --disable-eval can bypass checkpoint rollout evaluation.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

TOTAL_STEPS = 30_000
WARMUP_STEPS = 500
PEAK_LR = 1e-4
MIN_LR = 1e-5
SAVE_EVERY = 5_000
COMPLETION_FILE = "complete.json"
REQUIRED_ARTIFACTS = {
    "model.safetensors",
    "config.json",
    "state_normalization.json",
    "experiment.json",
    "split.json",
    "language_metadata.json",
    "language_embeddings.npy",
    "language_masks.npy",
    "optimizer.pt",
}


def cosine_lr(step, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS, lr=PEAK_LR, min_lr=MIN_LR):
    """One-indexed optimizer updates: warmup ends at 500, cosine ends at 30000."""
    if not 0 < warmup_steps < total_steps or not 0 < min_lr <= lr:
        raise ValueError("Require 0 < warmup < cosine endpoint and 0 < min_lr <= lr")
    if not 0 <= step <= total_steps:
        raise ValueError("Step outside the authorized cosine schedule")
    if step <= warmup_steps:
        return lr * step / warmup_steps
    fraction = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * fraction)) / 2


@lru_cache(maxsize=1)
def training_helpers():
    """Reuse train.py verbatim, including its script-local data import."""
    folder = str(Path(__file__).resolve().parent)
    spec = importlib.util.spec_from_file_location("_lpwm_ab_sweep_base", Path(folder) / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
    return module


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def isolated_rng():
    """Include Python/NumPy: train.py.validate also reseeds these generators."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def save_checkpoint(policy, optimizer, step, output, args, split, stats, generator, initial_hash):
    """Never replace/prune step directories. Completion marker is written LAST.

    Interrupted exports remain visibly incomplete, cannot be reused, and never
    acquire aliases. All files (including optimizer and all RNG states) are hashed.
    """
    destination = output / "checkpoints" / f"step_{step:06d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # exclusive creation: even an incomplete export is immutable
    policy.save_pretrained(destination)
    atomic_json(destination / "state_normalization.json", stats)
    atomic_json(destination / "split.json", split)
    atomic_json(
        destination / "experiment.json",
        {
            "step": step,
            "variant": "B",
            "seed": 42,
            "split_sha256": split["sha256"],
            "initial_weights_sha256": initial_hash,
            "dataset_manifest_sha256": file_sha256(args.data / "manifest.json"),
            "action_normalization": "identity",
            "language_embedding_file": "language_embeddings.npy",
            "language_mask_file": "language_masks.npy",
            "camera_keys": list(policy.config.image_features),
            "action_condition": "GT clean only",
            "action_token_repeat": 3,
            "preflight": args.preflight,
        },
    )
    manifest = json.loads((args.data / "manifest.json").read_text())
    atomic_json(destination / "language_metadata.json", {key: manifest[key] for key in ("language", "tasks")})
    for name in ("language_embeddings.npy", "language_masks.npy"):
        shutil.copyfile(args.data / name, destination / name)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            "dataloader_rng": generator.get_state(),
            "schedule": {
                "warmup_steps": WARMUP_STEPS,
                "total_steps": args.schedule_steps,
                "lr": PEAK_LR,
                "min_lr": MIN_LR,
            },
        },
        destination / "optimizer.pt",
    )
    if getattr(args, "full_task_count", None) is not None:
        shutil.copyfile(output / "task_catalog.json", destination / "task_catalog.json")
    files = {str(path.relative_to(destination)): path for path in destination.rglob("*") if path.is_file()}
    if not files.keys() >= REQUIRED_ARTIFACTS:
        raise RuntimeError("Incomplete native policy export; refusing completion marker")
    hashes = {}
    for name, path in sorted(files.items()):
        if path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"Invalid checkpoint artifact: {name}")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        hashes[name] = file_sha256(path)
    atomic_json(
        destination / COMPLETION_FILE,
        {
            "schema_version": 1,
            "status": "complete",
            "step": step,
            "initial_weights_sha256": initial_hash,
            "split_sha256": split["sha256"],
            "preflight": args.preflight,
            "sha256": hashes,
        },
    )
    return destination


def verify_checkpoint(checkpoint, step):
    manifest = json.loads((checkpoint / COMPLETION_FILE).read_text())
    if manifest.get("status") != "complete" or manifest.get("step") != step:
        raise ValueError("Checkpoint completion/step mismatch")
    hashes = manifest.get("sha256", {})
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
    if not hashes.keys() >= REQUIRED_ARTIFACTS or actual != set(hashes) | {COMPLETION_FILE}:
        raise ValueError("Incomplete checkpoint artifact manifest")
    for name, expected in hashes.items():
        path = checkpoint / name
        if path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    return manifest


def checkpoint_alias(checkpoint, name):
    if name not in {"latest", "best_loss", "final"}:
        raise ValueError("Only latest/best_loss/final symlink aliases are allowed")
    step = json.loads((checkpoint / COMPLETION_FILE).read_text())["step"]
    verify_checkpoint(checkpoint, step)
    alias = checkpoint.parent / name
    if name == "final" and (alias.exists() or alias.is_symlink()):
        raise FileExistsError("The final alias is immutable")
    if alias.exists() and not alias.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink alias: {alias}")
    temporary = checkpoint.parent / f".{name}.tmp"
    temporary.symlink_to(checkpoint.name, target_is_directory=True)
    temporary.replace(alias)


def validate_eval_result(result, step, *, task_ids=None, episodes_per_task=10, preflight=False):
    """Fail closed on incomplete/wrong-step results or inconsistent task totals."""
    task_ids = list(range(10)) if task_ids is None else task_ids
    if not preflight and (task_ids != list(range(10)) or episodes_per_task != 10):
        raise ValueError("Reduced rollout budgets require explicit preflight")
    expected_episodes = len(task_ids) * episodes_per_task
    if not isinstance(result, dict) or result.get("status") != "complete":
        raise ValueError("Evaluation status must be complete")
    if type(result.get("num_episodes")) is not int or result["num_episodes"] != expected_episodes:
        raise ValueError(f"Evaluation must contain exactly {expected_episodes} episodes")
    checkpoint_info = result.get("checkpoint", {})
    if not isinstance(checkpoint_info, dict):
        raise ValueError("Invalid evaluation checkpoint metadata")
    reported_step = checkpoint_info.get("step", result.get("step"))
    if type(reported_step) is not int or reported_step != step:
        raise ValueError("Evaluation checkpoint step mismatch")
    if "step" in result and result["step"] != step:
        raise ValueError("Evaluation top-level step mismatch")

    def summary(row, episodes):
        if type(row.get("num_episodes")) is not int or row["num_episodes"] != episodes:
            raise ValueError("Evaluation per-task episode count mismatch")
        successes = row.get("successes")
        rate = row.get("success_rate")
        if type(successes) is not int or not 0 <= successes <= episodes:
            raise ValueError("Invalid evaluation success count")
        if type(rate) not in (int, float) or not math.isfinite(rate):
            raise ValueError("Invalid evaluation success rate")
        if not math.isclose(rate, successes / episodes, abs_tol=1e-8):
            raise ValueError("Evaluation success rate/count mismatch")
        if "pc_success" in row and not math.isclose(row["pc_success"], 100 * rate, abs_tol=1e-6):
            raise ValueError("Evaluation percentage/count mismatch")
        return successes, rate

    successes, rate = summary(result, expected_episodes)
    tasks = result.get("per_task")
    if not isinstance(tasks, list) or len(tasks) != len(task_ids):
        raise ValueError("Evaluation must report every requested task")
    logged = {
        "rollout/success_rate": rate,
        "rollout/pc_success": 100 * rate,
        "rollout/successes": successes,
        "rollout/num_episodes": expected_episodes,
    }
    seen_ids, total = set(), 0
    for row in tasks:
        if not isinstance(row, dict):
            raise ValueError("Invalid evaluation task result")
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id not in task_ids or task_id in seen_ids:
            raise ValueError("Invalid or duplicate evaluation task ID")
        seen_ids.add(task_id)
        count, task_rate = summary(row, episodes_per_task)
        total += count
        logged[f"rollout/task_{task_id:02d}/success_rate"] = task_rate
        logged[f"rollout/task_{task_id:02d}/successes"] = count
    if total != successes:
        raise ValueError("Evaluation aggregate/per-task success mismatch")
    if preflight:
        logged = {key.replace("rollout/", "preflight_rollout/", 1): value for key, value in logged.items()}
    return logged


def verify_eval_upload(output, *, preflight):
    """Require server-verified telemetry bound to this exact evaluation JSON."""
    sidecar = output.with_suffix(".swanlab.json")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise RuntimeError("Evaluation has no regular SwanLab upload sidecar")
    metadata = json.loads(sidecar.read_text())
    if (
        not isinstance(metadata, dict)
        or metadata.get("status") != "complete"
        or metadata.get("verified_upload") is not True
        or metadata.get("formal_result_eligible") is not (not preflight)
    ):
        raise RuntimeError(
            "Evaluation SwanLab upload is unverified/incomplete or has wrong formal eligibility"
        )
    if metadata.get("result_sha256") != file_sha256(output):
        raise RuntimeError("Evaluation SwanLab upload sidecar refers to a different result")


def run_checkpoint_eval(args, checkpoint, step):
    """Blocking child, no shell, inherited CUDA environment, exclusive result/log paths."""
    before = verify_checkpoint(checkpoint, step)
    output = args.output / "eval" / f"step_{step:06d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".swanlab.json").exists():
        raise FileExistsError(f"Refusing to reuse an evaluation result/upload sidecar: {output}")
    command = [
        str(args.eval_python),
        str(args.eval_runner),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
        "--credential-file",
        str(args.credential_file.resolve()),
        "--project",
        args.project,
        "--run-name",
        f"{args.run_name}-eval-step_{step:06d}",
        "--episodes-per-task",
        str(args.eval_episodes_per_task),
        "--seed",
        "42",
        "--seed-namespace",
        "validation",
    ]
    if args.preflight:
        command.append("--preflight")
    if args.eval_max_steps is not None:
        command.extend(["--max-steps", str(args.eval_max_steps)])
    if args.eval_task_ids is not None:
        command.extend(["--task-ids", *map(str, args.eval_task_ids)])
    # subprocess.run waits until the runner exits; training cannot advance in parallel.
    with output.with_suffix(".log").open("x") as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if child.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed at step {step} (exit {child.returncode}); see {output.with_suffix('.log')}"
        )
    if not output.is_file():
        raise RuntimeError(f"Evaluation produced no result at step {step}")
    result = json.loads(output.read_text())
    logged = validate_eval_result(
        result,
        step,
        task_ids=args.eval_task_ids,
        episodes_per_task=args.eval_episodes_per_task,
        preflight=args.preflight,
    )
    if verify_checkpoint(checkpoint, step) != before:
        raise ValueError("Checkpoint changed during evaluation")
    returned_checkpoint = result.get("checkpoint", {})
    if "path" in returned_checkpoint and Path(returned_checkpoint["path"]).resolve() != checkpoint.resolve():
        raise ValueError("Evaluation used a different checkpoint path")
    if (
        "model_sha256" in returned_checkpoint
        and returned_checkpoint["model_sha256"] != before["sha256"]["model.safetensors"]
    ):
        raise ValueError("Evaluation used different model weights")
    protocol = result.get("protocol", {})
    for key, value in (
        ("seed", 42),
        ("seed_namespace", "validation"),
        ("episodes_per_task", args.eval_episodes_per_task),
    ):
        if key in protocol and protocol[key] != value:
            raise ValueError(f"Evaluation protocol mismatch: {key}")
    verify_eval_upload(output, preflight=args.preflight)
    return logged


def encoder_gradient_diagnostics(policy, batch, step):
    """Fresh sequential graphs on actual DLP encoder parameters, never backward().

    FM gradients are detached/offloaded to CPU before constructing the world graph.
    RNG, buffers (including BatchNorm statistics), modes and existing .grad fields
    are unchanged. Diagnostics run on the last real microbatch, not synthetic data.
    """
    parameters = tuple(p for p in policy.world_model.encoder.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("No trainable shared DLP encoder parameters")
    buffers = [(buffer, buffer.detach().clone()) for buffer in policy.buffers()]

    def gradients(world):
        if world:
            loss, _ = policy.world_model.world_loss(
                batch["world.images"], batch["world.actions"].detach(), current_step=step
            )
            loss = loss * policy._world_scale(step)
        else:
            loss, _ = policy.flow_matching_loss(batch)
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        # Do not flatten on GPU or retain either forward graph across this return.
        return [
            torch.zeros_like(p, device="cpu") if g is None else g.detach().cpu()
            for p, g in zip(parameters, grads, strict=True)
        ]

    def restore_buffers():
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)

    try:
        with isolated_rng():
            fm = gradients(False)
        restore_buffers()
        with isolated_rng():
            world = gradients(True)
        fm_sq = sum(float(g.double().square().sum()) for g in fm)
        world_sq = sum(float(g.double().square().sum()) for g in world)
        dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(fm, world, strict=True))
        fm_norm, world_norm = math.sqrt(fm_sq), math.sqrt(world_sq)
        values = {
            "gradient/fm_encoder_norm": fm_norm,
            "gradient/weighted_world_encoder_norm": world_norm,
            "gradient/world_to_fm_ratio": world_norm / max(fm_norm, 1e-30),
            "gradient/cosine": max(-1.0, min(1.0, dot / max(fm_norm * world_norm, 1e-30))),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite encoder gradient diagnostic")
        return values
    finally:
        restore_buffers()


def parse_args(argv=None, *, allowed_schedule_steps=(TOTAL_STEPS,), allow_resume=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", "--train-steps", "--train_steps", dest="steps", type=int, default=TOTAL_STEPS
    )
    parser.add_argument("--world-weight", "--world_weight", type=float, default=0.1)
    parser.add_argument(
        "--reconstruction-weight", "--rec-weight", "--rec_weight", dest="rec_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--dynamics-weight", "--dyn-weight", "--dyn_weight", dest="dyn_weight", type=float, default=1.0
    )
    parser.add_argument("--prior-weight", "--prior_weight", type=float, default=0.001)
    parser.add_argument("--world-ramp-steps", type=int, choices=(1000,), default=1000)
    parser.add_argument("--lr", type=float, choices=(PEAK_LR,), default=PEAK_LR)
    parser.add_argument("--min-lr", type=float, choices=(MIN_LR,), default=MIN_LR)
    parser.add_argument("--warmup-steps", type=int, choices=(WARMUP_STEPS,), default=WARMUP_STEPS)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--grad-accumulation", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project", "--swanlab-project", default="lpwm-fm-libero-spatial-b-sweep")
    parser.add_argument("--run-name")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--eval-runner", type=Path)
    parser.add_argument("--eval-python", type=Path, default=Path(".venv-eval/bin/python"))
    parser.add_argument("--eval-task-ids", type=int, nargs="+", help="Preflight only: e.g. --eval-task-ids 0")
    parser.add_argument(
        "--eval-episodes-per-task", type=int, default=10, help="Preflight may explicitly request 1"
    )
    parser.add_argument("--eval-max-steps", type=int, help="Preflight only: cap simulator control steps")
    parser.add_argument("--gradient-log-every", type=int, choices=(0, 500), default=500)
    parser.add_argument("--expected-init-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--swanlab-mode", choices=("online", "disabled"), default="online")
    parser.add_argument(
        "--resume", nargs="?", const=True, help="Full-task adapter only: completed checkpoint path"
    )
    parser.add_argument(
        "--resume-eval-result", type=Path, help="Verified completed evaluation for resumed checkpoint"
    )
    parser.add_argument("--eval-workers", type=int, choices=(1, 8), default=1)
    parser.add_argument("--schedule-steps", type=int, choices=allowed_schedule_steps, default=TOTAL_STEPS)
    args = parser.parse_args(argv)
    # These are not tunable in the authorized sweep.
    args.variant = "B"
    args.log_every, args.validate_every, args.save_every = 10, 500, SAVE_EVERY
    args.run_name = args.run_name or args.output.name
    if args.batch_size * args.grad_accumulation != 32:
        parser.error("batch-size * grad-accumulation must remain 32")
    if args.resume is not None:
        if not allow_resume or args.resume is True:
            parser.error("Resume is not supported here without a full-task checkpoint path")
        args.resume = Path(args.resume)
        if (
            not args.resume.is_dir()
            or args.resume_eval_result is None
            or not args.resume_eval_result.is_file()
        ):
            parser.error("Resume requires an existing checkpoint and verified evaluation result")
    if not 1 <= args.steps <= args.schedule_steps:
        parser.error("--steps must be within the authorized --schedule-steps")
    if not args.preflight and args.steps < WARMUP_STEPS:
        parser.error("Production train_steps must be >= warmup_steps (500); short checks require --preflight")
    if not args.preflight and args.steps != args.schedule_steps:
        parser.error(
            "Production steps must equal the authorized schedule endpoint; use --preflight for shorter checks"
        )
    if not 1 <= args.eval_episodes_per_task <= 10:
        parser.error("--eval-episodes-per-task must be in [1,10]")
    if args.eval_max_steps is not None and not 1 <= args.eval_max_steps <= 280:
        parser.error("--eval-max-steps must be in [1,280]")
    if args.eval_task_ids is not None and (
        len(set(args.eval_task_ids)) != len(args.eval_task_ids)
        or any(task not in range(10) for task in args.eval_task_ids)
    ):
        parser.error("--eval-task-ids must be unique Spatial IDs in [0,9]")
    if not args.preflight and (
        args.eval_task_ids is not None or args.eval_episodes_per_task != 10 or args.eval_max_steps is not None
    ):
        parser.error("Reduced evaluation/task selection requires --preflight")
    if args.workers < 0 or args.validation_batches < 1:
        parser.error("workers must be nonnegative and validation-batches positive")
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) < 0
        for name in ("world_weight", "rec_weight", "dyn_weight", "prior_weight")
    ):
        parser.error("Loss weights must be finite and nonnegative")
    if not args.preflight and (args.disable_eval or args.swanlab_mode != "online"):
        parser.error("Production requires evaluation and online SwanLab; disabling requires --preflight")
    if not args.disable_eval:
        if args.eval_runner is None or not args.eval_runner.is_file():
            parser.error("Evaluation requires --eval-runner PATH (or --preflight --disable-eval)")
        if not args.eval_python.is_file() or not os.access(args.eval_python, os.X_OK):
            parser.error("--eval-python must name the eval environment's executable Python")
        args.eval_runner, args.eval_python = args.eval_runner.absolute(), args.eval_python.absolute()
    if (not args.disable_eval or args.swanlab_mode == "online") and (
        args.credential_file is None or not args.credential_file.is_file()
    ):
        parser.error("Online training/evaluation requires --credential-file PATH")
    return args


def make_config(base, manifest, args):
    return base.LPWMFMConfig(
        device=args.device,
        push_to_hub=False,
        variant="B",
        language_dim=manifest["language"]["hidden_dim"],
        hidden_dim=256,
        world_hidden_dim=256,
        n_heads=8,
        world_n_heads=8,
        scene_n_layers=2,
        expert_n_layers=4,
        world_n_layers=4,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        num_inference_steps=10,
        dropout=0.0,
        action_token_repeat=3,
        freeze_encoder=False,
        world_weight=args.world_weight,
        reconstruction_weight=args.rec_weight,
        dynamics_weight=args.dyn_weight,
        prior_weight=args.prior_weight,
        world_warmup_steps=0,
        world_ramp_steps=1000,
        input_features={
            "observation.state": base.PolicyFeature(type=base.FeatureType.STATE, shape=(8,)),
            **{
                name: base.PolicyFeature(
                    type=base.FeatureType.VISUAL, shape=(3, manifest["image_size"], manifest["image_size"])
                )
                for name in manifest["cameras"]
            },
        },
        output_features={"action": base.PolicyFeature(type=base.FeatureType.ACTION, shape=(7,))},
    )


def claim_output(output):
    """Exclusive persistent claim prevents competing trainers and accidental restart."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output (resume is unsupported): {output}")
    with (output / ".trainer.lock").open("x") as handle:
        handle.write(str(os.getpid()))


def train(args):
    base = training_helpers()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but no GPU is available")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    base.seed_everything(42)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if len(manifest["tasks"]) != getattr(args, "full_task_count", 10) and not args.preflight:
        raise ValueError("Production cache task count differs from the explicit training scope")
    if not args.preflight and (manifest["image_size"] != 128 or len(manifest["cameras"]) != 2):
        raise ValueError("Production sweep requires the existing image128/two-camera cache")
    split = base.build_split(manifest, 42)
    stats = base.training_state_statistics(
        np.load(args.data / "states.npy", mmap_mode="r"), manifest["episodes"], split["train_episode_ids"]
    )
    datasets = [
        base.LPWMCachedDataset(
            args.data, split[key], stats, history=2, action_horizon=16, world_horizon=1, include_world=True
        )
        for key in ("train_episode_ids", "validation_episode_ids")
    ]
    generator = torch.Generator().manual_seed(42)
    train_loader = base.DataLoader(
        datasets[0],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = base.DataLoader(
        datasets[1], batch_size=8, sampler=base.balanced_validation_order(datasets[1], 42), num_workers=0
    )
    if not len(train_loader):
        raise ValueError("Need at least eight complete training windows")
    if not args.preflight and args.validation_batches * 8 < len(manifest["tasks"]):
        raise ValueError("Validation budget must cover every task")
    config = make_config(base, manifest, args)
    policy = base.LPWMFMPolicy(config).to(device).train()
    initial_hash = base.weight_hash(policy)
    for expected, actual, label in (
        (args.expected_init_sha256, initial_hash, "base initialization"),
        (args.expected_split_sha256, split["sha256"], "dataset split"),
    ):
        if expected is not None and expected != actual:
            raise ValueError(f"Unexpected {label} hash: {actual}")
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=PEAK_LR, weight_decay=1e-6, betas=(0.9, 0.999)
    )
    continuation = None
    if args.resume is not None:
        from scripts.lpwm_full.resume import restore_training

        continuation = restore_training(
            args, policy, optimizer, generator, split, stats, initial_hash, len(datasets[0])
        )
        restored_sampler = continuation.pop("sampler")
        train_loader = base.DataLoader(
            datasets[0],
            batch_sampler=restored_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            generator=torch.Generator().manual_seed(242),
        )
        generator = restored_sampler.generator
    experiment = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "credential_file"
    }
    experiment.update(
        {
            "effective_batch_size": 32,
            "initial_weights_sha256": initial_hash,
            "split_sha256": split["sha256"],
            "dataset_manifest_sha256": file_sha256(args.data / "manifest.json"),
            "action_condition": "GT clean only",
            "action_token_repeat": 3,
            "imf_jvp": False,
            "predicted_action_dynamics": False,
            "action_latent_alignment": False,
            "cosine_endpoint_step": args.schedule_steps,
            "best_loss_scope": "saved checkpoints, validation/fm_loss",
            "train_windows": len(datasets[0]),
            "validation_windows": len(datasets[1]),
            "parameters": sum(p.numel() for p in policy.parameters()),
        }
    )
    if continuation is not None:
        experiment["continuation"] = continuation["receipt"]
    atomic_json(args.output / "experiment.json", experiment)
    atomic_json(args.output / "split.json", split)
    atomic_json(args.output / "state_normalization.json", stats)
    config.save_pretrained(args.output)
    swanlab = None
    try:
        if args.swanlab_mode == "online":
            import swanlab as swanlab_module

            swanlab = swanlab_module
            api_key = args.credential_file.read_text().strip()
            if not api_key:
                raise ValueError("Empty SwanLab credential file")
            if swanlab.login(api_key=api_key, save=False) is False:
                raise RuntimeError("SwanLab authentication failed")
            del api_key
            run = swanlab.init(
                project=args.project,
                name=args.run_name,
                group=f"B-only-{args.schedule_steps // 1000}k-cosine",
                job_type="train",
                config=experiment,
                log_dir=str(args.output / "swanlog"),
                mode="online",
                public=False,
            )
            atomic_json(args.output / "swanlab_run.json", {"id": run.id, "url": run.url})
            print(f"SWANLAB_RUN_URL={run.url}", flush=True)
        start_step, best_saved_loss = 0, float("inf")
        if continuation is not None:
            from scripts.lpwm_full.resume import restore_rng

            start_step = continuation["step"]
            best_saved_loss = continuation["best_saved_loss"]
            restore_rng(continuation.pop("rng"), generator)
        iterator = iter(train_loader)
        started = time.monotonic()
        with (args.output / "metrics.jsonl").open("x", buffering=1) as metrics_file:

            def log(values, step):
                values = base.scalar_metrics(values)
                row = {"step": step, **values}
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                if swanlab is not None and swanlab.log(values, step=step) is False:
                    raise RuntimeError("Training SwanLab metric logging failed")

            if continuation is not None:
                log({"system/resumed_from_step": start_step, **continuation["metrics"]}, start_step)
                atomic_json(
                    args.output / "status.json", {"status": "running", "step": start_step, "resumed": True}
                )
            for step in range(start_step + 1, args.steps + 1):
                step_start = time.monotonic()
                lr = cosine_lr(step, total_steps=args.schedule_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accumulated = {}
                for micro in range(args.grad_accumulation):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        batch = next(iterator)
                    batch = base.to_device(batch, device)
                    torch.manual_seed(42 + step * 4 + micro)
                    loss, metrics = policy(batch, current_step=step)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise FloatingPointError(f"Invalid training loss at step {step}")
                    (loss / args.grad_accumulation).backward()
                    values = base.scalar_metrics(metrics)
                    values["total_loss"] = float(loss.detach())
                    for key, value in values.items():
                        accumulated[key] = accumulated.get(key, 0.0) + value / args.grad_accumulation
                    del loss, metrics
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 10.0, error_if_nonfinite=True
                )
                if step == start_step + 1 and not any(
                    p.grad is not None and torch.count_nonzero(p.grad)
                    for p in policy.world_model.encoder.parameters()
                ):
                    raise RuntimeError("Shared DLP encoder received no gradients")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                logged = {f"train/{key}": value for key, value in accumulated.items()}
                logged.update(
                    {
                        "train/lr": lr,
                        "train/grad_norm": float(gradient_norm),
                        "train/seen_windows": step * 32,
                        "system/step_seconds": time.monotonic() - step_start,
                        "system/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0,
                        "system/cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30
                        if device.type == "cuda"
                        else 0.0,
                    }
                )
                if args.gradient_log_every and (
                    step % args.gradient_log_every == 0 or (args.preflight and step in (1, args.steps))
                ):
                    logged.update(encoder_gradient_diagnostics(policy, batch, step))
                del batch
                if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                    log(logged, step)
                validation = None
                if step % args.validate_every == 0 or step == args.steps:
                    with isolated_rng():
                        validation = base.validate(policy, val_loader, device, args, step)
                    if "validation/fm_loss" not in validation:
                        raise RuntimeError("Native FM validation metric missing")
                    log(validation, step)
                if step % args.save_every == 0 or step == args.steps:
                    checkpoint = save_checkpoint(
                        policy, optimizer, step, args.output, args, split, stats, generator, initial_hash
                    )
                    checkpoint_alias(checkpoint, "latest")
                    if not args.disable_eval:
                        atomic_json(args.output / "status.json", {"status": "evaluating", "step": step})
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            torch.cuda.empty_cache()
                        log(run_checkpoint_eval(args, checkpoint, step), step)
                    saved_loss = validation["validation/fm_loss"]
                    if saved_loss < best_saved_loss:
                        best_saved_loss = saved_loss
                        checkpoint_alias(checkpoint, "best_loss")
                    if step == args.steps:
                        checkpoint_alias(checkpoint, "final")
                if step == start_step + 1 or step % args.log_every == 0:
                    atomic_json(args.output / "status.json", {"status": "running", "step": step})
            completed_status = {
                "status": "completed",
                "step": args.steps,
                "preflight": args.preflight,
                "elapsed_seconds": time.monotonic() - started,
                "best_saved_validation_fm_loss": best_saved_loss,
            }
    finally:
        if swanlab is not None and swanlab.finish() is False:
            raise RuntimeError("Training SwanLab finish failed")
    # Publish success only after final evaluation AND the online run flush succeed.
    atomic_json(args.output / "status.json", completed_status)


def main(argv=None):
    args = parse_args(argv)
    claim_output(args.output)
    try:
        train(args)
    except BaseException as error:
        # Do not persist exception text: external services may include credentials.
        previous = args.output / "status.json"
        status = json.loads(previous.read_text()) if previous.is_file() else {}
        atomic_json(
            previous, {"status": "failed", "step": status.get("step"), "error_type": type(error).__name__}
        )
        raise


if __name__ == "__main__":
    main()
