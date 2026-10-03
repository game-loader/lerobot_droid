"""Paired single-GPU LPWM-FM A/B training with factual GT dynamics and SwanLab."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from data import LPWMCachedDataset, balanced_validation_order, build_split, training_state_statistics
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.lpwm_fm.configuration_lpwm_fm import LPWMFMConfig
from lerobot.policies.lpwm_fm.modeling_lpwm_fm import LPWMFMPolicy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().float().item()
        if isinstance(value, (float, int, np.floating)):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite metric {key}={value}")
            result[key] = float(value)
    return result


def weight_hash(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def save_checkpoint(policy, optimizer, step, output, args, split, stats, is_best=False):
    destination = output / "checkpoints" / ("best" if is_best else "latest")
    destination.mkdir(parents=True, exist_ok=True)
    # Export actual policy weights/config plus normalization/text-cache metadata.
    policy.save_pretrained(destination)
    atomic_json(destination / "state_normalization.json", stats)
    atomic_json(
        destination / "experiment.json",
        {
            "step": step,
            "variant": args.variant,
            "seed": args.seed,
            "split_sha256": split["sha256"],
            "action_normalization": "identity",
            "language_embedding_file": "language_embeddings.npy",
            "language_mask_file": "language_masks.npy",
            "camera_keys": list(policy.config.image_features),
        },
    )
    manifest = json.loads((args.data / "manifest.json").read_text())
    atomic_json(
        destination / "language_metadata.json", {"language": manifest["language"], "tasks": manifest["tasks"]}
    )
    for name in ("language_embeddings.npy", "language_masks.npy"):
        # Tiny task-level frozen text tensors, not model weights.
        (destination / name).write_bytes((args.data / name).read_bytes())
    if not is_best:
        temporary = destination / "optimizer.pt.tmp"
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            temporary,
        )
        temporary.replace(destination / "optimizer.pt")
    atomic_json(output / "last_checkpoint.json", {"step": step, "path": str(destination), "best": is_best})


@torch.no_grad()
def validate(policy, loader, device, args, step):
    was_training = policy.training
    policy.eval()
    rows = []
    # Keep validation RNG isolated so it cannot change training noise/data streams.
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.validation_batches:
                break
            seed_everything(args.seed + 1_000_000 + batch_index)
            batch = to_device(batch, device)
            loss, metrics = policy(batch, current_step=step)
            row = scalar_metrics(metrics)
            row["total_loss"] = float(loss)
            if batch_index == 0:
                sampled = policy.predict_action_chunk(batch)
                target = batch["action"][:, : sampled.shape[1]]
                row["sampled_action_l1"] = float((sampled - target).abs().mean())
            rows.append(row)
    policy.train(was_training)
    return {
        f"validation/{key}": float(np.mean([row[key] for row in rows if key in row]))
        for key in {key for row in rows for key in row}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--world-weight", type=float, default=0.1)
    parser.add_argument("--world-ramp-steps", type=int, default=1000)
    parser.add_argument("--world-horizon", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swanlab-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--swanlab-project", default="lpwm-fm-libero-spatial-ab")
    parser.add_argument("--run-name")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.batch_size,
            args.grad_accumulation,
            args.log_every,
            args.validate_every,
            args.save_every,
            args.validation_batches,
        )
        < 1
    ):
        parser.error("Step counts and batch settings must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty experiment directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but no GPU is available")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    seed_everything(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    split = build_split(manifest, args.seed)
    states = np.load(args.data / "states.npy", mmap_mode="r")
    stats = training_state_statistics(states, manifest["episodes"], split["train_episode_ids"])
    train_data = LPWMCachedDataset(
        args.data,
        split["train_episode_ids"],
        stats,
        world_horizon=args.world_horizon,
        include_world=args.variant == "B",
    )
    val_data = LPWMCachedDataset(
        args.data,
        split["validation_episode_ids"],
        stats,
        world_horizon=args.world_horizon,
        include_world=args.variant == "B",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        sampler=balanced_validation_order(val_data, args.seed),
        num_workers=0,
    )
    if args.validation_batches * args.batch_size < len(manifest["tasks"]):
        raise ValueError("Validation budget must cover at least one window per task")
    image_size = manifest["image_size"]
    config = LPWMFMConfig(
        device=str(device),
        push_to_hub=False,
        variant=args.variant,
        language_dim=manifest["language"]["hidden_dim"],
        hidden_dim=args.hidden_dim,
        world_hidden_dim=args.hidden_dim,
        dropout=0.0,
        world_weight=args.world_weight,
        world_ramp_steps=args.world_ramp_steps,
        reconstruction_weight=1.0,
        dynamics_weight=1.0,
        prior_weight=1e-3,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            **{
                name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
                for name in manifest["cameras"]
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = LPWMFMPolicy(config).to(device).train()
    initial_hash = weight_hash(policy)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    experiment = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "credential_file"
    }
    experiment.update(
        {
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "initial_weights_sha256": initial_hash,
            "split_sha256": split["sha256"],
            "train_windows": len(train_data),
            "validation_windows": len(val_data),
            "train_episodes": len(split["train_episode_ids"]),
            "validation_episodes": len(split["validation_episode_ids"]),
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "language_encoder": manifest["language"],
            "action_condition": "GT clean only",
            "action_token_repeat": 3,
            "imf_jvp": False,
            "action_latent_alignment": False,
            "predicted_action_dynamics": False,
            "fp32_with_tf32": device.type == "cuda",
            "dataset_temporal_semantics": "one original retained controller action per row transition",
        }
    )
    atomic_json(args.output / "experiment.json", experiment)
    atomic_json(args.output / "split.json", split)
    atomic_json(args.output / "state_normalization.json", stats)
    config.save_pretrained(args.output)
    swanlab = None
    if args.swanlab_mode == "online":
        import swanlab as swanlab_module

        swanlab = swanlab_module
        api_key = os.environ.get("SWANLAB_API_KEY")
        if args.credential_file:
            api_key = args.credential_file.read_text().strip()
        if not api_key:
            raise RuntimeError("Online SwanLab requested but no credential supplied")
        swanlab.login(api_key=api_key, save=False)
        del api_key
        run = swanlab.init(
            project=args.swanlab_project,
            name=args.run_name or f"LPWM-FM-{args.variant}-Spatial-s{args.seed}",
            group="A-vs-B-GT-action-only",
            job_type="train",
            config=experiment,
            log_dir=str(args.output / "swanlog"),
            mode="online",
            public=False,
        )
        atomic_json(args.output / "swanlab_run.json", {"id": run.id, "url": run.url})
        print(f"SWANLAB_RUN_URL={run.url}", flush=True)
    print(json.dumps({"event": "training_start", **experiment}, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(train_loader)
    best_validation = float("inf")
    started = time.monotonic()
    metric_file = (args.output / "metrics.jsonl").open("a", buffering=1)
    try:
        for step in range(1, args.steps + 1):
            step_start = time.monotonic()
            lr_scale = min(1.0, step / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for micro in range(args.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch = to_device(batch, device)
                # Same policy noise stream in paired jobs, regardless of extra B computations.
                torch.manual_seed(args.seed + step * args.grad_accumulation + micro)
                loss, metrics = policy(batch, current_step=step)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError(f"Invalid training loss at step {step}")
                (loss / args.grad_accumulation).backward()
                values = scalar_metrics(metrics)
                values["total_loss"] = float(loss.detach())
                for key, value in values.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / args.grad_accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            if step == 1:
                encoder_grads = [
                    parameter.grad
                    for name, parameter in policy.named_parameters()
                    if "encoder" in name and parameter.grad is not None
                ]
                if not encoder_grads or not any(torch.count_nonzero(grad) for grad in encoder_grads):
                    raise RuntimeError("Shared visual encoder received no gradients")
            optimizer.step()
            if step == 1 or step % args.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                logged = {f"train/{key}": value for key, value in accumulated.items()}
                logged.update(
                    {
                        "train/grad_norm": float(gradient_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "system/step_seconds": time.monotonic() - step_start,
                        "system/elapsed_seconds": time.monotonic() - started,
                        "train/seen_windows": step * args.batch_size * args.grad_accumulation,
                    }
                )
                if device.type == "cuda":
                    logged.update(
                        {
                            "system/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "system/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        }
                    )
                row = {"step": step, **logged}
                metric_file.write(json.dumps(row) + "\n")
                atomic_json(args.output / "status.json", {"status": "running", **row})
                print(json.dumps(row, sort_keys=True), flush=True)
                if swanlab:
                    swanlab.log(logged, step=step)
            if not args.smoke and (step % args.validate_every == 0 or step == args.steps):
                validation = validate(policy, val_loader, device, args, step)
                metric_file.write(json.dumps({"step": step, **validation}) + "\n")
                print(json.dumps({"step": step, **validation}, sort_keys=True), flush=True)
                if swanlab:
                    swanlab.log(validation, step=step)
                # Same action-only metric selects each variant, not incomparable combined world losses.
                key = next(
                    (
                        key
                        for key in ("validation/fm_loss", "validation/action_loss", "validation/flow_loss")
                        if key in validation
                    ),
                    "validation/sampled_action_l1",
                )
                if validation[key] < best_validation:
                    best_validation = validation[key]
                    save_checkpoint(policy, optimizer, step, args.output, args, split, stats, is_best=True)
            if not args.smoke and (step % args.save_every == 0 or step == args.steps):
                save_checkpoint(policy, optimizer, step, args.output, args, split, stats)
        atomic_json(
            args.output / "status.json",
            {
                "status": "completed",
                "step": args.steps,
                "elapsed_seconds": time.monotonic() - started,
                "best_validation": best_validation if math.isfinite(best_validation) else None,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0,
            },
        )
        if swanlab:
            swanlab.finish()
    except BaseException as error:
        atomic_json(
            args.output / "status.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        metric_file.close()


if __name__ == "__main__":
    main()
