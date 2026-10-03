"""Verified full-policy continuation into a NEW output; original runs stay immutable.

Reconstruct legacy RandomSampler permutations and the microbatch offset from
the global step. Require the reconstructed generator state to match the saved
state exactly; never silently restart data epochs.
"""

import json
import math
import random
from pathlib import Path

import draccus
import numpy as np
import torch
from safetensors.torch import load_model

from lerobot.configs.policies import PreTrainedConfig
from scripts.lpwm_ab import train_b_sweep as trainer
from scripts.lpwm_full.evaluate import catalog_suites, protocol_suites, validate_result
from scripts.lpwm_monitor.resume_train import RestoredBatchSampler


def read(path):
    return json.loads(Path(path).read_text())


def verify_resume_inputs(args, policy, split, stats, initial_hash):
    checkpoint = Path(args.resume).resolve()
    completion = read(checkpoint / "complete.json")
    step = completion["step"]
    trainer.verify_checkpoint(checkpoint, step)
    if not 0 < step < args.steps or step % trainer.SAVE_EVERY:
        raise ValueError("Resume must start at a completed earlier 5k checkpoint")
    if completion["preflight"] or completion["initial_weights_sha256"] != initial_hash:
        raise ValueError("Resume checkpoint is preflight or has a different initialization")
    if split != read(checkpoint / "split.json") or stats != read(checkpoint / "state_normalization.json"):
        raise ValueError("Resume split/state normalization differs from original")
    if draccus.encode(policy.config, PreTrainedConfig) != read(checkpoint / "config.json"):
        raise ValueError("Resume model configuration differs from original")
    experiment = read(checkpoint / "experiment.json")
    if experiment["dataset_manifest_sha256"] != trainer.file_sha256(args.data / "manifest.json"):
        raise ValueError("Resume dataset manifest changed")
    parent = checkpoint.parent.parent
    old = read(parent / "experiment.json")
    for key in (
        "batch_size",
        "grad_accumulation",
        "schedule_steps",
        "seed",
        "world_weight",
        "rec_weight",
        "dyn_weight",
        "prior_weight",
        "validation_batches",
        "world_ramp_steps",
        "warmup_steps",
        "lr",
        "min_lr",
        "workers",
    ):
        if old[key] != getattr(args, key):
            raise ValueError(f"Resume changes original training setting: {key}")
    result_path = args.resume_eval_result
    if result_path is None:
        raise ValueError("Resume requires a verified completed checkpoint evaluation")
    result = read(result_path)
    metrics = validate_result(result, preflight=False)
    trainer.verify_eval_upload(result_path, preflight=False)
    expected_suites = catalog_suites(read(checkpoint / "task_catalog.json"))
    if protocol_suites(result["protocol"]) != expected_suites:
        raise ValueError("Resume evaluation has wrong task coverage")
    if (
        result["checkpoint"]["step"] != step
        or result["checkpoint"]["model_sha256"] != completion["sha256"]["model.safetensors"]
    ):
        raise ValueError("Resume evaluation refers to wrong checkpoint")
    return checkpoint, completion, old, metrics


def restore_training(args, policy, optimizer, generator, split, stats, initial_hash, dataset_size):
    checkpoint, completion, old, metrics = verify_resume_inputs(args, policy, split, stats, initial_hash)
    # Verified user-owned pickle; hash verification MUST precede this load.
    state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    required = {
        "step",
        "optimizer",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "dataloader_rng",
        "schedule",
    }
    if not state.keys() >= required or state["step"] != completion["step"]:
        raise ValueError("Resume optimizer/RNG state incomplete or wrong step")
    schedule = {
        "warmup_steps": trainer.WARMUP_STEPS,
        "total_steps": args.schedule_steps,
        "lr": trainer.PEAK_LR,
        "min_lr": trainer.MIN_LR,
    }
    if state["schedule"] != schedule:
        raise ValueError("Resume would restart or change cosine schedule")
    adam_steps = [int(value["step"]) for value in state["optimizer"]["state"].values()]
    if not adam_steps or min(adam_steps) < 1 or max(adam_steps) != state["step"]:
        raise ValueError("Optimizer update counters do not match checkpoint")
    expected_lr = trainer.cosine_lr(state["step"], total_steps=args.schedule_steps)
    if any(
        not math.isclose(group["lr"], expected_lr, rel_tol=1e-10)
        for group in state["optimizer"]["param_groups"]
    ):
        raise ValueError("Saved optimizer LR disagrees with global cosine step")
    if args.workers < 1:
        raise ValueError("Exact legacy sampler reconstruction requires persistent workers")
    sampler = RestoredBatchSampler(
        dataset_size,
        args.batch_size,
        state["step"] * args.grad_accumulation,
        args.steps * args.grad_accumulation,
        state["dataloader_rng"],
    )
    # Native exports deduplicate shared DLP parameter aliases. The safetensors
    # model-aware strict loader validates aliases without requiring duplicate keys.
    load_model(policy, str(checkpoint / "model.safetensors"), strict=True, device="cpu")
    optimizer.load_state_dict(state["optimizer"])
    del state["optimizer"]
    generator.set_state(state["dataloader_rng"])
    parent = checkpoint.parent.parent
    losses = {}
    for line in (parent / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if (
            "validation/fm_loss" in row
            and row["step"] <= state["step"]
            and row["step"] % trainer.SAVE_EVERY == 0
        ):
            losses[row["step"]] = row["validation/fm_loss"]
    if state["step"] not in losses or not all(math.isfinite(value) for value in losses.values()):
        raise ValueError("Missing finite validation loss for resumed checkpoint")
    inherited = []
    destination = args.output / "checkpoints"
    destination.mkdir(exist_ok=True)
    for step in sorted(losses):
        source = parent / "checkpoints" / f"step_{step:06d}"
        trainer.verify_checkpoint(source, step)
        target = destination / source.name
        target.symlink_to(source.resolve(), target_is_directory=True)
        inherited.append({"step": step, "source": str(source.resolve())})
    best_step = min(losses, key=losses.get)
    trainer.checkpoint_alias(destination / f"step_{state['step']:06d}", "latest")
    trainer.checkpoint_alias(destination / f"step_{best_step:06d}", "best_loss")
    receipt = {
        "status": "restored",
        "source_checkpoint": str(checkpoint),
        "source_output": str(parent),
        "source_checkpoint_sha256": completion["sha256"],
        "step": state["step"],
        "next_step": state["step"] + 1,
        "target_step": args.steps,
        "schedule": schedule,
        "previous_lr": expected_lr,
        "next_lr": trainer.cosine_lr(state["step"] + 1, total_steps=args.schedule_steps),
        "optimizer_state_count": len(adam_steps),
        "optimizer_step_min": min(adam_steps),
        "optimizer_step_max": max(adam_steps),
        "source_swanlab": read(parent / "swanlab_run.json"),
        "source_initial_weights_sha256": old["initial_weights_sha256"],
        "inherited_checkpoints": inherited,
        "best_saved_loss": losses[best_step],
        "best_saved_step": best_step,
        "completed_resume_eval": str(args.resume_eval_result.resolve()),
        "rng_restore": "Python/NumPy/Torch/CUDA plus verified reconstructed shuffle generator",
        "loader_continuation": "reconstructed original persistent-worker RandomSampler permutation and microbatch offset; fail closed unless generator state equals checkpoint",
        "sampler_epoch": sampler.epoch,
        "sampler_batch_offset": sampler.offset,
        "sampler_batches_per_epoch": sampler.per_epoch,
        "bitwise_data_order_resume": True,
        "evaluation_scheduler": "task",
        "evaluation_workers": args.eval_workers,
    }
    trainer.atomic_json(args.output / "resume.json", receipt)
    return {
        "step": state["step"],
        "sampler": sampler,
        "best_saved_loss": losses[best_step],
        "rng": state,
        "metrics": {k.replace("eval/", "rollout/", 1): v for k, v in metrics.items()},
        "receipt": receipt,
    }


def restore_rng(state, generator):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    generator.set_state(state["dataloader_rng"])
