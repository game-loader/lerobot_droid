"""Resume a validated B checkpoint after power loss without changing frozen trainer source.

Reconstruct the original RandomSampler permutation and offset. Reject if checkpoint
sampler state does not match; never silently restart data epochs. New log segment and
SwanLab run avoid mixing rolled-back steps into old metric history.
"""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RestoredBatchSampler(Sampler):
    def __init__(self, size, batch_size, start_microbatch, total_microbatches, saved_state, seed=42):
        self.size, self.batch_size, self.start, self.total = (
            size,
            batch_size,
            start_microbatch,
            total_microbatches,
        )
        self.per_epoch = size // batch_size
        if self.per_epoch < 1:
            raise ValueError("Dataset too small")
        self.epoch, self.offset = divmod(start_microbatch, self.per_epoch)
        self.generator = torch.Generator().manual_seed(seed)
        # Original DataLoader draws worker base seed ONCE (persistent workers).
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        for _ in range(self.epoch):
            torch.randperm(size, generator=self.generator)
            torch.randperm(size, generator=self.generator)  # RandomSampler trailing [:0]
        self.order = torch.randperm(size, generator=self.generator)
        if not torch.equal(self.generator.get_state(), saved_state):
            raise ValueError("Cannot reconstruct checkpoint sampler/prefetch state exactly")

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        offset = self.offset
        order = self.order
        for _ in range(self.start, self.total):
            if offset == self.per_epoch:
                torch.randperm(self.size, generator=self.generator)
                order = torch.randperm(self.size, generator=self.generator)
                offset = 0
            yield order[offset * self.batch_size : (offset + 1) * self.batch_size].tolist()
            offset += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=25000)
    parser.add_argument("--preflight", action="store_true")
    cli = parser.parse_args()
    suite = cli.suite.resolve()
    output = suite / "runs" / cli.run_id
    cp = output / "checkpoints" / f"step_{cli.checkpoint_step:06d}"
    trainer = load_module(suite / "repo/scripts/lpwm_ab/train_b_sweep.py", "native_trainer")
    base = trainer.training_helpers()
    exp = json.loads((output / "experiment.json").read_text())
    args = SimpleNamespace(**exp)
    for k in ("data", "output", "eval_runner", "eval_python"):
        setattr(args, k, Path(getattr(args, k)))
    args.credential_file = Path("/root/lpwm_ab/.swanlab_api_key")
    args.run_name = exp["run_name"] + "-power-resume-20260919"
    assert args.output == output and args.steps == 30000 and not args.preflight and not args.disable_eval
    trainer.verify_checkpoint(cp, cli.checkpoint_step)
    state = torch.load(cp / "optimizer.pt", map_location="cpu", weights_only=False)
    assert state["step"] == cli.checkpoint_step
    assert state["schedule"] == {"warmup_steps": 500, "total_steps": 30000, "lr": 1e-4, "min_lr": 1e-5}
    assert trainer.file_sha256(args.data / "manifest.json") == exp["dataset_manifest_sha256"]
    base.seed_everything(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    split = base.build_split(manifest, 42)
    assert split["sha256"] == exp["split_sha256"]
    stats = json.loads((cp / "state_normalization.json").read_text())
    datasets = [
        base.LPWMCachedDataset(
            args.data, split[key], stats, history=2, action_horizon=16, world_horizon=1, include_world=True
        )
        for key in ("train_episode_ids", "validation_episode_ids")
    ]
    sampler = RestoredBatchSampler(
        len(datasets[0]), 8, state["step"] * 4, args.steps * 4, state["dataloader_rng"]
    )
    loader = DataLoader(
        datasets[0],
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(242),
    )
    val_loader = DataLoader(
        datasets[1], batch_size=8, sampler=base.balanced_validation_order(datasets[1], 42), num_workers=0
    )
    config = base.LPWMFMConfig.from_pretrained(cp)
    config.device = args.device
    expected = trainer.make_config(base, manifest, args)
    import dataclasses

    actual = dataclasses.asdict(config)
    expected_dict = dataclasses.asdict(expected)
    for key in expected_dict:
        if key not in ("pretrained_path", "pretrained_revision"):
            assert actual[key] == expected_dict[key], key
    policy = base.LPWMFMPolicy.from_pretrained(cp, config=config, strict=True).to(device).train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999))
    optimizer.load_state_dict(state["optimizer"])
    steps = {int(v["step"]) for v in optimizer.state.values() if "step" in v}
    assert steps == {state["step"]}, steps
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    metadata = {
        "resumed_from_step": state["step"],
        "original_last_reported": json.loads((output / "status.json").read_text()),
        "optimizer_restored": True,
        "sampler_state_verified_exact": True,
        "checkpoint_sha256": trainer.file_sha256(cp / "model.safetensors"),
        "schedule": state["schedule"],
        "preflight": cli.preflight,
        "time": time.time(),
    }
    print("RESTORE_VERIFIED", json.dumps(metadata), flush=True)
    iterator = iter(loader)
    if cli.preflight:
        optimizer.zero_grad(set_to_none=True)
        step = state["step"] + 1
        for group in optimizer.param_groups:
            group["lr"] = trainer.cosine_lr(step)
        for micro in range(4):
            batch = base.to_device(next(iterator), device)
            torch.manual_seed(42 + step * 4 + micro)
            loss, _ = policy(batch, current_step=step)
            assert torch.isfinite(loss)
            (loss / 4).backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        print(
            "RESUME_PREFLIGHT_OK",
            json.dumps(
                {
                    "next_step": step,
                    "loss": float(loss.detach()),
                    "grad_norm": float(norm),
                    "lr": trainer.cosine_lr(step),
                }
            ),
            flush=True,
        )
        return
    segment = output / "recovery_20260919"
    segment.mkdir(exist_ok=False)
    trainer.atomic_json(segment / "resume.json", metadata)
    import swanlab

    assert swanlab.login(api_key=args.credential_file.read_text().strip(), save=False, relogin=True)
    run = swanlab.init(
        project=args.project,
        name=args.run_name,
        mode="online",
        public=False,
        log_dir=str(segment / "swanlog"),
        config={**exp, "recovery": metadata},
    )
    assert run.mode == "online"
    trainer.atomic_json(
        segment / "swanlab_run.json", {"id": run.id, "url": run.url, "resumed_from_step": state["step"]}
    )
    # Preserve entire interrupted history; canonical log now represents the restored trajectory.
    (output / "metrics.jsonl").rename(segment / "interrupted_metrics.jsonl")
    with (output / "metrics.jsonl").open("x") as f:
        for line in (segment / "interrupted_metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["step"] <= state["step"]:
                f.write(line + "\n")
    (output / "swanlab_run.json").rename(segment / "original_swanlab_run.json")
    trainer.atomic_json(
        output / "swanlab_run.json", {"id": run.id, "url": run.url, "resumed_from_step": state["step"]}
    )
    best = min(
        json.loads(line).get("validation/fm_loss", float("inf"))
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["step"] % 5000 == 0
    )
    started = time.monotonic()
    current_step = state["step"]
    try:
        with (output / "metrics.jsonl").open("a", buffering=1) as metrics_file:

            def log(values, step):
                values = base.scalar_metrics(values)
                row = {"step": step, **values}
                metrics_file.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                swanlab.log(values, step=step)

            for step in range(state["step"] + 1, args.steps + 1):
                current_step = step
                t = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                acc = {}
                for group in optimizer.param_groups:
                    group["lr"] = trainer.cosine_lr(step)
                for micro in range(4):
                    batch = base.to_device(next(iterator), device)
                    torch.manual_seed(42 + step * 4 + micro)
                    loss, metrics = policy(batch, current_step=step)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    (loss / 4).backward()
                    values = base.scalar_metrics(metrics)
                    values["total_loss"] = float(loss.detach())
                    for k, v in values.items():
                        acc[k] = acc.get(k, 0) + v / 4
                    del loss, metrics
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                logged = {f"train/{k}": v for k, v in acc.items()}
                logged.update(
                    {
                        "train/lr": trainer.cosine_lr(step),
                        "train/grad_norm": float(norm),
                        "train/seen_windows": step * 32,
                        "system/step_seconds": time.monotonic() - t,
                    }
                )
                if step % 500 == 0:
                    logged.update(trainer.encoder_gradient_diagnostics(policy, batch, step))
                del batch
                if step % 10 == 0 or step == state["step"] + 1:
                    log(logged, step)
                    trainer.atomic_json(
                        output / "status.json",
                        {"status": "running", "step": step, "resumed_from_step": state["step"]},
                    )
                if step % 500 == 0:
                    with trainer.isolated_rng():
                        validation = base.validate(policy, val_loader, device, args, step)
                    log(validation, step)
                if step % 5000 == 0:
                    destination = trainer.save_checkpoint(
                        policy,
                        optimizer,
                        step,
                        output,
                        args,
                        split,
                        stats,
                        sampler.generator,
                        exp["initial_weights_sha256"],
                    )
                    trainer.checkpoint_alias(destination, "latest")
                    trainer.atomic_json(output / "status.json", {"status": "evaluating", "step": step})
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    log(trainer.run_checkpoint_eval(args, destination, step), step)
                    if validation["validation/fm_loss"] < best:
                        best = validation["validation/fm_loss"]
                        trainer.checkpoint_alias(destination, "best_loss")
                    if step == args.steps:
                        trainer.checkpoint_alias(destination, "final")
            trainer.atomic_json(
                output / "status.json",
                {
                    "status": "completed",
                    "step": args.steps,
                    "preflight": False,
                    "resumed_from_step": state["step"],
                    "resume_elapsed_seconds": time.monotonic() - started,
                    "best_saved_validation_fm_loss": best,
                },
            )
    except BaseException as error:
        trainer.atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "step": current_step,
                "error_type": type(error).__name__,
                "resumed_from_step": state["step"],
            },
        )
        raise
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
