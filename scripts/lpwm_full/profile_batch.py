"""Isolated real-cache GPU training probe; no persistent training/checkpoint or telemetry."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.lpwm_ab import train_b_sweep as trainer
from scripts.lpwm_full.data import FullCachedDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--updates", type=int, default=8)
    args = p.parse_args()
    if args.updates < 3:
        p.error("Use at least three updates to separate warmup from timing")
    if args.batch_size not in (8, 16, 32):
        raise ValueError("Preserve effective batch32")
    base = trainer.training_helpers()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base.seed_everything(42)
    device = torch.device("cuda")
    m = json.loads((args.data / "manifest.json").read_text())
    split = base.build_split(m, 42)
    stats = base.training_state_statistics(
        np.load(args.data / "states.npy", mmap_mode="r"), m["episodes"], split["train_episode_ids"]
    )
    ds = FullCachedDataset(args.data, split["train_episode_ids"], stats)
    dl = base.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    cfgargs = SimpleNamespace(
        device="cuda", world_weight=1.0, rec_weight=1.0, dyn_weight=1.0, prior_weight=0.001
    )
    policy = base.LPWMFMPolicy(trainer.make_config(base, m, cfgargs)).to(device).train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999))
    result = {
        "batch_size": args.batch_size,
        "grad_accumulation": 32 // args.batch_size,
        "effective_batch": 32,
        "formal_training": False,
        "data_manifest_sha256": trainer.file_sha256(args.data / "manifest.json"),
        "data_path": str(args.data.resolve()),
        "precision": "float32 with TF32; same as winning B recipe",
        "initial_weights_sha256": base.weight_hash(policy),
        "device": torch.cuda.get_device_name(),
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "world_scale_step": 1000,
    }
    iterator = iter(dl)
    times = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.updates):
            torch.cuda.synchronize()
            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for _micro in range(result["grad_accumulation"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dl)
                    batch = next(iterator)
                batch = base.to_device(batch, device)
                loss, _ = policy(batch, current_step=1000)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss / result["grad_accumulation"]).backward()
                del loss
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            print(
                json.dumps(
                    {
                        "update": step + 1,
                        "seconds": times[-1],
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                ),
                flush=True,
            )
        result.update(
            train_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            train_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            update_seconds=times,
            steady_update_seconds=float(np.median(times[2:])),
        )
        result["windows_per_second"] = 32 / result["steady_update_seconds"]
        trainer.encoder_gradient_diagnostics(policy, batch, 1000)
        result.update(
            status="passed",
            with_diagnostics_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            with_diagnostics_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
    except torch.cuda.OutOfMemoryError:
        result.update(
            status="oom",
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
