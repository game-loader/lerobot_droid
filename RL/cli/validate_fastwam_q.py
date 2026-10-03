"""Run one paired stability treatment from the same saved Q, EMA, Adam, batches and seed."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint
from RL.fastwam_q.model import chunk_targets
from RL.fastwam_q.stability import CASES, GradientGuard, monitored_update
from RL.fastwam_q.trainer import FastWAMQTrainer


def write_json(path, value):
    """Publish complete progress snapshots, preserving nonfinite diagnostic values as strings."""

    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PairedBatches(Dataset):
    """Every treatment sees exactly the same precomputed frame-index batches."""

    def __init__(self, source, indices):
        """Retain only a dataset and a small index matrix, not a new image cache."""
        self.source, self.indices = source, indices

    def __len__(self):
        """Return the fixed experimental update budget."""
        return len(self.indices)

    def __getitem__(self, index):
        """Fetch this update's agreed batch using existing RGB records."""
        return default_collate([self.source[int(i)] for i in self.indices[index]])


@torch.inference_mode()
def monitor(trainer, batches, frozen_targets=None):
    """A fixed training-pool panel, not a held-out success/generalization evaluation."""
    q = trainer.q.eval()
    collected, predictions, moving_targets = [], [], []
    for raw in batches:
        batch = trainer.prepare_batch(raw)
        text = q.text_features(batch["tasks"], trainer.device)
        # All arms use FP32 for this probe and share original BF16-cached T5 features.
        prediction = q.q_values(q.online(batch["images"], batch["action_chunk"], text, batch["valid"]))
        next_q = q.q_values(
            q.target(batch["next_images"], batch["next_action_chunk"], text, batch["next_valid"])
        )
        targets = chunk_targets(batch["rewards"], batch["valid"], batch["done"], next_q, q.config.gamma)
        predictions.append(prediction.cpu())
        moving_targets.append(targets.cpu())
        collected.append(targets.cpu())
    predicted, moving = torch.cat(predictions), torch.cat(moving_targets)
    fixed = torch.cat(collected) if frozen_targets is None else frozen_targets
    error = predicted - fixed
    metrics = {
        "monitor/fixed_target_mae": float(error.abs().mean()),
        "monitor/fixed_target_mse": float(error.square().mean()),
        "monitor/current_target_mae": float((predicted - moving).abs().mean()),
        "monitor/target_drift_mae": float((moving - fixed).abs().mean()),
        "monitor/q_mean": float(predicted.mean()),
    }
    return metrics, fixed


def run(args):
    """Observe a treatment, with no promotion or automatic restart of production training."""
    import swanlab

    out = args.output / args.case
    out.mkdir(parents=True, exist_ok=False)
    case = CASES[args.case]
    torch.set_num_threads(8)
    q, normalizer = load_checkpoint(args.checkpoint, device="cpu")
    q.config.batch_size = 64
    q.config.amp_dtype = case.precision
    trainer = FastWAMQTrainer(q, normalizer, "cuda")
    trainer.restore(args.checkpoint)
    initial_step = trainer.step
    if not case.update_dino:
        # Keep the original train-mode RoPE augmentation, so the only intervention
        # is stopping backbone parameter updates, not changing feature augmentation.
        q.online.image_encoder.backbone.requires_grad_(False)
    dataset = CachedDemoChunkDataset(args.cache, config=q.config)
    indices = np.load(args.plan / "batch_indices.npy")
    panel_indices = np.load(args.plan / "monitor_indices.npy")
    settings = {
        **asdict(case),
        "checkpoint": str(args.checkpoint),
        "initial_step": initial_step,
        "additional_updates": len(indices),
        "batch_size": int(indices.shape[1]),
        "seed": args.seed,
        "gradient_clip": q.config.grad_clip_norm,
        "head_lr_original": 3e-4,
        "dino_lr_original": 9e-5,
        "optimizer_state": "restored, including Adam moments and counters",
        "ema_state": "restored",
        "video_cache_rebuilt": False,
        "batch_indices_sha256": hashlib.sha256((args.plan / "batch_indices.npy").read_bytes()).hexdigest(),
        "t5_features": "same frozen BF16 cache in all treatments",
        "monitor": "fixed training-pool examples; not held-out validation",
        "freeze_control": "backbone parameter updates off; train-mode augmentation preserved",
        "replicates": 1,
        "precision_rng_note": "same seeds, but kernels may consume RNG differently across precisions",
        "gradient_guard": asdict(GradientGuard()),
    }
    write_json(out / "config.json", settings)
    swanlab.login(api_key=args.credential_file.read_text().strip(), save=False)
    run = swanlab.init(
        project="fastwam-q-stability",
        name=f"s5k-{case.name}-seed{args.seed}",
        mode="online",
        public=False,
        config=settings,
        log_dir=str(out / "swanlab"),
    )
    write_json(out / "swanlab_run.json", {"id": run.id, "url": run.url, "mode": run.mode})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q.text_features(dataset.tasks, trainer.device)
    panel = [default_collate([dataset[int(i)] for i in row]) for row in panel_indices]
    initial_monitor, fixed = monitor(trainer, panel)
    torch.save(fixed, out / "fixed_monitor_targets.pt")
    swanlab.log({"update": 0, **initial_monitor}, step=initial_step)
    write_json(out / "initial_monitor.json", initial_monitor)
    loader = DataLoader(
        PairedBatches(dataset, indices),
        batch_size=None,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)
    guard = GradientGuard()
    history = []
    status, reason = "completed_window", None
    started = time.monotonic()
    try:
        with (out / "metrics.jsonl").open("w") as log:
            for update in range(1, len(indices) + 1):
                tick = time.monotonic()
                batch = next(iterator)
                loaded = time.monotonic()
                metrics, reason = monitored_update(trainer, batch, case, update, guard, args.seed)
                metrics.update(
                    data_seconds=loaded - tick,
                    update_seconds=time.monotonic() - loaded,
                    elapsed_seconds=time.monotonic() - started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                history.append(metrics)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                if update % 10 == 0 or update == 1 or reason:
                    swanlab.log(metrics, step=initial_step + update)
                    write_json(
                        out / "progress.json",
                        {
                            "status": "running" if not reason else "guard_stopped",
                            "case": case.name,
                            "reason": reason,
                            **metrics,
                        },
                    )
                    print(json.dumps({"case": case.name, **metrics}), flush=True)
                if reason:
                    status = "guard_stopped"
                    break
                if update % 250 == 0:
                    results, _ = monitor(trainer, panel, fixed)
                    swanlab.log({"update": update, **results}, step=trainer.step)
                    with (out / "monitor.jsonl").open("a") as f:
                        f.write(json.dumps({"update": update, **results}) + "\n")
        final_monitor, _ = monitor(trainer, panel, fixed)
        tail = history[-500:]
        finite_norms = [r["grad_norm"] for r in history if np.isfinite(r["grad_norm"])]
        summary = {
            "status": status,
            "case": case.name,
            "reason": reason,
            "initial_step": initial_step,
            "final_step": trainer.step,
            "attempted_updates": len(history),
            "tail500_loss_median": float(np.median([r["loss"] for r in tail])),
            "tail500_td_mae_median": float(np.median([r["td_mae"] for r in tail])),
            "tail500_grad_median": float(np.median([r["grad_norm"] for r in tail])),
            "tail500_grad_p95": float(np.quantile([r["grad_norm"] for r in tail], 0.95)),
            "tail500_clip_fraction": float(np.mean([r["clipped"] for r in tail])),
            "max_preclip_grad": max(finite_norms, default=0.0),
            "initial_monitor": initial_monitor,
            "final_monitor": final_monitor,
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "single-seed stability screen, not proof of 45k stability or policy success",
        }
        summary["passes_gradient_screen"] = (
            status == "completed_window"
            and summary["tail500_grad_median"] < 10
            and summary["tail500_grad_p95"] < 100
            and summary["tail500_clip_fraction"] <= 0.2
        )
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.save(out / ("end_checkpoint" if reason is None else "guard_checkpoint"))
        write_json(out / "summary.json", summary)
        write_json(out / "progress.json", summary)
        swanlab.log(
            {
                "screen/completed_window": int(status == "completed_window"),
                "screen/passes_gradient_screen": int(summary["passes_gradient_screen"]),
            },
            step=trainer.step + 1,
        )
        swanlab.finish()
    except BaseException as exc:
        write_json(out / "FAILED.json", {"type": type(exc).__name__, "error": str(exc), "step": trainer.step})
        swanlab.finish(state="crashed")
        raise


def main():
    """Parse one treatment configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260930)
    run(p.parse_args())


if __name__ == "__main__":
    main()
