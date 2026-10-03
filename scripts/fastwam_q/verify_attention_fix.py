"""Short optimizer/EMA continuation from a guarded checkpoint, without production promotion."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from RL.cli.validate_fastwam_q import PairedBatches, monitor, write_json
from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint
from RL.fastwam_q.stability import CASES, GradientGuard, monitored_update
from RL.fastwam_q.trainer import FastWAMQTrainer


def main():
    """Keep learning rates, dropout, checkpointing, Adam and EMA unchanged."""
    import swanlab

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--start-update", type=int, default=2030)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    q, normalizer = load_checkpoint(args.checkpoint, device="cpu")
    trainer = FastWAMQTrainer(q, normalizer, "cuda")
    trainer.restore(args.checkpoint)
    initial_step = trainer.step
    dataset = CachedDemoChunkDataset(args.cache, config=q.config)
    indices = np.load(args.plan / "batch_indices.npy")
    paired = indices[args.start_update - 1 : args.start_update - 1 + args.steps]
    panel_indices = np.load(args.plan / "monitor_indices.npy")
    settings = {
        "intervention": "decoder training math SDPA; DINO and inference attention unchanged",
        "checkpoint": str(args.checkpoint),
        "initial_step": initial_step,
        "steps": args.steps,
        "batch_size": int(indices.shape[1]),
        "head_lr": q.config.learning_rate,
        "dino_lr": q.config.dino_learning_rate,
        "dropout": q.config.dropout,
        "gradient_checkpointing": q.config.gradient_checkpointing,
        "seed": args.seed,
        "production_restarted": False,
        "limitation": "Single-seed continuation verification; not a 45k stability or policy-quality claim.",
    }
    write_json(args.output / "config.json", settings)
    swanlab.login(api_key=args.credential_file.read_text().strip(), save=False)
    run = swanlab.init(
        project="fastwam-q-stability",
        name=f"decoder-math-s{initial_step}-{args.steps}updates",
        public=False,
        config=settings,
        log_dir=str(args.output / "swanlab"),
    )
    write_json(args.output / "swanlab_run.json", {"id": run.id, "url": run.url})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q.text_features(dataset.tasks, "cuda")
    panel = [default_collate([dataset[int(index)] for index in row]) for row in panel_indices]
    initial_monitor, fixed_targets = monitor(trainer, panel)
    write_json(args.output / "initial_monitor.json", initial_monitor)
    swanlab.log(initial_monitor, step=initial_step)
    loader = DataLoader(
        PairedBatches(dataset, paired),
        batch_size=None,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    guard = GradientGuard()
    history = []
    reason = None
    started = time.monotonic()
    try:
        with (args.output / "metrics.jsonl").open("w") as log:
            for update, batch in enumerate(loader, start=args.start_update):
                metrics, reason = monitored_update(
                    trainer, batch, CASES["baseline"], update, guard, args.seed
                )
                history.append(metrics)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                if update % 10 == 0 or reason:
                    swanlab.log(metrics, step=trainer.step)
                    write_json(args.output / "progress.json", metrics)
                    print(json.dumps(metrics), flush=True)
                if reason:
                    break
                if (update - args.start_update + 1) % 250 == 0:
                    values, _ = monitor(trainer, panel, fixed_targets)
                    swanlab.log(values, step=trainer.step)
                    with (args.output / "monitor.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": trainer.step, **values}) + "\n")
        final_monitor, _ = monitor(trainer, panel, fixed_targets)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.save(args.output / ("guard_checkpoint" if reason else "end_checkpoint"))
        summary = {
            **settings,
            "status": "guard_stopped" if reason else "completed_short_verification",
            "reason": reason,
            "final_step": trainer.step,
            "applied_updates": trainer.step - initial_step,
            "grad_median": float(np.median([entry["grad_norm"] for entry in history])),
            "grad_max": max(entry["grad_norm"] for entry in history),
            "clip_fraction": float(np.mean([entry["clipped"] for entry in history])),
            "first25_loss_median": float(np.median([entry["loss"] for entry in history[:25]])),
            "last25_loss_median": float(np.median([entry["loss"] for entry in history[-25:]])),
            "seconds": time.monotonic() - started,
            "initial_monitor": initial_monitor,
            "final_monitor": final_monitor,
        }
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
