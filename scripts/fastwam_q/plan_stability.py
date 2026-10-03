"""Freeze paired batch/panel indices and the finite-horizon stability experiment plan."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from RL.fastwam_q.stability import CASES, GradientGuard


def main():
    """Plan a single-seed diagnostic screen without decoding or caching any video."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--updates", type=int, default=4000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = np.load(args.cache / "task_index.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(tasks), size=(args.updates, 64), dtype=np.int64)
    monitor = np.stack(
        [rng.choice(np.flatnonzero(tasks == task), 64, replace=False) for task in np.unique(tasks)]
    )
    np.save(args.output / "batch_indices.npy", indices)
    np.save(args.output / "monitor_indices.npy", monitor)
    remaining = [name for name in CASES if name not in ("baseline", "both_lr_div10_warmup")]
    random.Random(args.seed).shuffle(remaining)
    queues = {0: ["baseline"], 1: ["both_lr_div10_warmup"]}
    for i, name in enumerate(remaining):
        queues[i % 2].append(name)
    plan = {
        "question": "Which intervention stabilizes the same 5k critic while preserving meaningful TD learning?",
        "initial_checkpoint_step": 5000,
        "additional_updates": args.updates,
        "final_step_if_completed": 5000 + args.updates,
        "batch_size": 64,
        "gradient_accumulation": 1,
        "cases": {name: asdict(case) for name, case in CASES.items()},
        "seed": args.seed,
        "independent_seed_replicates": 1,
        "fixed": [
            "checkpoint weights",
            "Adam moments/counters",
            "EMA weights",
            "BC statistics",
            "batch indices",
            "per-update RNG seeds",
            "chunk32",
            "gamma0.99",
            "clip10",
            "EMA tau0.005",
        ],
        "contrast": "2x2 head/DINO LR factorial, one added warmup contrast, FP32 control, no-DINO-update control",
        "warmup": "300-update linear restart warmup; then constant LR; not a fresh-initialization warmup test",
        "no_dino_updates": "retain train-mode augmentation, freeze only backbone parameters",
        "monitor": "256 task-balanced training-pool examples; fixed 5k bootstrap targets and current TD targets; not held-out evaluation",
        "monitor_every": 250,
        "queues": queues,
        "order": "baseline and warmup candidate prioritized; remaining arms seed-randomized",
        "guards": asdict(GradientGuard()),
        "screen_pass": "complete all updates, tail500 grad median<10, p95<100, clip fraction<=0.2; separately inspect fixed-target error to avoid mistaking stagnation for learning",
        "interpretation_limit": "No significance claims from correlated optimizer steps; one seed cannot prove 45k stability or robot success",
        "production_training": "remains stopped; no automatic promotion",
        "video_cache_rebuilt": False,
        "indices_sha256": hashlib.sha256((args.output / "batch_indices.npy").read_bytes()).hexdigest(),
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {"queues": queues, "updates_per_case": args.updates, "indices_sha256": plan["indices_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
