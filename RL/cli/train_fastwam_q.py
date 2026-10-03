"""Train a DINOv3 chunk-Q on LeRobot demos, optionally mixing recorded online rollouts 1:1."""

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from RL.fastwam_q import FastWAMQConfig, FastWAMQFunction, FastWAMQTrainer, model as q_model
from RL.fastwam_q.bundle import BundleActionNormalizer, bundle_q_options
from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint
from RL.fastwam_q.data import ActionNormalizer, LeRobotChunkDataset, sample_mixed
from RL.fastwam_q.integration import load_processors


class MixedBatches(Dataset):
    """Each fetched item is one already-collated demo/online microbatch."""

    def __init__(self, demos, online, size, batches):
        """Construct the component from its configuration and supplied dependencies."""
        self.demos, self.online, self.size, self.batches = demos, online, size, batches

    def __len__(self):
        """Return the number of available replay samples."""
        return self.batches

    def __getitem__(self, index):
        """Read one episode-bounded training sample."""
        return sample_mixed(self.demos, self.online, self.size)


def main():
    """Run the command-line Q-only training loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fastwam-checkpoint", help="BC pretrained_model directory; only config/processors are loaded"
    )
    source.add_argument(
        "--fastwam-bundle", type=Path, help="Custom FR3 bundle with config.yaml and dataset_stats.json"
    )
    parser.add_argument(
        "--cache-root", type=Path, help="Already-built lossless FR3 demo RGB cache (no video decoding)"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--online-repo-id", action="append", default=[])
    parser.add_argument("--online-root", action="append", default=[])
    parser.add_argument("--online-labels", action="append", default=[])
    parser.add_argument("--config", type=Path, help="JSON overrides for FastWAMQConfig")
    source.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", type=int, default=12000, help="Additional optimizer updates this invocation"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device microbatch")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-project", help="Enable online/private SwanLab logging")
    parser.add_argument("--run-name", default="fastwam-dinov3-q")
    parser.add_argument("--credential-file", type=Path, help="Existing SwanLab key file; never logged")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    if args.resume:
        q, normalizer = load_checkpoint(args.resume)
        config = q.config
    elif args.fastwam_bundle:
        options = json.loads(args.config.read_text()) if args.config else {}
        config = FastWAMQConfig(**{**bundle_q_options(args.fastwam_bundle), **options})
        normalizer = BundleActionNormalizer.from_bundle(args.fastwam_bundle)
        q = FastWAMQFunction(config)
    else:
        bc = FastWAMConfig.from_pretrained(args.fastwam_checkpoint)
        options = json.loads(args.config.read_text()) if args.config else {}
        config = FastWAMQConfig(
            **{
                "action_dim": bc.action_dim,
                "chunk_size": bc.action_horizon,
                "camera_keys": tuple(bc.image_features),
                **options,
            }
        )
        pre, _ = load_processors(bc, args.fastwam_checkpoint)
        normalizer = ActionNormalizer.from_processor(pre)
        q = FastWAMQFunction(config)
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.accumulation
    trainer = FastWAMQTrainer(q, normalizer, args.device)
    if args.resume:
        trainer.restore(args.resume)
    demos = (
        CachedDemoChunkDataset(args.cache_root, config=config, stride=args.stride)
        if args.cache_root
        else LeRobotChunkDataset(
            args.repo_id, args.root, config=config, demonstrations=True, stride=args.stride
        )
    )
    online = []
    for i, repo_id in enumerate(args.online_repo_id):
        root = args.online_root[i] if i < len(args.online_root) else None
        labels = args.online_labels[i] if i < len(args.online_labels) else None
        online.append(LeRobotChunkDataset(repo_id, root, config=config, labels=labels, stride=args.stride))
    data = MixedBatches(
        demos, ConcatDataset(online) if online else None, config.batch_size, args.steps * args.accumulation
    )
    # Worker RNG states are not checkpointed; start a deterministic new replay stream on resume.
    data_seed = args.seed + trainer.step
    loader = DataLoader(
        data,
        batch_size=None,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        pin_memory=args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(data_seed),
    )
    iterator = iter(loader)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        **asdict(config),
        "steps": args.steps,
        "initial_step": trainer.step,
        "target_steps": trainer.step + args.steps,
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "save_freq": args.save_freq,
        "repo_id": args.repo_id,
        "dataset_root": args.root,
        "cache_root": str(args.cache_root) if args.cache_root else None,
        "demo_chunks": len(demos),
        "seed": args.seed,
        "data_seed": data_seed,
        "sampling": "uniform overlapping chunk starts with replacement",
        "reward_convention": "existing demo convention: +1 at final transition when rewards absent",
        "trainable_parameters": sum(p.numel() for p in q.parameters() if p.requires_grad),
        "decoder_training_attention_backend": "math",
        "q_model_source_sha256": hashlib.sha256(Path(q_model.__file__).read_bytes()).hexdigest(),
    }
    if args.fastwam_bundle:
        settings["bc_stats_sha256"] = hashlib.sha256(
            (args.fastwam_bundle / "dataset_stats.json").read_bytes()
        ).hexdigest()
    (args.output / "training_config.json").write_text(json.dumps(settings, indent=2))
    run = None
    if args.swanlab_project:
        import swanlab

        if args.credential_file:
            swanlab.login(api_key=args.credential_file.read_text().strip(), save=False)
        run = swanlab.init(
            project=args.swanlab_project,
            name=args.run_name,
            mode="online",
            public=False,
            config=settings,
            log_dir=str(args.output / "swanlab"),
        )
        (args.output / "swanlab_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url, "mode": run.mode}, indent=2)
        )
        logging.info("SwanLab: %s", run.url)
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("a") as metrics_file:
        for update in range(args.steps):
            tick = time.monotonic()
            batches = [next(iterator) for _ in range(args.accumulation)]
            loaded = time.monotonic()
            metrics = trainer.update(batches)
            if not math.isfinite(metrics["loss"]) or not math.isfinite(metrics["grad_norm"]):
                raise FloatingPointError(f"Nonfinite Q training at step {trainer.step}")
            metrics.update(
                data_seconds=loaded - tick,
                update_seconds=time.monotonic() - loaded,
                elapsed_seconds=time.monotonic() - started,
                learning_rate=trainer.optimizer.param_groups[0]["lr"],
                dino_learning_rate=trainer.optimizer.param_groups[1]["lr"],
            )
            if args.device.startswith("cuda"):
                metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated(trainer.device) / 2**30
            if (update + 1) % args.log_freq == 0 or update == 0:
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                logging.info("Q update: %s", metrics)
                progress = {"status": "running", "target_steps": settings["target_steps"], **metrics}
                temporary = args.output / "progress.json.tmp"
                temporary.write_text(json.dumps(progress, indent=2))
                temporary.replace(args.output / "progress.json")
                if run is not None:
                    swanlab.log(metrics, step=trainer.step)
            if (update + 1) % args.save_freq == 0:
                trainer.save(args.output / f"step_{trainer.step:06d}")
                (args.output / "latest_checkpoint.txt").write_text(f"step_{trainer.step:06d}\n")
                if run is not None:
                    swanlab.log({"checkpoint/step": trainer.step}, step=trainer.step)
    trainer.save(args.output / "last")
    (args.output / "progress.json").write_text(
        json.dumps({"status": "complete", "target_steps": settings["target_steps"], **metrics}, indent=2)
    )
    if run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
