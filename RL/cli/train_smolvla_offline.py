"""Single-task SmolVLA offline RL: frozen tokens -> query IQL + flow PPO + token AM-Q."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Completed IL pretrained_model directory")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", default="local/franka_duo_lerobot_rgb20d_v1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path)
    p.add_argument(
        "--confirmed-manifest-rewards",
        action="store_true",
        help="Explicitly attest manifest_reward is task success, NOT just saved status",
    )
    p.add_argument("--task", default="pick cup and bowl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--offline", action="store_true", help="Only use cached HF dependencies")
    p.add_argument("--resume", type=Path, help="Offline RL checkpoint, distinct from --checkpoint IL root")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--iql-steps", type=int, default=10000)
    p.add_argument("--dynamics-steps", type=int, default=10000)
    p.add_argument("--actor-steps", type=int, default=10000)
    p.add_argument(
        "--stop-after",
        choices=("iql", "dynamics", "actor"),
        default="actor",
        help="Save and stop after this stage; resume later to continue",
    )
    p.add_argument("--actor-lr", type=float, default=1e-7)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128, help="Q/V Transformer width (not flattened MLP width)")
    p.add_argument("--critic-layers", type=int, default=2)
    p.add_argument("--critic-heads", type=int, default=4)
    p.add_argument("--value-tokens", type=int, default=2, help="Learned query tokens per Q/V head")
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--dynamics-hidden", type=int, default=64)
    p.add_argument("--dynamics-layers", type=int, default=2)
    p.add_argument("--dynamics-heads", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--flow-noise", type=float, default=0.1)
    p.add_argument("--clip-ratio", type=float, default=0.1)
    p.add_argument("--amq-interval", type=int, default=50)
    p.add_argument("--amq-horizon", type=int, default=5)
    p.add_argument("--amq-margin", type=float, default=0.05)
    p.add_argument("--max-validation-loss", type=float, default=1.0)
    p.add_argument("--max-disagreement", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2 IQL + 2 dynamics + 1 actor update, for plumbing validation only",
    )
    return p


def run(args):
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from torch.utils.data import DataLoader, Subset

    from RL.smolvla.adapter import load_policy, make_adapters
    from RL.smolvla.checkpoint import file_hash, restore_checkpoint, save_checkpoint, validate_checkpoint
    from RL.smolvla.data import SmolVLADecisionDataset
    from RL.smolvla.flow import FlowConfig
    from RL.smolvla.trainer import SmolVLAOfflineTrainer

    if args.smoke:
        args.iql_steps = args.dynamics_steps = 2
        args.actor_steps, args.batch_size = 1, 2
        args.amq_interval, args.amq_horizon = 1, 2
        args.num_workers = 0
    for name in (
        "batch_size",
        "torch_threads",
        "iql_steps",
        "dynamics_steps",
        "actor_steps",
        "save_every",
        "log_every",
        "amq_interval",
        "amq_horizon",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory, including when resuming")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    # Keep exact replay deterministic; TF32 does not change between old/new paths.
    torch.backends.cuda.matmul.allow_tf32 = True
    old_metadata = validate_checkpoint(args.resume) if args.resume else None
    checkpoint = args.resume / "pretrained_model" if args.resume else args.checkpoint
    policy, pre, post, active = load_policy(checkpoint, args.device)
    encoder, current, behavior = make_adapters(policy, pre, FlowConfig(args.flow_steps, args.flow_noise))
    dataset = SmolVLADecisionDataset(
        args.dataset_root,
        args.repo_id,
        chunk_size=current.execution_steps,
        gamma=args.gamma,
        labels_path=args.labels,
        confirmed_manifest_rewards=args.confirmed_manifest_rewards,
        task=args.task,
    )
    train_ids, val_ids, held_out = dataset.split(args.seed)

    def batches(ids, shuffle):
        loader = DataLoader(
            Subset(dataset, ids),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        while True:
            yield from loader

    train_iter, val_iter = batches(train_ids, True), batches(val_ids, False)
    first = next(train_iter)
    example, _ = encoder.encode(first["observation"], args.task)
    trainer = SmolVLAOfflineTrainer(
        encoder,
        current,
        behavior,
        active,
        example,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden=args.hidden,
        critic_layers=args.critic_layers,
        critic_heads=args.critic_heads,
        value_tokens=args.value_tokens,
        ensemble=args.ensemble,
        dynamics_hidden=args.dynamics_hidden,
        dynamics_layers=args.dynamics_layers,
        dynamics_heads=args.dynamics_heads,
        clip_ratio=args.clip_ratio,
        relative_margin=args.amq_margin,
        max_validation_loss=args.max_validation_loss,
        max_disagreement=args.max_disagreement,
    )
    dataset_digest = hashlib.sha256()
    for file in sorted([args.dataset_root / "meta/info.json", *args.dataset_root.glob("data/**/*.parquet")]):
        dataset_digest.update(file_hash(file).encode())
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "il_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "dataset_sha256": dataset_digest.hexdigest(),
        "held_out_episodes": held_out,
        "labels": {str(k): v for k, v in dataset.labels.items()},
        "kv_shape": list(example.kv.shape[1:]),
        "token_shape": list(example.tokens.shape[1:]),
        "dynamic_slots": example.dynamic_mask[0].tolist(),
        "parameter_counts": trainer.parameter_counts(),
        "task": args.task,
        "deployment_policy": "AM-Q accepted behavior, not ungated candidate",
        "frozen_modules": ["SmolVLM", "state_proj"],
        "representation": "frozen contextual tokens + native K/V; dynamics predicts input prefix",
        "architecture": "value_query_transformer_v2",
        "dynamics_error": "per-token RMS normalized prefix MSE, balanced visual/state + reward MSE + done BCE",
    }
    counters = {"iql": 0, "dynamics": 0, "actor": 0}
    if old_metadata:
        for key in (
            "il_sha256",
            "dataset_sha256",
            "labels",
            "held_out_episodes",
            "kv_shape",
            "task",
            "token_shape",
            "dynamic_slots",
            "architecture",
            "dynamics_error",
        ):
            if metadata[key] != old_metadata[key]:
                raise ValueError(f"Resume contract changed: {key}")
        for key in (
            "flow_steps",
            "flow_noise",
            "gamma",
            "hidden",
            "critic_layers",
            "critic_heads",
            "value_tokens",
            "ensemble",
            "dynamics_hidden",
            "dynamics_layers",
            "dynamics_heads",
            "clip_ratio",
            "actor_lr",
            "critic_lr",
            "amq_margin",
            "max_validation_loss",
            "max_disagreement",
        ):
            if vars(args)[key] != old_metadata["arguments"][key]:
                raise ValueError(f"Resume configuration changed: {key}")
        counters = restore_checkpoint(args.resume, trainer)
        if any(counters[stage] > getattr(args, f"{stage}_steps") for stage in counters):
            raise ValueError("Resume step targets cannot be below already completed steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "ready",
                "train_decisions": len(train_ids),
                "validation_decisions": len(val_ids),
                "kv_shape": metadata["kv_shape"],
                "token_shape": metadata["token_shape"],
                "parameter_counts": metadata["parameter_counts"],
                "frozen_shared_vlm": True,
                "counters": counters,
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    with (args.output_dir / "metrics.jsonl").open("a", buffering=1) as log:
        for stage, limit in (
            ("iql", args.iql_steps),
            ("dynamics", args.dynamics_steps),
            ("actor", args.actor_steps),
        ):
            if stage == "actor" and counters["actor"] == 0:
                validation = trainer.encode_batch(next(val_iter), args.task)
                validation_loss = trainer.dynamics.validation_loss(*validation.dynamics_args())
                if not math.isfinite(validation_loss):
                    raise RuntimeError("Nonfinite dynamics validation error; actor was NOT started")
                print(
                    json.dumps(
                        {
                            "event": "warmups_complete",
                            "counters": dict(counters),
                            "dynamics_validation_loss": validation_loss,
                        }
                    ),
                    flush=True,
                )
                if not args.smoke and validation_loss > args.max_validation_loss:
                    raise RuntimeError(
                        f"Dynamics validation loss {validation_loss:.6f} exceeds {args.max_validation_loss}; "
                        "actor was NOT started. Resume the dynamics checkpoint and extend dynamics training."
                    )
            while counters[stage] < limit:
                batch = trainer.encode_batch(next(train_iter), args.task)
                if stage == "iql":
                    metrics = trainer.iql.update(batch.decision())
                elif stage == "dynamics":
                    metrics = trainer.dynamics.update(*batch.dynamics_args())
                else:
                    metrics = trainer.update_actor(batch, seed=args.seed + counters[stage])
                counters[stage] += 1
                if stage == "actor" and (
                    counters[stage] % args.amq_interval == 0 or counters[stage] == limit
                ):
                    validation = trainer.encode_batch(next(val_iter), args.task)
                    metrics.update(
                        trainer.evaluate_and_promote(validation, horizon=args.amq_horizon, seed=args.seed)
                    )
                record = {
                    "stage": stage,
                    "step": counters[stage],
                    "elapsed_s": time.monotonic() - started,
                    **metrics,
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if counters[stage] % args.log_every == 0 or counters[stage] == limit:
                    print(json.dumps(record, allow_nan=False), flush=True)
                if counters[stage] % args.save_every == 0 or counters[stage] == limit:
                    save_checkpoint(
                        args.output_dir / "checkpoints" / f"{stage}_{counters[stage]:06d}",
                        trainer,
                        pre,
                        post,
                        metadata,
                        dict(counters),
                    )
            if stage == args.stop_after:
                break
    print(
        json.dumps({"event": "complete", "counters": counters, "promotions": trainer.promotions}), flush=True
    )
    return trainer


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
