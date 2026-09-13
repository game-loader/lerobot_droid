#!/usr/bin/env python3
"""
Quick comparison script: PointNet vs PTv3 encoder training.

This script helps you quickly start training with PTv3 encoder or compare
against the baseline PointNet model.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train DP3 with PointNet or PTv3 encoder")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["pointnet", "ptv3", "sonata"],
        required=True,
        help="Point cloud encoder to use",
    )
    parser.add_argument(
        "--ptv3-weights",
        type=str,
        default="",
        help="Path to PTv3 pretrained weights (optional, uses placeholder if empty)",
    )
    parser.add_argument(
        "--sonata-weights",
        type=str,
        default="",
        help="Path to Sonata pretrained weights (optional, uses placeholder if empty)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Training steps (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output directory name (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )

    args = parser.parse_args()
    if args.encoder != "pointnet":
        parser.error(
            "PTv3/Sonata are unimplemented legacy MLP placeholders, not pretrained encoders. "
            "Use --encoder pointnet; see docs/EXPERIMENTAL_POINT_ENCODERS.md."
        )

    # Setup paths
    root = Path(__file__).parent.parent
    data_dir = Path("/home/droid/franka_duo_tmr_lerobot_v6_rl_action20_pc_only")
    repo_id = "franka_duo/franka_duo_tmr_lerobot_v6_rl_action20_pc_only"

    # Determine output directory
    if args.output_name:
        output_name = args.output_name
    else:
        encoder_suffix = {"pointnet": "pointnet", "ptv3": "ptv3", "sonata": "sonata"}[args.encoder]
        output_name = f"franka_duo_dp3_action20_pc_only_dit768_{encoder_suffix}_il_{args.steps // 1000}k"

    output_dir = root / "outputs" / output_name / "train"
    log_dir = root / "outputs" / output_name / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "uv",
        "run",
        "--with",
        "swanlab==0.9.4",
        "python",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={data_dir}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=true",
        "--policy.type=dp3",
        "--policy.device=cuda",
        f"--policy.repo_id={repo_id}",
        "--policy.push_to_hub=false",
        "--policy.use_amp=true",
        "--policy.horizon=32",
        "--policy.n_action_steps=16",
        "--policy.wrist_image_keys=[]",
        "--policy.point_cloud_num_points=2048",
        "--policy.point_cloud_random_subsample=false",
        "--policy.crop_is_random=false",
        "--policy.use_pc_color=false",
        "--policy.diffusion_backbone=transformer",
        "--policy.transformer_hidden_dim=768",
        "--policy.transformer_num_layers=9",
        "--policy.transformer_num_heads=8",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--num_workers=16",
        "--save_freq=5000",
        "--eval_freq=0",
        "--log_freq=100",
        "--wandb.enable=true",
        "--wandb.project=franka-duo-dp3-il",
        "--wandb.disable_artifact=true",
        "--wandb.mode=online",
        f"--output_dir={output_dir}",
    ]

    # Encoder-specific settings
    if args.encoder == "sonata":
        cmd.extend(
            [
                "--policy.use_sonata_encoder=true",
                f"--policy.sonata_model_path={args.sonata_weights}",
                "--policy.sonata_feature_dim=512",
                "--policy.sonata_freeze_backbone=true",
                "--policy.point_cloud_encoder_output_dim=256",
                "--policy.point_cloud_use_projection=true",
            ]
        )
        print("🚀 Training with Sonata encoder (frozen backbone)")
        print("   Sonata 5-stage: 512-d → Projection: 256-d")
        print("   Model: 108.5M params (CVPR 2025 Highlight)")
        if args.sonata_weights:
            print(f"   Loading weights from: {args.sonata_weights}")
        else:
            print("   ⚠️  No Sonata weights specified - using placeholder model")
    elif args.encoder == "ptv3":
        cmd.extend(
            [
                "--policy.use_ptv3_encoder=true",
                f"--policy.ptv3_model_path={args.ptv3_weights}",
                "--policy.ptv3_feature_dim=512",
                "--policy.ptv3_freeze_backbone=true",
                "--policy.point_cloud_encoder_output_dim=256",
                "--policy.point_cloud_use_projection=true",
            ]
        )
        print("🚀 Training with PTv3 encoder (frozen backbone)")
        print("   PTv3 bottleneck: 512-d → Projection: 256-d")
        if args.ptv3_weights:
            print(f"   Loading weights from: {args.ptv3_weights}")
        else:
            print("   ⚠️  No PTv3 weights specified - using placeholder model")
    else:
        cmd.extend(
            [
                "--policy.use_ptv3_encoder=false",
                "--policy.point_cloud_encoder_output_dim=64",
                "--policy.point_cloud_use_projection=false",
            ]
        )
        print("🚀 Training with PointNet encoder (simple MLP)")

    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Training steps: {args.steps}")
    print(f"🔢 Batch size: {args.batch_size}")

    if args.dry_run:
        print("\n🔍 Dry run - command that would be executed:")
        print(" ".join(cmd))
        return 0

    # Setup environment
    env = os.environ.copy()
    env["RL100_SWANLAB_MODE"] = "online"
    env["RL100_SWANLAB_LOG_DIR"] = str(output_dir / "swanlog")
    env["UV_CACHE_DIR"] = str(root / ".uv-cache")
    env["HF_HOME"] = str(root / ".hf-datasets-cache" / "hf")
    env["HF_DATASETS_CACHE"] = str(root / ".hf-datasets-cache" / "datasets")
    env["HF_HUB_CACHE"] = str(root / ".hf-datasets-cache" / "hub")

    # Run training
    print("\n▶️  Starting training...\n")
    log_file = log_dir / "train.log"

    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(line, end="")
                f.write(line)

            process.wait()

            if process.returncode != 0:
                print(f"\n❌ Training failed with exit code {process.returncode}")
                print(f"   Check log file: {log_file}")
                return process.returncode
            else:
                print("\n✅ Training completed successfully!")
                print(f"   Checkpoints saved to: {output_dir / 'checkpoints'}")
                return 0

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
