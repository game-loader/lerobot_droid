#!/usr/bin/env python3
"""Train standard Diffusion Policy on LIBERO spatial and eval every N steps.

Target: reproduce SmolVLA paper's 78.3% success rate on libero_spatial.

Protocol:
  - Train one model on all 10 tasks of libero_spatial (432 episodes)
  - Every 10k steps, eval the checkpoint (10 episodes/task for speed)
  - After training, final eval with 50 episodes/task on best checkpoint

Usage:
  cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres
  .venv/bin/python scripts/imf_attnres_experiments/train_diffusion_spatial.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SUITE = "spatial"
ENV_TASK = "libero_spatial"
EPISODE_LENGTH = 280


def run(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as f:
        f.write(f"\n\n===== COMMAND @ {time.strftime('%F %T')} =====\n".encode())
        f.write((" ".join(cmd) + "\n").encode())
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def eval_checkpoint(
    *, repo: Path, ckpt: Path, step: int, eval_root: Path,
    env: dict[str, str], n_episodes: int, batch_size: int,
) -> dict:
    out_dir = eval_root / f"step_{step:06d}"
    log_path = out_dir / "eval.log"
    cmd = [
        "uv", "run", "--extra", "libero", "--extra", "evaluation", "lerobot-eval",
        f"--policy.path={ckpt}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        f"--env.task={ENV_TASK}",
        "--env.control_mode=relative",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.camera_name=agentview_image,robot0_eye_in_hand_image",
        "--env.init_states=true",
        f"--env.episode_length={EPISODE_LENGTH}",
        "--env.max_parallel_tasks=1",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.use_async_envs=true",
        f"--output_dir={out_dir}",
        "--seed=1000",
    ]
    rc = run(cmd, log_path, repo, env)
    info_path = out_dir / "eval_info.json"
    if rc != 0 or not info_path.exists():
        return {"step": step, "checkpoint": str(ckpt), "ok": False, "returncode": rc}
    info = json.loads(info_path.read_text())
    overall = info.get("overall", {})
    return {
        "step": step,
        "checkpoint": str(ckpt),
        "ok": True,
        "pc_success": overall.get("pc_success"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "n_episodes": overall.get("n_episodes"),
        "eval_s": overall.get("eval_s"),
    }


def main() -> int:
    exp_root = Path("/data/lerobot-imf-attnres-exp")
    repo = Path("/data/lerobot-imf-attnres-exp/lerobot-imf-attnres")
    dataset_root = exp_root / "datasets" / "libero_spatial"

    # Training hyperparameters
    steps = 100000
    batch_size = 64
    save_freq = 10000
    eval_every = 10000
    eval_episodes = 10
    final_eval_episodes = 50
    eval_batch_size = 10
    lr = "1e-4"
    gpu = os.environ.get("GPU", "0")
    wandb_mode = os.environ.get("WANDB_MODE", "online")

    run_name = f"diffusion-libero-{SUITE}-s{steps}-b{batch_size}"
    out_dir = exp_root / "outputs" / "train" / run_name
    run_dir = exp_root / "runs" / run_name
    eval_root = exp_root / "outputs" / "eval" / run_name

    # Clean previous run if exists
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = os.environ.copy()
    cache_root = exp_root / "cache"
    for d in [cache_root / "uv", cache_root / "hf-home", cache_root / "hf-datasets",
              cache_root / "xdg", cache_root / "wandb", cache_root / "wandb-config",
              exp_root / "tmp", exp_root / "wandb"]:
        d.mkdir(parents=True, exist_ok=True)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "UV_CACHE_DIR": str(cache_root / "uv"),
        "HF_HOME": str(cache_root / "hf-home"),
        "HF_DATASETS_CACHE": str(cache_root / "hf-datasets"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "TMPDIR": str(exp_root / "tmp"),
        "WANDB_DIR": str(exp_root / "wandb"),
        "WANDB_CACHE_DIR": str(cache_root / "wandb"),
        "WANDB_CONFIG_DIR": str(cache_root / "wandb-config"),
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "LIBERO_CONFIG_PATH": str(repo / ".libero_config"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
        "WANDB_SILENT": "True",
    })

    # Verify dataset
    if not (dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: Dataset not found at {dataset_root}", file=sys.stderr)
        print("Run the imf-attnres train_eval_suite.py first to materialize the subset.", file=sys.stderr)
        return 1

    # Build training command
    train_cmd = [
        "uv", "run", "--extra", "training", "--extra", "libero", "--extra", "evaluation", "lerobot-train",
        f"--dataset.repo_id=local/libero_spatial",
        f"--dataset.root={dataset_root}",
        "--dataset.use_imagenet_stats=true",
        "--policy.type=diffusion",
        "--policy.device=cuda",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.vision_backbone=resnet18",
        "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
        "--policy.resize_shape=[128,128]",
        "--policy.crop_ratio=0.9",
        "--policy.crop_is_random=true",
        "--policy.use_group_norm=false",
        "--policy.spatial_softmax_num_keypoints=32",
        "--policy.use_separate_rgb_encoder_per_camera=true",
        "--policy.down_dims=[256,512,1024]",
        "--policy.kernel_size=5",
        "--policy.n_groups=8",
        "--policy.diffusion_step_embed_dim=128",
        "--policy.use_film_scale_modulation=true",
        "--policy.noise_scheduler_type=DDPM",
        "--policy.num_train_timesteps=100",
        "--policy.num_inference_steps=10",
        "--policy.beta_schedule=squaredcos_cap_v2",
        "--policy.prediction_type=epsilon",
        "--policy.clip_sample=true",
        "--policy.clip_sample_range=1.0",
        f"--policy.optimizer_lr={lr}",
        "--policy.optimizer_betas=[0.95,0.999]",
        "--policy.optimizer_weight_decay=1e-6",
        "--policy.scheduler_name=cosine",
        "--policy.scheduler_warmup_steps=500",
        "--policy.push_to_hub=false",
        "--env.type=libero",
        f"--env.task={ENV_TASK}",
        "--env.control_mode=relative",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.camera_name=agentview_image,robot0_eye_in_hand_image",
        "--env.init_states=true",
        f"--env.episode_length={EPISODE_LENGTH}",
        "--env.max_parallel_tasks=1",
        f"--eval.batch_size={eval_batch_size}",
        f"--eval.n_episodes={eval_episodes}",
        "--eval.use_async_envs=true",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=24",
        "--prefetch_factor=2",
        "--persistent_workers=true",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        "--log_freq=100",
        f"--eval_freq={eval_every}",
        f"--output_dir={out_dir}",
        f"--wandb.enable={'true' if wandb_mode != 'disabled' else 'false'}",
        "--wandb.project=lerobot-diffusion-libero-baseline",
        "--wandb.disable_artifact=true",
        f"--wandb.mode={wandb_mode}",
        f"--job_name={run_name}",
    ]

    (run_dir / "train_cmd.json").write_text(json.dumps(train_cmd, indent=2))
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"Steps: {steps}, Batch: {batch_size}, LR: {lr}, GPU: {gpu}")
    print(f"Eval every {eval_every} steps ({eval_episodes} ep/task), final: {final_eval_episodes} ep/task")
    print()

    # Launch training
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    rc = run(train_cmd, run_dir / "train.log", repo, env)
    (run_dir / "train.returncode").write_text(str(rc))
    if rc != 0:
        print(f"Training failed rc={rc}; see {run_dir / 'train.log'}", file=sys.stderr)
        return rc

    # Evaluate checkpoints
    print()
    print("=" * 60)
    print("EVALUATING CHECKPOINTS")
    print("=" * 60)
    ckpt_root = out_dir / "checkpoints"
    checkpoints = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and (p / "pretrained_model" / "model.safetensors").exists()],
        key=lambda p: int(p.name) if p.name.isdigit() else -1,
    )
    checkpoints = [p for p in checkpoints if p.name.isdigit() and int(p.name) % eval_every == 0]

    eval_results = []
    for ckpt in checkpoints:
        step = int(ckpt.name)
        print(f"\n--- Eval step {step} ({eval_episodes} episodes/task) ---")
        result = eval_checkpoint(
            repo=repo, ckpt=ckpt / "pretrained_model", step=step,
            eval_root=eval_root, env=env, n_episodes=eval_episodes, batch_size=eval_batch_size,
        )
        eval_results.append(result)
        if result["ok"]:
            print(f"  SUCCESS RATE: {result['pc_success']:.1f}%  (eval_s={result.get('eval_s', 0):.0f}s)")
        else:
            print(f"  EVAL FAILED (rc={result.get('returncode')})")
        (run_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))

    # Select best checkpoint
    ok_results = [r for r in eval_results if r.get("ok")]
    if not ok_results:
        print("ERROR: all checkpoint evals failed", file=sys.stderr)
        return 4
    best = max(ok_results, key=lambda r: (float(r.get("pc_success") or -1), int(r["step"])))
    (run_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=2))
    print(f"\nBest checkpoint: step {best['step']} with {best['pc_success']:.1f}% success")

    # Final eval with more episodes
    print()
    print("=" * 60)
    print(f"FINAL EVAL (step {best['step']}, {final_eval_episodes} episodes/task)")
    print("=" * 60)
    final_result = eval_checkpoint(
        repo=repo, ckpt=Path(best["checkpoint"]), step=int(best["step"]),
        eval_root=exp_root / "outputs" / "final_eval" / run_name, env=env,
        n_episodes=final_eval_episodes, batch_size=eval_batch_size,
    )
    final = {"best_by_rollout10": best, "final_eval": final_result}
    (run_dir / "final_result.json").write_text(json.dumps(final, indent=2))

    print()
    print("=" * 60)
    if final_result["ok"]:
        print(f"FINAL SUCCESS RATE: {final_result['pc_success']:.1f}%")
        print(f"(Target: 78.3% from SmolVLA paper)")
    else:
        print("FINAL EVAL FAILED")
    print("=" * 60)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
