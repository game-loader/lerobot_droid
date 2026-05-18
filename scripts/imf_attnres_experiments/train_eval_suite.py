#!/usr/bin/env python3
"""Train IMF-AttnRes on one LIBERO suite, eval checkpoints, and pick the best.

This wrapper intentionally keeps W&B API keys out of scripts/logs. Set WANDB_API_KEY
in the environment before launching if online W&B sync is desired.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SUITES = {
    "spatial": {"env_task": "libero_spatial", "episode_file": "spatial_episodes.txt", "episode_length": 280},
    "object": {"env_task": "libero_object", "episode_file": "object_episodes.txt", "episode_length": 280},
    "goal": {"env_task": "libero_goal", "episode_file": "goal_episodes.txt", "episode_length": 300},
    "long": {"env_task": "libero_10", "episode_file": "long_episodes.txt", "episode_length": 520},
}


def run(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as f:
        f.write(("\n\n===== COMMAND @ %s =====\n" % time.strftime("%F %T")).encode())
        f.write((" ".join(cmd) + "\n").encode())
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def parse_step(p: Path) -> int:
    m = re.match(r"^(\d+)$", p.name)
    if m:
        return int(m.group(1))
    return -1


def is_checkpoint_ready(p: Path) -> bool:
    return p.is_dir() and (p / "pretrained_model" / "model.safetensors").exists()


def ensure_subset_dataset(*, source_root: Path, subset_root: Path, episodes_path: Path) -> Path:
    """Materialize a local LeRobot subset with episode/frame indices remapped.

    LeRobot's EpisodeAwareSampler consumes dataset_from_index/dataset_to_index from
    metadata. When using dataset.episodes on the original aggregate dataset those
    metadata indices remain global, which can sample out-of-bounds relative rows.
    The single-task pilot avoided that by remapping a physical subset; do the
    same here for suite-level training.

    Suite subsets must remain sharded. Writing all selected image frames into one
    multi-GB parquet row group makes PyArrow/HF Datasets fail while reading nested
    Image structs (ArrowNotImplementedError: nested data conversions for chunked
    array outputs). Therefore we write one output shard per source data parquet
    that contributes selected episodes.
    """
    import json

    import datasets
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import embed_images

    def _existing_subset_looks_usable(root: Path) -> bool:
        data_paths = sorted((root / "data").glob("*/*.parquet"))
        required = [
            root / "meta/info.json",
            root / "meta/episodes/chunk-000/file-000.parquet",
            root / "meta/tasks.parquet",
            root / "meta/stats.json",
        ]
        if not all(p.exists() for p in required) or not data_paths:
            return False
        # Old broken materializations were one ~6GB parquet. Rebuild those.
        if max(p.stat().st_size for p in data_paths) > 1_500_000_000:
            return False
        return True

    lock_dir = subset_root.with_name(f"{subset_root.name}.lock")
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            if _existing_subset_looks_usable(subset_root):
                return subset_root
            time.sleep(10)

    try:
        if _existing_subset_looks_usable(subset_root):
            return subset_root

        tmp = subset_root.with_suffix(".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        (tmp / "data/chunk-000").mkdir(parents=True, exist_ok=True)

        episodes = [int(x) for x in episodes_path.read_text().split()]
        ep_set = set(episodes)

        info = json.loads((source_root / "meta/info.json").read_text())
        src_eps = pd.read_parquet(source_root / "meta/episodes/chunk-000/file-000.parquet")
        src_tasks = pd.read_parquet(source_root / "meta/tasks.parquet")
        selected_eps = src_eps[src_eps["episode_index"].isin(episodes)].copy()
        selected_eps = selected_eps.sort_values("episode_index").reset_index(drop=True)
        if selected_eps.empty:
            raise RuntimeError(f"No episodes matched {episodes_path}")

        old_to_new_ep = {old: new for new, old in enumerate(selected_eps["episode_index"].astype(int).tolist())}
        task_texts: list[str] = []
        for x in selected_eps["tasks"]:
            task_texts.append(str(x[0]) if hasattr(x, "__len__") and len(x) > 0 else str(x))
        unique_task_texts = list(dict.fromkeys(task_texts))
        task_old_to_new = {int(src_tasks.loc[text, "task_index"]): i for i, text in enumerate(unique_task_texts)}

        features_for_hf = {k: dict(v, shape=tuple(v["shape"])) for k, v in info["features"].items()}
        hf_features = get_hf_features_from_features(features_for_hf)

        cursor = 0
        out_file_idx = 0
        ep_records: dict[int, dict[str, int]] = {}
        data_columns = list(hf_features.keys())

        # The aggregate LIBERO metadata in this local copy has stale data/file_index
        # values, so use the actual source parquet files as the source of truth.
        for src_path in sorted((source_root / "data").glob("*/*.parquet")):
            df = pd.read_parquet(src_path)
            sub = df[df["episode_index"].isin(ep_set)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
            old_eps_in_file = sub["episode_index"].astype(int).drop_duplicates().tolist()

            local_cursor = 0
            for old_ep in old_eps_in_file:
                count = int((sub["episode_index"].astype(int) == old_ep).sum())
                expected = int(selected_eps.loc[selected_eps["episode_index"] == old_ep, "length"].iloc[0])
                if count != expected:
                    raise RuntimeError(
                        f"Episode {old_ep} length mismatch in {src_path}: frames={count}, meta={expected}. "
                        "This subset writer assumes episodes are not split across source parquet files."
                    )
                ep_records[old_ep] = {
                    "data/chunk_index": 0,
                    "data/file_index": out_file_idx,
                    "dataset_from_index": cursor + local_cursor,
                    "dataset_to_index": cursor + local_cursor + count,
                }
                local_cursor += count

            sub["episode_index"] = sub["episode_index"].astype(int).map(old_to_new_ep).astype("int64")
            sub["task_index"] = sub["task_index"].astype(int).map(task_old_to_new).astype("int64")
            sub["index"] = np.arange(cursor, cursor + len(sub), dtype=np.int64)
            cursor += len(sub)
            sub = sub[data_columns]

            ds = datasets.Dataset.from_dict(sub.to_dict(orient="list"), features=hf_features, split="train")
            ds = embed_images(ds)
            table = ds.with_format("arrow")[:]
            dst_path = tmp / f"data/chunk-000/file-{out_file_idx:03d}.parquet"
            with pq.ParquetWriter(dst_path, schema=table.schema, compression="snappy", use_dictionary=True) as writer:
                for start in range(0, table.num_rows, 1000):
                    writer.write_table(table.slice(start, 1000))
            out_file_idx += 1

        if cursor == 0:
            raise RuntimeError(f"No frames matched selected episodes from {episodes_path}")
        missing = sorted(ep_set.difference(ep_records))
        if missing:
            raise RuntimeError(f"Missing frame records for episodes: {missing[:10]}")

        new_eps = selected_eps[[
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "tasks",
            "length",
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
        ]].copy()
        for i in range(len(new_eps)):
            old_ep = int(new_eps.loc[i, "episode_index"])
            rec = ep_records[old_ep]
            new_eps.loc[i, "episode_index"] = old_to_new_ep[old_ep]
            new_eps.loc[i, "data/chunk_index"] = rec["data/chunk_index"]
            new_eps.loc[i, "data/file_index"] = rec["data/file_index"]
            new_eps.loc[i, "dataset_from_index"] = rec["dataset_from_index"]
            new_eps.loc[i, "dataset_to_index"] = rec["dataset_to_index"]
            new_eps.loc[i, "meta/episodes/chunk_index"] = 0
            new_eps.loc[i, "meta/episodes/file_index"] = 0

        new_tasks = pd.DataFrame(
            {"task_index": list(range(len(unique_task_texts)))},
            index=pd.Index(unique_task_texts, name="task"),
        )

        info["total_episodes"] = int(len(new_eps))
        info["total_frames"] = int(cursor)
        info["total_tasks"] = int(len(new_tasks))
        info["splits"] = {"train": f"0:{len(new_eps)}"}
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

        new_eps.to_parquet(tmp / "meta/episodes/chunk-000/file-000.parquet", index=False)
        new_tasks.to_parquet(tmp / "meta/tasks.parquet")
        (tmp / "meta/info.json").write_text(json.dumps(info, indent=2))
        # Reuse aggregate stats, matching the previous pilot subset behavior.
        shutil.copy2(source_root / "meta/stats.json", tmp / "meta/stats.json")

        shutil.rmtree(subset_root, ignore_errors=True)
        tmp.rename(subset_root)
        return subset_root
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def read_done_steps(path: Path) -> set[int]:
    if not path.exists():
        return set()
    steps: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "eval_done" and payload.get("ok") and payload.get("step") is not None:
            steps.add(int(payload["step"]))
    return steps


def eval_checkpoint(
    *,
    repo: Path,
    suite: str,
    ckpt: Path,
    step: int,
    eval_root: Path,
    env: dict[str, str],
    n_episodes: int,
    batch_size: int,
    use_async: bool,
) -> dict[str, Any]:
    spec = SUITES[suite]
    out_dir = eval_root / f"step_{step:06d}"
    log_path = out_dir / "eval.log"
    cmd = [
        "uv", "run", "--extra", "libero", "--extra", "evaluation", "lerobot-eval",
        f"--policy.path={ckpt / 'pretrained_model'}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        f"--env.task={spec['env_task']}",
        "--env.control_mode=relative",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.camera_name=agentview_image,robot0_eye_in_hand_image",
        "--env.init_states=true",
        f"--env.episode_length={spec['episode_length']}",
        "--env.max_parallel_tasks=1",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.use_async_envs={'true' if use_async else 'false'}",
        f"--output_dir={out_dir}",
        "--seed=1000",
    ]
    rc = run(cmd, log_path, repo, env)
    info_path = out_dir / "eval_info.json"
    if rc != 0 or not info_path.exists():
        return {"step": step, "checkpoint": str(ckpt), "ok": False, "returncode": rc, "eval_dir": str(out_dir)}
    info = json.loads(info_path.read_text())
    overall = info.get("overall", {})
    return {
        "step": step,
        "checkpoint": str(ckpt),
        "ok": True,
        "returncode": rc,
        "eval_dir": str(out_dir),
        "pc_success": overall.get("pc_success"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "avg_max_reward": overall.get("avg_max_reward"),
        "n_episodes": overall.get("n_episodes"),
        "eval_s": overall.get("eval_s"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--source-root", type=Path, default=Path("/data/lerobot_datasets/HuggingFaceVLA/libero"))
    ap.add_argument("--exp-root", type=Path, default=Path("/data/lerobot-imf-attnres-exp"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--final-eval-episodes", type=int, default=50)
    ap.add_argument("--eval-batch-size", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--policy-n-obs-steps", type=int, default=2)
    ap.add_argument("--policy-horizon", type=int, default=16)
    ap.add_argument("--policy-n-action-steps", type=int, default=8)
    ap.add_argument("--policy-n-layer", type=int, default=4)
    ap.add_argument("--policy-n-emb", type=int, default=256)
    ap.add_argument("--policy-n-head", type=int, default=1)
    ap.add_argument("--policy-n-kv-head", type=int, default=1)
    ap.add_argument("--policy-optimizer-lr", type=float, default=1e-4)
    ap.add_argument("--policy-optimizer-weight-decay", type=float, default=1e-6)
    ap.add_argument("--policy-optimizer-grad-clip-norm", type=float, default=10.0)
    ap.add_argument("--policy-scheduler-type", default="none", choices=["none", "cosine_decay_with_warmup"])
    ap.add_argument("--policy-scheduler-warmup-steps", type=int, default=1000)
    ap.add_argument("--policy-scheduler-decay-steps", type=int, default=100000)
    ap.add_argument("--policy-scheduler-decay-lr", type=float, default=2.5e-6)
    ap.add_argument("--run-suffix", default="", help="Optional suffix appended to the run name, e.g. d384-l16-b32.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--wandb-project", default="lerobot-imf-attnres-libero-main")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="Run training and exit before checkpoint eval.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training and evaluate existing checkpoints.")
    ap.add_argument("--watch-eval", action="store_true", help="Evaluate checkpoints as they appear during training.")
    ap.add_argument("--watch-timeout-s", type=int, default=604800)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    spec = SUITES[args.suite]
    run_name = f"imf-attnres-libero-{args.suite}-s{args.steps}-eval{args.eval_every}"
    if args.run_suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_suffix).strip("-")
        run_name = f"{run_name}-{safe_suffix}"
    out_dir = args.exp_root / "outputs" / "train" / run_name
    run_dir = args.exp_root / "runs" / run_name
    if args.eval_only:
        args.skip_train = True
    if args.overwrite and not args.skip_train and not args.eval_only:
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cache_root = args.exp_root / "cache"
    (cache_root / "uv").mkdir(parents=True, exist_ok=True)
    (cache_root / "hf-home").mkdir(parents=True, exist_ok=True)
    (cache_root / "hf-datasets" / args.suite).mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    (cache_root / "wandb").mkdir(parents=True, exist_ok=True)
    (cache_root / "wandb-config").mkdir(parents=True, exist_ok=True)
    (args.exp_root / "tmp").mkdir(parents=True, exist_ok=True)
    (args.exp_root / "wandb").mkdir(parents=True, exist_ok=True)
    env.update({
        "PATH": f"{Path.home() / '.local' / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "UV_CACHE_DIR": str(cache_root / "uv"),
        "HF_HOME": str(cache_root / "hf-home"),
        "HF_DATASETS_CACHE": str(cache_root / "hf-datasets" / args.suite),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "TMPDIR": str(args.exp_root / "tmp"),
        "WANDB_DIR": str(args.exp_root / "wandb"),
        "WANDB_CACHE_DIR": str(cache_root / "wandb"),
        "WANDB_CONFIG_DIR": str(cache_root / "wandb-config"),
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "LIBERO_CONFIG_PATH": str(repo / ".libero_config"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
        "WANDB_SILENT": "True",
    })
    # Keep API keys out of argv/logs. W&B can authenticate from WANDB_API_KEY
    # inherited in the environment or from an existing local wandb login/netrc.

    episodes_path = args.exp_root / spec["episode_file"]
    if not episodes_path.exists():
        print(f"ERROR: missing episodes file {episodes_path}", file=sys.stderr)
        return 2
    episodes = [int(x) for x in episodes_path.read_text().split()]
    subset_root = ensure_subset_dataset(
        source_root=args.source_root,
        subset_root=args.exp_root / "datasets" / f"libero_{args.suite}",
        episodes_path=episodes_path,
    )

    train_cmd = [
        "uv", "run", "--extra", "training", "--extra", "libero", "--extra", "evaluation", "lerobot-train",
        f"--dataset.repo_id=local/libero_{args.suite}",
        f"--dataset.root={subset_root}",
        "--dataset.use_imagenet_stats=true",
        "--policy.type=imf-attnres",
        "--policy.device=cuda",
        f"--policy.n_obs_steps={args.policy_n_obs_steps}",
        f"--policy.horizon={args.policy_horizon}",
        f"--policy.n_action_steps={args.policy_n_action_steps}",
        f"--policy.n_layer={args.policy_n_layer}",
        f"--policy.n_emb={args.policy_n_emb}",
        f"--policy.n_head={args.policy_n_head}",
        f"--policy.n_kv_head={args.policy_n_kv_head}",
        "--policy.backbone_type=attnres_full",
        "--policy.vision_backbone=resnet18",
        "--policy.pretrained_backbone_weights=null",
        "--policy.resize_shape=[128,128]",
        "--policy.spatial_softmax_num_keypoints=32",
        "--policy.use_separate_rgb_encoder_per_camera=true",
        "--policy.num_inference_steps=1",
        f"--policy.optimizer_lr={args.policy_optimizer_lr}",
        f"--policy.optimizer_weight_decay={args.policy_optimizer_weight_decay}",
        f"--policy.optimizer_grad_clip_norm={args.policy_optimizer_grad_clip_norm}",
        f"--policy.scheduler_type={args.policy_scheduler_type}",
        f"--policy.scheduler_warmup_steps={args.policy_scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={args.policy_scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={args.policy_scheduler_decay_lr}",
        "--policy.push_to_hub=false",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        "--prefetch_factor=2",
        "--persistent_workers=true",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        "--log_freq=50",
        "--eval_freq=0",
        f"--output_dir={out_dir}",
        "--wandb.enable=true" if args.wandb_mode != "disabled" else "--wandb.enable=false",
        f"--wandb.project={args.wandb_project}",
        "--wandb.disable_artifact=true",
        f"--wandb.mode={args.wandb_mode}",
        f"--job_name={run_name}",
    ]

    (run_dir / "train_cmd.json").write_text(json.dumps(train_cmd, indent=2))
    manifest = {
        "suite": args.suite,
        "env_task": spec["env_task"],
        "run_name": run_name,
        "out_dir": str(out_dir),
        "run_dir": str(run_dir),
        "subset_root": str(args.exp_root / "datasets" / f"libero_{args.suite}"),
        "episodes_count": len(episodes),
        "steps": args.steps,
        "save_freq": args.save_freq,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "eval_batch_size": args.eval_batch_size,
        "policy_n_obs_steps": args.policy_n_obs_steps,
        "policy_horizon": args.policy_horizon,
        "policy_n_action_steps": args.policy_n_action_steps,
        "policy_n_layer": args.policy_n_layer,
        "policy_n_emb": args.policy_n_emb,
        "policy_n_head": args.policy_n_head,
        "policy_n_kv_head": args.policy_n_kv_head,
        "policy_optimizer_lr": args.policy_optimizer_lr,
        "policy_optimizer_weight_decay": args.policy_optimizer_weight_decay,
        "policy_optimizer_grad_clip_norm": args.policy_optimizer_grad_clip_norm,
        "policy_scheduler_type": args.policy_scheduler_type,
        "policy_scheduler_warmup_steps": args.policy_scheduler_warmup_steps,
        "policy_scheduler_decay_steps": args.policy_scheduler_decay_steps,
        "policy_scheduler_decay_lr": args.policy_scheduler_decay_lr,
        "run_suffix": args.run_suffix,
        "gpu": args.gpu,
        "watch_eval": args.watch_eval,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ckpt_root = out_dir / "checkpoints"
    eval_root = args.exp_root / "outputs" / "eval" / run_name
    eval_results_path = run_dir / "eval_results.json"
    eval_events_path = run_dir / "eval_events.jsonl"

    if args.watch_eval:
        deadline = time.time() + args.watch_timeout_s
        eval_results = json.loads(eval_results_path.read_text()) if eval_results_path.exists() else []
        done_steps = read_done_steps(eval_events_path)
        final_rc_path = run_dir / "train.returncode"
        while time.time() < deadline:
            checkpoints = []
            if ckpt_root.exists():
                checkpoints = sorted(
                    [p for p in ckpt_root.iterdir() if is_checkpoint_ready(p)],
                    key=parse_step,
                )
                checkpoints = [
                    p for p in checkpoints
                    if parse_step(p) > 0
                    and (parse_step(p) % args.eval_every == 0 or parse_step(p) == args.steps)
                    and parse_step(p) not in done_steps
                ]
            for ckpt in checkpoints:
                step = parse_step(ckpt)
                result = eval_checkpoint(
                    repo=repo, suite=args.suite, ckpt=ckpt, step=step, eval_root=eval_root,
                    env=env, n_episodes=args.eval_episodes, batch_size=args.eval_batch_size, use_async=True,
                )
                eval_results.append(result)
                eval_results_path.write_text(json.dumps(eval_results, indent=2))
                with eval_events_path.open("a") as f:
                    f.write(json.dumps({"event": "eval_done", **result}) + "\n")
                if result.get("ok"):
                    done_steps.add(step)
            if final_rc_path.exists() and (not ckpt_root.exists() or not checkpoints):
                break
            time.sleep(args.poll_s)
        if time.time() >= deadline:
            print(f"watch eval timed out after {args.watch_timeout_s}s", file=sys.stderr)
            return 6
        # Continue below to select the best completed eval and run final eval.
    elif not args.skip_train:
        if args.overwrite:
            # In train-only mode this is already removed before run_dir creation,
            # but keep the cleanup close to the actual train launch as well so a
            # failed smoke/fallback attempt cannot poison a retry with the same
            # run identity.
            shutil.rmtree(out_dir, ignore_errors=True)
        rc = run(train_cmd, run_dir / "train.log", repo, env)
        (run_dir / "train.returncode").write_text(str(rc))
        if rc != 0:
            print(f"training failed rc={rc}; see {run_dir/'train.log'}", file=sys.stderr)
            return rc
        if args.train_only:
            return 0

    if args.train_only:
        return 0

    checkpoints = sorted([p for p in ckpt_root.iterdir() if is_checkpoint_ready(p)], key=parse_step)
    # If save_freq is not an exact multiple of eval_every, evaluate the saved
    # checkpoints at the nearest available cadence instead of requiring exact
    # divisibility. The final checkpoint is always included.
    checkpoints = [
        p for p in checkpoints
        if parse_step(p) > 0 and (parse_step(p) % args.eval_every == 0 or parse_step(p) == args.steps)
    ]
    if not checkpoints:
        print(f"ERROR: no checkpoints found under {ckpt_root}", file=sys.stderr)
        return 3

    eval_results = json.loads(eval_results_path.read_text()) if eval_results_path.exists() else []
    done_steps = {int(r["step"]) for r in eval_results if r.get("ok") and r.get("step") is not None}
    for ckpt in checkpoints:
        step = parse_step(ckpt)
        if step in done_steps:
            continue
        result = eval_checkpoint(
            repo=repo,
            suite=args.suite,
            ckpt=ckpt,
            step=step,
            eval_root=eval_root,
            env=env,
            n_episodes=args.eval_episodes,
            batch_size=args.eval_batch_size,
            use_async=True,
        )
        eval_results.append(result)
        eval_results_path.write_text(json.dumps(eval_results, indent=2))

    ok_results = [r for r in eval_results if r.get("ok")]
    if not ok_results:
        print("ERROR: all checkpoint evals failed", file=sys.stderr)
        return 4
    best = max(ok_results, key=lambda r: (float(r.get("pc_success") or -1), int(r["step"])))
    (run_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=2))

    final_result = eval_checkpoint(
        repo=repo,
        suite=args.suite,
        ckpt=Path(best["checkpoint"]),
        step=int(best["step"]),
        eval_root=args.exp_root / "outputs" / "final_eval" / run_name,
        env=env,
        n_episodes=args.final_eval_episodes,
        batch_size=args.eval_batch_size,
        use_async=True,
    )
    final = {"best_by_rollout10": best, "final_eval": final_result}
    (run_dir / "final_result.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    return 0 if final_result.get("ok") else 5


if __name__ == "__main__":
    raise SystemExit(main())
