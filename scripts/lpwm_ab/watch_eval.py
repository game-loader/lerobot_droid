"""Freeze newly saved A/B checkpoints and evaluate them without restarting training."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REQUIRED = (
    "config.json",
    "model.safetensors",
    "experiment.json",
    "state_normalization.json",
    "language_metadata.json",
    "language_embeddings.npy",
    "language_masks.npy",
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def freeze_checkpoint(source: Path, destination_root: Path):
    """Copy only a stable, committed model export; never evaluate a mutable latest directory.

    The trainer writes model weights before experiment.json. Requiring the latter
    to be newer plus a two-pass identity/mtime check prevents old step metadata
    from labeling freshly overwritten weights. Optimizer files are not needed.
    """
    if not all((source / name).is_file() for name in REQUIRED):
        return None
    before = {name: signature(source / name) for name in REQUIRED}
    if before["experiment.json"][2] < before["model.safetensors"][2]:
        return None
    if time.time_ns() - max(value[2] for value in before.values()) < 2_000_000_000:
        return None
    metadata = json.loads((source / "experiment.json").read_text())
    step = int(metadata["step"])
    destination = destination_root / f"step_{step:06d}"
    if destination.is_dir():
        return destination
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = destination_root / f".copy-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in REQUIRED:
            shutil.copy2(source / name, temporary / name)
        after = {name: signature(source / name) for name in REQUIRED}
        if after != before or json.loads((source / "experiment.json").read_text()) != metadata:
            return None
        weight_hash = hashlib.sha256((temporary / "model.safetensors").read_bytes()).hexdigest()
        atomic_json(
            temporary / "snapshot.json",
            {
                "step": step,
                "variant": metadata["variant"],
                "source": str(source),
                "weight_sha256": weight_hash,
                "captured_unix": time.time(),
            },
        )
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this invocation's private incomplete snapshot.


def publish_best(run_dir: Path, snapshot: Path, result: dict, result_path: Path):
    """Maintain an independent simulator-selected model, leaving loss-selected best untouched."""
    output = run_dir / "checkpoints/best_success.json"
    old = json.loads(output.read_text()) if output.exists() else None
    if old and old["protocol_sha256"] != result["protocol_sha256"]:
        raise ValueError("Refusing to compare simulator success under different evaluation protocols")
    if old and float(result["success_rate"]) <= old["success_rate"]:
        return False
    payload = {
        "success_rate": result["success_rate"],
        "pc_success": result["pc_success"],
        "successes": result["successes"],
        "num_episodes": result["num_episodes"],
        "protocol_sha256": result["protocol_sha256"],
        "checkpoint": str(snapshot),
        "result": str(result_path),
        "step": json.loads((snapshot / "experiment.json").read_text())["step"],
    }
    target = run_dir / "checkpoints/best_success"
    if target.exists() and not target.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing checkpoint directory: {target}")
    temporary = target.with_name(".best_success-link.tmp")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(snapshot.resolve(), target_is_directory=True)
    temporary.replace(target)
    atomic_json(output, payload)
    return True


def log_swanlab(args, variant, step, result, watcher_state):
    """Use separate resumable evaluation runs, never concurrent writers to the training runs."""
    if args.swanlab_mode == "disabled":
        return
    import swanlab

    key = (
        args.credential_file.read_text().strip()
        if args.credential_file
        else os.environ.get("SWANLAB_API_KEY")
    )
    if not key:
        raise RuntimeError("SwanLab online evaluation requested without credential")
    swanlab.login(api_key=key, save=False)
    del key
    previous = watcher_state.setdefault("swanlab_runs", {}).get(variant)
    kwargs = {"id": previous["id"], "resume": "allow"} if previous else {}
    run = swanlab.init(
        project=args.project,
        name=f"LPWM-FM-{variant}-Spatial-simulation-eval-{args.stamp}",
        job_type="evaluation",
        group="A-vs-B-GT-action-only",
        mode="online",
        public=False,
        reinit=True,
        log_dir=str(args.output / "swanlog"),
        config={
            "variant": variant,
            "training_run": str(args.root / f"runs/{args.stamp}-{variant}"),
            "protocol": result["protocol"],
            "metric": "successful episodes / completed episodes",
        },
        **kwargs,
    )
    watcher_state["swanlab_runs"][variant] = {"id": run.id, "url": run.url}
    metrics = {
        f"eval/{key}": result[key]
        for key in ("successes", "num_episodes", "success_rate", "pc_success", "duration_seconds")
    }
    metrics["eval/checkpoint_step"] = step
    for task in result["per_task"]:
        metrics[f"eval/task_{task['task_id']}/success_rate"] = task["success_rate"]
        metrics[f"eval/task_{task['task_id']}/pc_success"] = task["pc_success"]
    swanlab.log(metrics, step=step)
    swanlab.finish()
    print(f"SWANLAB_EVAL_{variant}={run.url}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/lpwm_ab"))
    parser.add_argument("--stamp", default="20260917-183742")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--swanlab-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--project", default="lpwm-fm-libero-spatial-ab")
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.output / "watcher_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}, "swanlab_runs": {}}
    # A previous watcher may have died while its evaluator was running. Do not
    # duplicate it automatically: preserve the job state for explicit recovery.
    if any(job["status"] == "running" for job in state["jobs"].values()):
        raise RuntimeError("Recorded evaluation still running; verify/recover it before restarting watcher")
    active = None
    idle_after_training = 0
    while True:
        training_complete = True
        for variant in ("A", "B"):
            run_dir = args.root / f"runs/{args.stamp}-{variant}"
            status = json.loads((run_dir / "status.json").read_text())
            training_complete &= status["status"] in {"completed", "failed", "superseded"}
            try:
                snapshot = freeze_checkpoint(
                    run_dir / "checkpoints/latest", args.output / "snapshots" / variant
                )
            except (OSError, json.JSONDecodeError) as error:
                print(f"Snapshot retry {variant}: {type(error).__name__}: {error}", flush=True)
                snapshot = None
            if snapshot:
                metadata = json.loads((snapshot / "experiment.json").read_text())
                job_key = f"{variant}:{metadata['step']}"
                if job_key not in state["jobs"]:
                    state["jobs"][job_key] = {
                        "variant": variant,
                        "step": metadata["step"],
                        "snapshot": str(snapshot),
                        "status": "pending",
                        "queued_unix": time.time(),
                    }
                    print(f"SNAPSHOT_READY {job_key} {snapshot}", flush=True)
        atomic_json(state_path, state)
        if args.capture_only:
            return
        if active:
            process, job_key, log_file, started = active
            rc = process.poll()
            if rc is None and time.monotonic() - started > args.timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                rc = process.returncode
            if rc is not None:
                log_file.close()
                job = state["jobs"][job_key]
                result_path = Path(job["result_path"])
                job["exit_code"] = rc
                job["finished_unix"] = time.time()
                if rc == 0 and result_path.is_file():
                    result = json.loads(result_path.read_text())
                    job.update(
                        status="complete",
                        success_rate=result["success_rate"],
                        successes=result["successes"],
                        num_episodes=result["num_episodes"],
                    )
                    publish_best(
                        args.root / f"runs/{args.stamp}-{job['variant']}",
                        Path(job["snapshot"]),
                        result,
                        result_path,
                    )
                    try:
                        log_swanlab(args, job["variant"], job["step"], result, state)
                    except Exception as error:
                        job["swanlab_error"] = f"{type(error).__name__}: {error}"
                        print(
                            f"Evaluation completed but SwanLab upload failed: {job['swanlab_error']}",
                            flush=True,
                        )
                    print(f"EVAL_COMPLETE {job_key} {job['successes']}/{job['num_episodes']}", flush=True)
                else:
                    job["status"] = "failed"
                    print(f"EVAL_FAILED {job_key} exit={rc}; see {job['log_path']}", flush=True)
                active = None
                atomic_json(state_path, state)
        if active is None:
            pending = [(key, job) for key, job in state["jobs"].items() if job["status"] == "pending"]
            if pending:
                job_key, job = min(pending, key=lambda item: item[1]["queued_unix"])
                result_path = args.output / "results" / f"{job['variant']}-step_{job['step']:06d}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                log_path = result_path.with_suffix(".log")
                command = [
                    str(args.python),
                    "-u",
                    str(Path(__file__).with_name("evaluate.py")),
                    "--checkpoint",
                    job["snapshot"],
                    "--output",
                    str(result_path),
                    "--device",
                    args.device,
                    "--episodes-per-task",
                    str(args.episodes_per_task),
                    "--seed",
                    str(args.seed),
                    "--seed-namespace",
                    "validation",
                ]
                if args.video:
                    command.append("--video")
                log_file = log_path.open("w")
                process = subprocess.Popen(
                    command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
                )
                job.update(
                    status="running", pid=process.pid, result_path=str(result_path), log_path=str(log_path)
                )
                active = process, job_key, log_file, time.monotonic()
                print(f"EVAL_STARTED {job_key} pid={process.pid}", flush=True)
                atomic_json(state_path, state)
        if (
            training_complete
            and active is None
            and not any(job["status"] == "pending" for job in state["jobs"].values())
        ):
            idle_after_training += 1
            if idle_after_training >= 3:
                state["status"] = (
                    "completed_with_failures"
                    if any(job["status"] == "failed" for job in state["jobs"].values())
                    else "completed"
                )
                atomic_json(state_path, state)
                return
        else:
            idle_after_training = 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
