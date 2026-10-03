"""Fail-closed single-GPU pipeline for common40 or all130 cache and fresh B training and native task rollouts."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--train-python", type=Path, default=Path("/root/lpwm_ab/.venv/bin/python"))
    p.add_argument("--eval-python", type=Path, default=Path("/root/lpwm_ab/.venv-eval/bin/python"))
    p.add_argument("--libero-config", type=Path, default=Path("/root/lpwm_ab/libero_config"))
    p.add_argument("--credential-file", type=Path, default=Path("/root/lpwm_ab/.swanlab_api_key"))
    p.add_argument(
        "--expected-init-sha256",
        default="3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e",
        help="Required initialization SHA256 for both preflight and fresh production training",
    )
    p.add_argument("--run-name", help="Override the task-count/budget-derived SwanLab run name")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, choices=(1, 8), default=1)
    p.add_argument("--steps", type=int, choices=(30000, 80000, 200000), default=30000)
    p.add_argument(
        "--cache-storage-root",
        type=Path,
        action="append",
        default=[],
        help="Explicit trusted overflow roots; reject all other symlink escapes",
    )
    p.add_argument("--task-count", type=int, choices=(40, 130), default=130)
    p.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--grad-accumulation", type=int, choices=(1, 2, 4), default=4)
    p.add_argument(
        "--preflight-run",
        type=Path,
        default=Path("preflight_run_v3"),
        help="Preflight output directory, absolute or relative to --root",
    )
    p.add_argument("--preflight-step", type=int, default=2)
    p.add_argument(
        "--profile-gate",
        type=Path,
        help="Require a passed full-world-strength profile JSON matching batch/accumulation",
    )
    args = p.parse_args(argv)
    if args.batch_size * args.grad_accumulation != 32:
        p.error("batch-size * grad-accumulation must remain 32")
    if args.preflight_step < 1:
        p.error("preflight-step must be positive")
    if args.workers < 0:
        p.error("workers must be non-negative")
    args.expected_init_sha256 = args.expected_init_sha256.lower()
    if len(args.expected_init_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.expected_init_sha256
    ):
        p.error("expected-init-sha256 must be exactly 64 hexadecimal characters")
    return args


def verify_profile(root, args):
    if args.profile_gate is None:
        return
    profile = read(root / args.profile_gate)
    expected = {
        "status": "passed",
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "effective_batch": 32,
        "world_scale_step": 1000,
        "formal_training": False,
    }
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Profile gate must pass at full world strength for selected batch/accumulation")


def verify_preflight(root, args):
    from scripts.lpwm_full.evaluate import protocol_suites, suites_for_count, validate_result
    from scripts.lpwm_full.train import validate_parallel_eval

    preflight = root / args.preflight_run
    state = read(preflight / "status.json")
    if state.get("status") != "completed" or state.get("step") != args.preflight_step:
        raise ValueError("RealGPU all-suite preflight not complete at selected step")
    output = preflight / "eval" / f"step_{args.preflight_step:06d}.json"
    result = read(output)
    validate_result(result, preflight=True)
    validate_parallel_eval(result, getattr(args, "eval_workers", 1))
    if protocol_suites(result["protocol"]) != suites_for_count(args.task_count):
        raise ValueError("Preflight task scope differs from production")
    if result["checkpoint"]["step"] != args.preflight_step:
        raise ValueError("Wrong preflight checkpoint step")
    receipt = read(output.with_suffix(".swanlab.json"))
    if (
        not receipt.get("verified_upload")
        or receipt.get("formal_result_eligible") is not False
        or receipt.get("result_sha256") != digest(output)
    ):
        raise ValueError("Preflight upload not verified/incorrectly marked formal or changed result")
    experiment = read(preflight / "experiment.json")
    if getattr(args, "steps", 30000) != 30000 and experiment.get("cosine_endpoint_step") != args.steps:
        raise ValueError(f"Preflight cosine endpoint differs from requested {args.steps // 1000}k")
    if experiment["initial_weights_sha256"] != args.expected_init_sha256:
        raise ValueError("Architecture/init differs from selectedB")


def verify_cache(root, manifest, task_count=130, storage_roots=()):
    """Check manifest-referenced images/indices; common caches need no per-task complete.json."""
    from scripts.lpwm_full.train import validate_manifest

    suites = validate_manifest(manifest)
    if sum(suites.values()) != task_count:
        raise ValueError("Cache task scope differs from requested task count")
    if manifest["image_codec"] != "xor16_zlib_rgb128_v1":
        raise ValueError("Unexpected cache codec")
    root = root.resolve()
    allowed_roots = (root, *(Path(p).resolve(strict=True) for p in storage_roots))
    if any(str(p) == "/" for p in allowed_roots):
        raise ValueError("An entire filesystem is not an allowed cache root")
    checked = {}
    completion_records = {}

    def check_file(relative, expected):
        path = (root / relative).resolve()
        if not any(path.is_relative_to(p) for p in allowed_roots) or not expected:
            raise ValueError(f"Changed cache file or missing hash: {relative}")
        if path in checked:
            if checked[path] != expected:
                raise ValueError(f"Conflicting cache hashes for same path: {relative}")
            return
        if digest(path) != expected:
            raise ValueError(f"Changed cache file or missing hash: {relative}")
        checked[path] = expected

    scalar_hashes = manifest["scalar_sha256"]
    required_scalars = {
        f"{name}.npy" for name in ("states", "actions", "task_index", "language_embeddings", "language_masks")
    }
    if not scalar_hashes.keys() >= required_scalars:
        raise ValueError("Missing scalar/language cache hashes")
    for name, expected in scalar_hashes.items():
        check_file(name, expected)
    cursor = 0
    for shard in manifest["image_shards"]:
        if shard["start"] != cursor or shard["end"] <= cursor:
            raise ValueError("Image shard coverage is not contiguous")
        folder = Path(shard["path"]).parent
        complete_path = (root / folder / "complete.json").resolve()
        if not any(complete_path.is_relative_to(p) for p in allowed_roots):
            raise ValueError("Cache completion record outside cache root")
        if complete_path not in completion_records:
            complete_hashes = read(complete_path)["files_sha256"] if complete_path.is_file() else {}
            for name, expected in complete_hashes.items():
                check_file(folder / name, expected)
            completion_records[complete_path] = complete_hashes
        hashes = shard.get("files_sha256", {})
        for key, hash_key in (("path", "sha256"), ("index", "index_sha256")):
            relative = shard[key]
            # Existing full cache maps basenames relative to the shard directory;
            # common caches may instead use cache-root-relative keys or explicit hashes.
            expected_hashes = [hashes[name] for name in {relative, Path(relative).name} if name in hashes]
            if hash_key in shard:
                expected_hashes.append(shard[hash_key])
            for expected in expected_hashes or [None]:
                check_file(relative, expected)
        cursor = shard["end"]
    if cursor != manifest["num_frames"] or cursor <= 0:
        raise ValueError("Incomplete image shard coverage")


def main():
    from scripts.lpwm_full.evaluate import protocol_suites, suites_for_count, validate_result
    from scripts.lpwm_full.train import validate_parallel_eval

    a = parse_args()
    root = a.root.resolve()
    with (root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(state, **values):
            atomic(root / "pipeline_status.json", {"status": state, "updated_unix": time.time(), **values})

        try:
            verify_profile(root, a)
            verify_preflight(root, a)
            while not (root / "cache/manifest.json").is_file():
                progress = (
                    read(root / "cache/preparation_status.json")
                    if (root / "cache/preparation_status.json").exists()
                    else {}
                )
                pid = int((root / "preparation.pid").read_text())
                cmd = Path(f"/proc/{pid}/cmdline")
                command_line = cmd.read_bytes() if cmd.is_file() else b""
                if not (
                    b"tools/cache.py" in command_line
                    or (
                        b"lpwm_full" in command_line
                        and (b"cache" in command_line or b"prepare_common40" in command_line)
                    )
                ):
                    raise RuntimeError(
                        f"Cache preparation exited before complete {a.task_count}-task manifest"
                    )
                if shutil.disk_usage(root).free < (3 if a.steps >= 80000 else 7) * 2**30:
                    raise OSError("Preparation disk reserve breached; training will not start")
                status("preparing_data", cache_progress=progress, cache_pid=pid, training_started=False)
                time.sleep(30)
            manifest = read(root / "cache/manifest.json")
            verify_cache(root / "cache", manifest, a.task_count, a.cache_storage_root)
            reserve_gib = 12 if a.steps == 200000 else 5
            if shutil.disk_usage(root).free < reserve_gib * 2**30:
                raise OSError(f"Preserve{reserve_gib}GiB for retained checkpoints and disk margin")
            for name, expected in read(root / "frozen_source_sha256.json").items():
                if digest(root / "repo" / name) != expected:
                    raise ValueError(f"Frozen training source changed: {name}")
            run = root / "run"
            if run.exists():
                raise FileExistsError("Productionrun exists; no accidentalrestart/overwrite")
            command = [
                str(a.train_python),
                "-u",
                "-m",
                "scripts.lpwm_full.train",
                "--data",
                str(root / "cache"),
                "--output",
                str(run),
                "--steps",
                str(a.steps),
                "--schedule-steps",
                str(a.steps),
                "--world-weight",
                "1",
                "--rec-weight",
                "1",
                "--dyn-weight",
                "1",
                "--prior-weight",
                "0.001",
                "--workers",
                str(a.workers),
                "--eval-workers",
                str(a.eval_workers),
                "--batch-size",
                str(a.batch_size),
                "--grad-accumulation",
                str(a.grad_accumulation),
                "--validation-batches",
                str(a.task_count),
                "--project",
                "lpwm-fm-b-full-libero",
                "--run-name",
                a.run_name
                or (
                    f"{'common40' if a.task_count == 40 else 'full130'}"
                    f"-B-w1-rec1-dyn1-cos{a.steps // 1000}k-seed42"
                ),
                "--credential-file",
                str(a.credential_file),
                "--eval-python",
                str(a.eval_python),
                "--eval-runner",
                str(root / "repo/scripts/lpwm_full/evaluate.py"),
                "--expected-init-sha256",
                a.expected_init_sha256,
            ]
            env = os.environ.copy()
            env.update(
                PYTHONPATH=f"{root}/repo:{root}/repo/src",
                LIBERO_CONFIG_PATH=str(a.libero_config),
                OMP_NUM_THREADS="4",
                PYTHONUNBUFFERED="1",
            )
            env.setdefault("MUJOCO_GL", "egl")
            env.setdefault("PYOPENGL_PLATFORM", "egl")
            # With an external CUDA mask/UUID, do not guess a physical EGL device index.
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
            status(
                "starting_training",
                training_started=False,
                frames=manifest["num_frames"],
                episodes=len(manifest["episodes"]),
            )
            with (root / "training.log").open("x") as log:
                proc = subprocess.Popen(
                    command, cwd=root / "repo", env=env, stdout=log, stderr=subprocess.STDOUT
                )
                atomic(root / "training.pid", proc.pid)
                while proc.poll() is None:
                    current = read(run / "status.json") if (run / "status.json").exists() else {}
                    status("training", training_started=True, training_pid=proc.pid, training_status=current)
                    time.sleep(30)
                if proc.returncode:
                    raise RuntimeError(f"Training/evaluation exited{proc.returncode}; inspecttraining.log")
            if (
                read(run / "status.json").get("status") != "completed"
                or read(run / "status.json").get("step") != a.steps
            ):
                raise ValueError("Not actually completed the requested training budget")
            for step in range(5000, a.steps + 1, 5000):
                result = read(run / "eval" / f"step_{step:06d}.json")
                receipt = read(run / "eval" / f"step_{step:06d}.swanlab.json")
                validate_result(result)
                validate_parallel_eval(result, a.eval_workers)
                if (
                    protocol_suites(result["protocol"]) != suites_for_count(a.task_count)
                    or result["checkpoint"]["step"] != step
                    or not receipt.get("verified_upload")
                    or receipt.get("formal_result_eligible") is not True
                    or receipt.get("result_sha256") != digest(run / "eval" / f"step_{step:06d}.json")
                ):
                    raise ValueError("Missing full checkpoint rollout or verified formal upload")
            status(
                "completed",
                training_started=True,
                steps=a.steps,
                task_count=a.task_count,
                evaluated_episodes=(a.steps // 5000) * a.task_count * 10,
            )
        except BaseException as error:
            status("failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
