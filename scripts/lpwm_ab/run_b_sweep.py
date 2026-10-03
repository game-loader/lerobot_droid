#!/usr/bin/env python
"""Fail-closed, durable two-GPU B-only sweep over an already-snapshotted source tree.

The trainer owns run directories and synchronous checkpoint evaluation. This queue
never resumes training, prunes checkpoints, replaces a run, or signals a process.
Use --resume to verify and continue a queue; --recover-stale RUN_ID only adopts a
fully completed orphan after an explicit operator check and a liveness check.

Output contract: run/status.json, run/experiment.json, and six
run/eval/step_NNNNNN.json files in the evaluate.py schema. Evaluations must
bind to immutable checkpoint bundles beneath their own run directory. The source
passed as --repo is used as-is (not copied) and fingerprinted for every round.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STEPS = tuple(range(5_000, 30_001, 5_000))
SCORE_STEPS = (20_000, 25_000, 30_000)
GIB = 2**30
ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "experiment.json",
    "state_normalization.json",
    "language_metadata.json",
    "language_embeddings.npy",
    "language_masks.npy",
)
FIXED = {
    "steps": 30_000,
    "prior_weight": 0.001,
    "world_ramp_steps": 1_000,
    "lr": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 500,
    "batch_size": 8,
    "grad_accumulation": 4,
    "seed": 42,
    "workers": 4,
}


class QueueError(RuntimeError):
    """A failed gate; no subsequent work may be launched."""


def require(condition, message):
    if not condition:
        raise QueueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def invalid(value):
        raise QueueError(f"Nonfinite JSON value {value}: {path}")

    try:
        value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    except (OSError, ValueError) as error:
        raise QueueError(f"Missing/unreadable JSON {path}: {error}") from error
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def atomic_json(path, value):
    """Commit both contents and rename to disk before advancing queue state."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_hash(path):
    try:
        with Path(path).open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError as error:
        raise QueueError(f"Missing/unreadable artifact {path}: {error}") from error


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def spec(run_id, phase, round_number, gpu, world, rec=1.0, dyn=1.0):
    return {
        "id": run_id,
        "phase": phase,
        "round": round_number,
        "gpu": gpu,
        "world_weight": world,
        "reconstruction_weight": rec,
        "dynamics_weight": dyn,
    }


def phase1_rounds():
    return [
        [spec("phase1_world_0p03", 1, 1, 0, 0.03), spec("phase1_world_0p3", 1, 1, 1, 0.3)],
        [spec("phase1_world_0p1", 1, 2, 0, 0.1), spec("phase1_world_1p0", 1, 2, 1, 1.0)],
    ]


def phase2_round(world):
    return [
        spec("phase2_rec_0p5_dyn_1p5", 2, 3, 0, world, 0.5, 1.5),
        spec("phase2_rec_1p5_dyn_0p5", 2, 3, 1, world, 1.5, 0.5),
    ]


def expected_settings(run):
    return {
        **FIXED,
        "variant": "B",
        **{key: run[key] for key in ("world_weight", "reconstruction_weight", "dynamics_weight")},
    }


def child_environment(args, run):
    env = os.environ.copy()  # In particular, preserve LIBERO_CONFIG_PATH.
    env.update(
        CUDA_VISIBLE_DEVICES=str(run["gpu"]),
        MUJOCO_EGL_DEVICE_ID=str(run["gpu"]),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONPATH=str(args.repo / "src"),
        PYTHONUNBUFFERED="1",
    )
    return env


def train_command(args, run, expected=None):
    command = [str(args.python), "-u", str(args.repo / "scripts/lpwm_ab/train_b_sweep.py")]
    if args.uv is not None:
        command = [str(args.uv), "run", "--no-project", "--no-config", "--", *command]
    command += ["--data", str(args.data), "--output", str(args.root / "runs" / run["id"])]
    for key, value in expected_settings(run).items():
        if key != "variant":  # train_b_sweep.py is B-only, not the old A/B trainer.
            command += ["--" + key.replace("_", "-"), str(value)]
    command += [
        "--eval-runner",
        str(args.repo / "scripts/lpwm_ab/eval_b_sweep.py"),
        "--eval-python",
        str(args.eval_python),
        "--credential-file",
        str(args.credential_file),
        "--project",
        args.project,
        "--run-name",
        run["id"],
    ]
    if expected is not None:
        command += [
            "--expected-init-sha256",
            expected["initial_weights_sha256"],
            "--expected-split-sha256",
            expected["split_sha256"],
        ]
    return command


def verify_counts(result, count, where):
    require(
        type(result.get("num_episodes")) is int and result["num_episodes"] == count,
        f"Expected {count} complete episodes: {where}",
    )
    successes = result.get("successes")
    require(type(successes) is int and 0 <= successes <= count, f"Invalid successes: {where}")
    rate = result.get("success_rate")
    require(
        type(rate) in (int, float)
        and math.isfinite(rate)
        and math.isclose(rate, successes / count, rel_tol=0, abs_tol=1e-12),
        f"Inconsistent success_rate: {where}",
    )
    if "pc_success" in result:
        require(
            type(result["pc_success"]) in (int, float)
            and math.isclose(result["pc_success"], 100 * successes / count, rel_tol=0, abs_tol=1e-9),
            f"Inconsistent pc_success: {where}",
        )
    return successes


def verify_upload(path, experiment):
    """The wrapper commits verified_upload=True only after exact server metric read-back."""
    receipt_path = path.with_suffix(".swanlab.json")
    receipt = read_json(receipt_path)
    require(
        receipt.get("status") == "complete"
        and receipt.get("verified_upload") is True
        and receipt.get("uploaded") is True
        and receipt.get("evaluation_status") == "complete",
        f"Missing completed server-verified upload: {receipt_path}",
    )
    require(
        receipt.get("preflight") is False
        and receipt.get("formal_result_eligible") is True
        and receipt.get("protocol_role") == "validation"
        and receipt.get("mode") == "online"
        and receipt.get("public") is False
        and receipt.get("project") == experiment.get("project")
        and isinstance(receipt.get("project"), str)
        and bool(receipt["project"]),
        f"Wrong upload project/privacy/production protocol: {receipt_path}",
    )
    result_path = receipt.get("result")
    require(
        isinstance(result_path, str)
        and Path(result_path).is_absolute()
        and Path(result_path).resolve() == path.resolve()
        and receipt.get("result_sha256") == file_hash(path),
        f"Upload receipt does not bind this evaluation JSON: {receipt_path}",
    )
    attempt = receipt.get("attempt_id")
    require(
        isinstance(attempt, str)
        and len(attempt) == 32
        and all(c in "0123456789abcdef" for c in attempt)
        and isinstance(receipt.get("id"), str)
        and len(receipt["id"]) == 8
        and all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in receipt["id"]),
        f"Upload receipt lacks fresh server run identity: {receipt_path}",
    )
    return {"path": str(receipt_path), "sha256": file_hash(receipt_path), "run_id": receipt["id"]}


def verify_evaluation(path, run_path, step, experiment):
    result = read_json(path)
    require(result.get("status") == "complete", f"Evaluation not complete: {path}")
    successes = verify_counts(result, 100, path)
    protocol = result.get("protocol", {})
    require(
        isinstance(protocol, dict)
        and protocol.get("seed") == 42
        and protocol.get("seed_namespace") == "validation"
        and protocol.get("suite") == "libero_spatial"
        and protocol.get("task_ids") == list(range(10))
        and protocol.get("episodes_per_task") == 10
        and protocol.get("name") == "lpwm-libero-spatial-rollout-v1"
        and protocol.get("init_state_offset") == 0
        and "requested_max_steps" in protocol
        and protocol["requested_max_steps"] is None
        and protocol.get("max_control_steps") == protocol.get("local_suite_max_steps") == 280,
        f"Wrong paired 100-episode protocol: {path}",
    )
    require(result.get("protocol_sha256") == json_hash(protocol), f"Protocol hash mismatch: {path}")
    tasks = result.get("per_task")
    require(isinstance(tasks, list) and len(tasks) == 10, f"Missing episode evidence: {path}")
    plans, task_ids, actual_successes = [], set(), 0
    for task in tasks:
        require(isinstance(task, dict), f"Invalid task evidence: {path}")
        task_id = task.get("task_id")
        require(
            type(task_id) is int and task_id in range(10) and task_id not in task_ids,
            f"Invalid/duplicate task: {path}",
        )
        task_ids.add(task_id)
        episodes = task.get("episodes")
        require(isinstance(episodes, list) and len(episodes) == 10, f"Partial task episodes: {path}")
        indices, initial_states, task_successes = set(), set(), 0
        for episode in episodes:
            require(
                isinstance(episode, dict) and type(episode.get("success")) is bool,
                f"Incomplete episode: {path}",
            )
            index, initial, seed = (episode.get(k) for k in ("episode_index", "init_state_index", "seed"))
            require(
                type(index) is int
                and index in range(10)
                and index not in indices
                and type(initial) is int
                and initial >= 0
                and initial not in initial_states
                and type(seed) is int
                and seed >= 0,
                f"Invalid/duplicate episode plan: {path}",
            )
            indices.add(index)
            initial_states.add(initial)
            plans.append([task_id, index, initial, seed])
            task_successes += int(episode["success"])
        require(verify_counts(task, 10, path) == task_successes, f"Task successes mismatch: {path}")
        actual_successes += task_successes
    require(actual_successes == successes, f"Aggregate successes mismatch: {path}")
    checkpoint = result.get("checkpoint", {})
    require(
        isinstance(checkpoint, dict)
        and checkpoint.get("step") == step
        and checkpoint.get("variant") == "B"
        and checkpoint.get("training_seed") == 42
        and checkpoint.get("split_sha256") == experiment["split_sha256"],
        f"Wrong checkpoint identity at step {step}: {path}",
    )
    require(isinstance(checkpoint.get("path"), str), f"Missing checkpoint path: {path}")
    checkpoint_path = Path(checkpoint["path"])
    require(
        checkpoint_path.is_absolute()
        and checkpoint_path.is_dir()
        and checkpoint_path.resolve().is_relative_to(run_path.resolve())
        and not checkpoint_path.is_symlink()
        and checkpoint_path.name not in {"latest", "best", "best_success"},
        f"Expected immutable checkpoint inside its run: {path}",
    )
    require(
        checkpoint_path.resolve() == (run_path / "checkpoints" / f"step_{step:06d}").resolve(),
        f"Checkpoint is not the expected immutable step directory: {path}",
    )
    marker = checkpoint_path / "complete.json"
    require(marker.is_file() and not marker.is_symlink(), f"Missing completion manifest: {path}")
    manifest = read_json(marker)
    require(
        manifest.get("status") == "complete"
        and manifest.get("step") == step
        and manifest.get("preflight") is False,
        f"Incomplete/preflight checkpoint manifest: {path}",
    )
    for key in ("initial_weights_sha256", "split_sha256"):
        require(manifest.get(key) == experiment[key], f"Checkpoint manifest {key} mismatch: {path}")
    all_hashes = manifest.get("sha256", {})
    require(
        isinstance(all_hashes, dict) and set(ARTIFACTS) | {"optimizer.pt", "split.json"} <= set(all_hashes),
        f"Incomplete checkpoint artifact manifest: {path}",
    )
    actual = {str(p.relative_to(checkpoint_path)) for p in checkpoint_path.rglob("*") if p.is_file()}
    require(actual == set(all_hashes) | {marker.name}, f"Checkpoint file inventory mismatch: {path}")
    for name, digest in all_hashes.items():
        artifact = checkpoint_path / name
        require(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and artifact.resolve().is_relative_to(checkpoint_path.resolve())
            and not artifact.is_symlink()
            and valid_hash(digest)
            and artifact.stat().st_size > 0
            and file_hash(artifact) == digest,
            f"Checkpoint artifact changed: {artifact}",
        )
    hashes = checkpoint.get("artifact_sha256", {})
    require(
        isinstance(hashes, dict)
        and set(ARTIFACTS) <= set(hashes) <= set(all_hashes)
        and all(all_hashes[name] == digest for name, digest in hashes.items()),
        f"Evaluation and completion manifest disagree: {path}",
    )
    require(
        checkpoint.get("model_sha256") == hashes["model.safetensors"]
        and checkpoint.get("bundle_sha256") == json_hash(hashes),
        f"Bundle hash mismatch: {path}",
    )
    metadata = read_json(checkpoint_path / "experiment.json")
    for key, expected in {
        "step": step,
        "variant": "B",
        "seed": 42,
        "split_sha256": experiment["split_sha256"],
        "initial_weights_sha256": experiment["initial_weights_sha256"],
    }.items():
        require(metadata.get(key) == expected, f"Checkpoint {key} mismatch: {path}")
    upload = verify_upload(path, experiment)
    return {
        "step": step,
        "upload": upload,
        "successes": successes,
        "success_rate": successes / 100,
        "checkpoint": str(checkpoint_path),
        "result": str(path),
        "result_sha256": file_hash(path),
        "protocol_sha256": result["protocol_sha256"],
        "episode_plan_sha256": json_hash(sorted(plans)),
    }


def verify_run(run_path, run):
    require(run_path.is_dir() and not run_path.is_symlink(), f"Missing/aliased run: {run_path}")
    status = read_json(run_path / "status.json")
    require(
        status.get("status") == "completed" and status.get("step") == 30_000,
        f"Training did not complete all 30000 steps: {run_path}",
    )
    experiment = read_json(run_path / "experiment.json")
    require(
        experiment.get("preflight") is False and experiment.get("disable_eval") is False,
        f"Production completion cannot be preflight or bypass evaluation: {run_path}",
    )
    aliases = {"reconstruction_weight": "rec_weight", "dynamics_weight": "dyn_weight"}
    for key, expected in expected_settings(run).items():
        present = [name for name in {key, aliases.get(key, key)} if name in experiment]
        require(
            present
            and all(experiment[name] == expected and type(experiment[name]) is not bool for name in present),
            f"Training setting {key} mismatch in {run_path}: expected {expected!r}",
        )
    for key in ("split_sha256", "initial_weights_sha256"):
        require(valid_hash(experiment.get(key)), f"Missing {key}: {run_path}")
    evaluations = [
        verify_evaluation(run_path / "eval" / f"step_{step:06d}.json", run_path, step, experiment)
        for step in STEPS
    ]
    for key in ("protocol_sha256", "episode_plan_sha256"):
        require(len({row[key] for row in evaluations}) == 1, f"Unpaired evaluations: {run_path}")
    best = max(evaluations, key=lambda row: (row["successes"], -row["step"]))
    total = sum(row["successes"] for row in evaluations if row["step"] in SCORE_STEPS)
    return {
        "id": run["id"],
        "world_weight": run["world_weight"],
        "reconstruction_weight": run["reconstruction_weight"],
        "dynamics_weight": run["dynamics_weight"],
        "score_successes": total,
        "score": total / 300,
        "final_successes": evaluations[-1]["successes"],
        "final_success_rate": evaluations[-1]["success_rate"],
        "best_checkpoint": best,
        "evaluations": evaluations,
        "split_sha256": experiment["split_sha256"],
        "initial_weights_sha256": experiment["initial_weights_sha256"],
        "protocol_sha256": evaluations[0]["protocol_sha256"],
        "episode_plan_sha256": evaluations[0]["episode_plan_sha256"],
    }


def verify_paired(summaries):
    for key in ("split_sha256", "initial_weights_sha256", "protocol_sha256", "episode_plan_sha256"):
        require(len({row[key] for row in summaries}) == 1, f"Runs disagree on paired {key}")


def select_phase1(summaries):
    expected = {run["id"] for pair in phase1_rounds() for run in pair}
    require(
        len(summaries) == 4 and {row["id"] for row in summaries} == expected,
        "Selection requires ALL FOUR completed and verified phase-1 runs",
    )
    verify_paired(summaries)
    ranked = sorted(
        summaries, key=lambda row: (-row["score_successes"], -row["final_successes"], row["world_weight"])
    )
    return {
        "criterion": "mean success at 20000,25000,30000; tie final30000; tie smaller world_weight",
        "score_steps": list(SCORE_STEPS),
        "winner": ranked[0]["id"],
        "world_weight": ranked[0]["world_weight"],
        "control": ranked[0]["id"],
        "ranking": ranked,
        "best_checkpoint_is_descriptive_only": True,
    }


def source_fingerprint(repo):
    for name in ("train_b_sweep.py", "eval_b_sweep.py"):
        require((repo / "scripts/lpwm_ab" / name).is_file(), f"Missing snapshot runner: {name}")
    paths = sorted(
        {
            *repo.glob("scripts/lpwm_ab/*.py"),
            *repo.glob("src/**/*.py"),
            *(p for p in (repo / "pyproject.toml", repo / "uv.lock") if p.is_file()),
        }
    )
    return json_hash({str(path.relative_to(repo)): file_hash(path) for path in paths})


def configuration(args):
    return {
        **{key: str(getattr(args, key)) for key in ("repo", "python", "eval_python", "data", "root")},
        "uv": str(args.uv) if args.uv else None,
        "project": args.project,
        "source_sha256": source_fingerprint(args.repo),
        "data_manifest_sha256": file_hash(args.data / "manifest.json"),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "min_free_gib": args.min_free_gib,
        "estimated_run_gib": args.estimated_run_gib,
        "fixed": FIXED,
        "phase1": phase1_rounds(),
    }


@contextlib.contextmanager
def queue_lock(root, resume):
    require(not root.is_symlink(), "Queue root must not be a symlink")
    if not resume:
        # Deployment creates repo/ and suite metadata here before the queue starts.
        # Serialize first, then reject conflicting queue-owned outputs below.
        root.mkdir(parents=True, exist_ok=True)
    require(root.is_dir(), "--resume requires an existing queue root")
    lock = os.open(root / "queue.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueError("Queue lock is held by another queue or its still-running child") from error
        if not resume:
            for name in ("queue_state.json", "phase1_selection.json", "phase2_comparison.json"):
                require(
                    not os.path.lexists(root / name),
                    f"Queue output already exists: {name}; use --resume (never overwrite)",
                )
            for name in ("runs", "logs"):
                path = root / name
                require(
                    not os.path.lexists(path)
                    or path.is_dir()
                    and not path.is_symlink()
                    and not any(path.iterdir()),
                    f"Conflicting existing queue {name}: {path}",
                )
        yield lock
    finally:
        # Do NOT LOCK_UN: children inherit this open-file description, so an
        # orphan trainer retains the lock even after the queue exits or crashes.
        os.close(lock)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def group_alive(record):
    require(record.get("hostname") == socket.gethostname(), "Cannot recover a run from another host")
    if record.get("boot_id") != boot_id():
        return False
    pgid = record.get("pgid")
    if pgid is None:
        # Crash in the spawn/record window: inherited queue lock plus explicit
        # operator confirmation and complete outputs are still required.
        return False
    require(type(pgid) is int and pgid > 1, "Invalid recorded process group")
    try:
        os.killpg(pgid, 0)  # Liveness probe only; NEVER send a destructive signal.
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def disk_guard(args, future_runs, round_runs):
    free = shutil.disk_usage(args.root).free
    # Reserve at least the floor for BOTH workers before starting a round.
    needed = args.min_free_gib * max(1, round_runs)
    if args.estimated_run_gib is not None:
        needed = max(needed, args.min_free_gib + args.estimated_run_gib * future_runs)
    require(
        free >= needed * GIB,
        f"Disk guard: {free / GIB:.2f} GiB free, need {needed:.2f} GiB; no pruning is performed",
    )
    return {"free_bytes": free, "required_bytes": math.ceil(needed * GIB), "checked_unix": time.time()}


class SweepQueue:
    def __init__(self, args, lock):
        self.args, self.lock = args, lock
        self.path = args.root / "queue_state.json"
        config = configuration(args)
        if args.resume:
            self.state = read_json(self.path)
            require(self.state.get("schema_version") == 1, "Unsupported queue state schema")
            require(self.state.get("configuration") == config, "Queue configuration/source/data changed")
            require(isinstance(self.state.get("runs"), dict), "Invalid queue run state")
        else:
            (args.root / "runs").mkdir(exist_ok=True)
            (args.root / "logs").mkdir(exist_ok=True)
            self.state = {
                "schema_version": 1,
                "configuration": config,
                "status": "pending",
                "created_unix": time.time(),
                "runs": {},
                "events": [],
            }
            self.save()

    def save(self):
        self.state["updated_unix"] = time.time()
        atomic_json(self.path, self.state)

    def event(self, kind, run_id=None, **fields):
        print(
            json.dumps({"event": kind, "run": run_id, "unix": time.time(), **fields}, sort_keys=True),
            flush=True,
        )
        self.state["events"].append({"event": kind, "run": run_id, "unix": time.time(), **fields})
        self.save()

    def verified(self, run):
        path = self.args.root / "runs" / run["id"]
        experiment = read_json(path / "experiment.json")
        require(
            experiment.get("dataset_manifest_sha256") == self.state["configuration"]["data_manifest_sha256"],
            f"Dataset manifest mismatch: {path}",
        )
        require(experiment.get("data") == str(self.args.data), f"Dataset path mismatch: {path}")
        require(experiment.get("project") == self.args.project, f"Project mismatch: {path}")
        return verify_run(path, run)

    def reconcile(self):
        recover = set(self.args.recover_stale)
        require(recover <= set(self.state["runs"]), "Unknown --recover-stale run ID")
        for run_id, record in self.state["runs"].items():
            require(record.get("spec", {}).get("id") == run_id, "Corrupt recorded run identity")
            status = record.get("status")
            if status == "completed":
                require(run_id not in recover, f"{run_id} is already completed, not stale")
                record["summary"] = self.verified(record["spec"])
            elif status in {"starting", "running", "interrupted"}:
                require(
                    run_id in recover, f"{run_id} is running/stale; inspect it, then --recover-stale {run_id}"
                )
                require(
                    not group_alive(record), f"{run_id} process group is still alive; will not duplicate it"
                )
                record["summary"] = self.verified(record["spec"])
                record.update(status="completed", finished_unix=time.time(), recovered=True)
                self.event("recovered_verified_completion", run_id)
            else:
                raise QueueError(f"{run_id} has terminal/unknown status {status!r}; use a NEW root, no retry")
        if self.state["runs"]:
            verify_paired([record["summary"] for record in self.state["runs"].values()])
        self.save()

    def run_round(self, runs, future_runs):
        require(
            source_fingerprint(self.args.repo) == self.state["configuration"]["source_sha256"],
            "Snapshot source changed during the queue",
        )
        require(
            file_hash(self.args.data / "manifest.json")
            == self.state["configuration"]["data_manifest_sha256"],
            "Dataset manifest changed during the queue",
        )
        pending = []
        for run in runs:
            record = self.state["runs"].get(run["id"])
            if record is not None:
                require(record["spec"] == run and record["status"] == "completed", "Run plan/status changed")
                record["summary"] = self.verified(run)
            else:
                for path in (
                    self.args.root / "runs" / run["id"],
                    self.args.root / "logs" / f"{run['id']}.log",
                ):
                    require(not os.path.lexists(path), f"Refusing preexisting run/log: {path}")
                pending.append(run)
        children, launch_error = [], None
        try:
            for index, run in enumerate(pending):
                try:
                    guard = disk_guard(self.args, future_runs - index, len(pending) - index)
                    summaries = [
                        record["summary"]
                        for record in self.state["runs"].values()
                        if record["status"] == "completed"
                    ]
                    command = train_command(self.args, run, summaries[0] if summaries else None)
                    record = {
                        "spec": run,
                        "status": "starting",
                        "started_unix": time.time(),
                        "hostname": socket.gethostname(),
                        "boot_id": boot_id(),
                        "command": command,
                        "disk_guard": guard,
                    }
                    self.state["runs"][run["id"]] = record
                    self.event("starting", run["id"])
                    # Logs live outside the trainer-owned run directory and are exclusive.
                    with (self.args.root / "logs" / f"{run['id']}.log").open("x") as log:
                        process = subprocess.Popen(
                            command,
                            cwd=self.args.repo,
                            env=child_environment(self.args, run),
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(self.lock,),
                        )
                    children.append((process, run))
                    record.update(status="running", pid=process.pid, pgid=process.pid)
                    self.event("started", run["id"], pid=process.pid)
                except Exception as error:
                    launch_error = error
                    record = self.state["runs"].get(run["id"])
                    if record is not None and record["status"] == "starting":
                        record.update(status="failed", error=str(error), finished_unix=time.time())
                        self.event("failed", run["id"], error=str(error))
                    break
            # A failing worker never terminates its sibling. Drain both before
            # reporting failure; never launch the next round/phase after failure.
            while children:
                for process, run in children[:]:
                    rc = process.poll()
                    if rc is None:
                        continue
                    process.wait()
                    record = self.state["runs"][run["id"]]
                    record.update(exit_code=rc, finished_unix=time.time())
                    try:
                        require(rc == 0, f"Trainer exited {rc}: {run['id']}")
                        require(
                            not group_alive(record), f"Trainer left a live child process group: {run['id']}"
                        )
                        record["summary"] = self.verified(run)
                        record["status"] = "completed"
                    except Exception as error:
                        record.update(status="failed", error=str(error))
                    self.event(record["status"], run["id"], exit_code=rc)
                    children.remove((process, run))
                if children:
                    time.sleep(self.args.poll_seconds)
        except BaseException:
            for _, run in children:
                self.state["runs"][run["id"]]["status"] = "interrupted"
            self.save()
            raise
        if launch_error is not None:
            raise QueueError(f"Round launch failed: {launch_error}") from launch_error
        failures = [run["id"] for run in runs if self.state["runs"][run["id"]]["status"] != "completed"]
        require(not failures, f"Round failed; no subsequent stages launched: {failures}")
        verify_paired([record["summary"] for record in self.state["runs"].values()])

    def execute(self):
        try:
            self.reconcile()
            self.state.update(status="running", started_unix=self.state.get("started_unix", time.time()))
            self.state.pop("finished_unix", None)
            self.state.pop("error", None)
            self.event("queue_started")
            for runs in phase1_rounds():
                self.run_round(runs, future_runs=6 - len(self.state["runs"]))
            summaries = [self.verified(run) for pair in phase1_rounds() for run in pair]
            selection = select_phase1(summaries)
            if "selection" in self.state:
                require(
                    self.state["selection"] == selection,
                    "Verified selection differs from committed selection",
                )
            self.state["selection"] = selection
            atomic_json(self.args.root / "phase1_selection.json", selection)
            self.event("phase1_selected", selection["winner"])
            # Phase 1's 1:1 winner is the control: no retraining or warm start.
            self.run_round(phase2_round(selection["world_weight"]), future_runs=6 - len(self.state["runs"]))
            comparison = {
                "control": selection["control"],
                "world_weight": selection["world_weight"],
                "runs": [
                    self.state["runs"][run_id]["summary"]
                    for run_id in [
                        selection["control"],
                        *(r["id"] for r in phase2_round(selection["world_weight"])),
                    ]
                ],
            }
            atomic_json(self.args.root / "phase2_comparison.json", comparison)
            self.state.update(status="completed", finished_unix=time.time())
            self.event("queue_completed")
            return self.state
        except BaseException as error:
            self.state.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                finished_unix=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
            self.event("queue_" + self.state["status"], error=str(error))
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("repo", "python", "eval-python", "root", "data", "credential-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--project", default="lpwm-fm-b-world-balance")
    parser.add_argument("--uv", type=Path, help="Optional uv executable, e.g. /opt/conda/bin/uv")
    parser.add_argument(
        "--resume", action="store_true", help="Verify completed entries; never restart training"
    )
    parser.add_argument(
        "--recover-stale",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="With --resume, explicitly confirm inspection of an orphan; adopt only verified completion",
    )
    parser.add_argument(
        "--min-free-gib", type=float, default=3.0, help="Per-launch free-space floor (at least 3 GiB)"
    )
    parser.add_argument(
        "--estimated-run-gib",
        type=float,
        help="Operator estimate of ALL retained checkpoints per fresh run; reserve all future runs + floor",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.recover_stale and not args.resume:
        parser.error("--recover-stale requires --resume")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 3:
        parser.error("--min-free-gib must be finite and >= 3")
    if args.estimated_run_gib is not None and (
        not math.isfinite(args.estimated_run_gib) or args.estimated_run_gib <= 0
    ):
        parser.error("--estimated-run-gib must be finite and positive")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    # Preserve interpreter symlinks: resolving venv/bin/python changes the environment.
    for name in ("repo", "python", "eval_python", "root", "data", "credential_file", "uv"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, Path(os.path.abspath(path.expanduser())))
    return args


def main(argv=None):
    args = parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; children are NOT killed; inspect before recovery")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for name in ("python", "eval_python", "uv"):
            executable = getattr(args, name)
            require(
                executable is None or executable.is_file() and os.access(executable, os.X_OK),
                f"Missing/non-executable --{name}: {executable}",
            )
        require(
            args.credential_file.is_file(),
            "Missing credential file (contents are never copied into queue state)",
        )
        require(
            args.repo.is_dir() and (args.repo / "src").is_dir(), "--repo must be the supplied source snapshot"
        )
        with queue_lock(args.root, args.resume) as lock:
            SweepQueue(args, lock).execute()
        return 0
    except (QueueError, OSError, KeyboardInterrupt) as error:
        print(f"B_SWEEP_STOPPED: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
