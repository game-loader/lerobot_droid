"""Isolated native LIBERO episode-queue benchmark, never a production evaluator.

Compare the unchanged task-level helper against episode-level scheduling. Each
worker owns one policy/RNG and at most one live environment. Consecutive jobs for
the same task reuse that environment, but retain the original hard reset and
per-episode seed/initial-state plan AND reset ordinal by replaying skipped resets.
No central inference batching or soft reset.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import time
from multiprocessing.util import Finalize
from pathlib import Path

if __package__:
    from . import benchmark_parallel as base
else:
    import benchmark_parallel as base

ALLOWED_WORKERS = (1, 2, 4, 8, 10, 16, 32)
_ENV = None
_ENV_TASK_ID = None
_ENV_META = None
_ENV_FINALIZER = None
_ENV_NEXT_EPISODE = 0


def make_jobs(tasks, episodes_per_task):
    if not 1 <= episodes_per_task <= 10:
        raise ValueError("Use 1..10 episodes from the original ten-episode plan")
    ids = [task["global_task_id"] for task in tasks]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Task jobs must be nonempty and unique")
    return [{"task": task, "episode_index": index} for task in tasks for index in range(episodes_per_task)]


def phase_plan(seed):
    """Bracket shuffled scaling conditions by repeated scheduler-specific controls."""
    scaling = [("episode", workers) for workers in (10, 16, 32)]
    random.Random(seed).shuffle(scaling)
    return [("task", 8), ("episode", 8), *scaling, ("episode", 8), ("task", 8)]


def close_environment():
    global _ENV, _ENV_TASK_ID, _ENV_META, _ENV_NEXT_EPISODE
    if _ENV is not None:
        _ENV.env.close()
    _ENV = _ENV_TASK_ID = _ENV_META = None
    _ENV_NEXT_EPISODE = 0


def initialize_worker(settings, ready):
    global _ENV_FINALIZER
    base.initialize_worker(settings, ready)
    _ENV_FINALIZER = Finalize(None, close_environment, exitpriority=10)


def environment_for(task):
    global _ENV, _ENV_TASK_ID, _ENV_META
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
    from scripts.lpwm_ab import evaluate as native

    task_id = task["global_task_id"]
    if _ENV is not None and task_id == _ENV_TASK_ID:
        return _ENV, _ENV_META, False
    close_environment()
    suite = _get_suite(task["suite"])
    actual = suite.get_task(task["task_id"])
    if actual.name != task["name"] or native.normalize_task_description(
        actual.language
    ) != native.normalize_task_description(task["language"]):
        raise ValueError("Task identity differs from checkpoint catalog")
    limit = TASK_SUITE_MAX_STEPS[task["suite"]]
    bundle = base._BUNDLE
    env = LiberoEnv(
        task_suite=suite,
        task_id=task["task_id"],
        task_suite_name=task["suite"],
        episode_length=limit,
        camera_name=list(bundle["mapping"]),
        camera_name_mapping=bundle["mapping"],
        obs_type="pixels_agent_pos",
        observation_width=128,
        observation_height=128,
        init_states=True,
        n_envs=1,
        num_steps_wait=10,
        control_freq=20,
        control_mode="relative",
        hard_reset=True,
    )
    _ENV = base.AuditEnv(env)
    _ENV_TASK_ID = task_id
    _ENV_META = {
        "limit": limit,
        "language": actual.language,
        "plans": native.episode_plan(len(env._init_states), 10, 42, "validation", task_id),
    }
    return _ENV, _ENV_META, True


def prepare_episode(env, metadata, episode_index):
    """Replay skipped resets to match the original per-task environment lifecycle.

    LIBERO hard resets rebuild samplers but append property initializers. Thus
    reset ordinal affects RNG consumption and fixture placement even with an
    explicit seed and saved qpos/qvel. Skipping prior rollouts is allowed only
    after bitwise parity is verified; skipping their resets is not equivalent.
    """
    global _ENV_NEXT_EPISODE
    from scripts.lpwm_ab import evaluate as native

    if episode_index < _ENV_NEXT_EPISODE:
        raise ValueError("Per-worker task episode indices must increase")
    start = time.perf_counter()
    replayed = 0
    while episode_index > _ENV_NEXT_EPISODE:
        plan = metadata["plans"][_ENV_NEXT_EPISODE]
        native.seed_rollout(plan["seed"])
        env.init_state_id = plan["init_state_index"]
        env.reset(seed=plan["seed"])
        _ENV_NEXT_EPISODE += 1
        replayed += 1
    return replayed, time.perf_counter() - start


def evaluate_episode(job):
    global _ENV_NEXT_EPISODE
    import numpy as np
    import torch

    from scripts.lpwm_ab import evaluate as native

    start = time.perf_counter()
    start_unix = time.time()
    task = job["task"]
    env, metadata, created = environment_for(task)
    setup_seconds = time.perf_counter() - start
    replayed, replay_seconds = prepare_episode(env, metadata, job["episode_index"])
    bundle, settings = base._BUNDLE, base._SETTINGS
    language, _ = bundle["language_cache"].select(metadata["language"], torch.device("cuda"))
    rollout_start_unix = time.time()
    row = native.rollout_episode(
        env,
        bundle["policy"],
        metadata["plans"][job["episode_index"]],
        bundle["mapping"],
        bundle["state_stats"],
        language,
        torch.device("cuda"),
        metadata["limit"],
        128,
    )
    rollout_finished_unix = time.time()
    _ENV_NEXT_EPISODE = job["episode_index"] + 1
    actions = np.stack(env.actions)
    filename = f"task_{task['global_task_id']:03d}_episode_{row['episode_index']:02d}.npy"
    np.save(Path(settings["variant_output"]) / "actions" / filename, actions, allow_pickle=False)
    row.update(
        global_task_id=task["global_task_id"],
        suite=task["suite"],
        suite_task_id=task["task_id"],
        max_control_steps=metadata["limit"],
        action_trace=filename,
        action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        observation_sha256=env.observations.hexdigest(),
        worker_pid=os.getpid(),
        environment_created=created,
        replayed_resets=replayed,
        replayed_reset_seconds=replay_seconds,
        environment_setup_seconds=setup_seconds,
        job_start_unix=start_unix,
        rollout_start_unix=rollout_start_unix,
        rollout_finished_unix=rollout_finished_unix,
        job_finished_unix=time.time(),
        job_seconds=time.perf_counter() - start,
    )
    base.atomic(Path(settings["variant_output"]) / filename.replace(".npy", ".json"), row)
    return row


def peak_concurrency(episodes):
    """Measure simultaneous actual rollouts, globally and per task (not job count)."""
    tasks = sorted({row["global_task_id"] for row in episodes})

    def peak(rows):
        events = sorted(
            event
            for row in rows
            for event in ((row["rollout_start_unix"], 1), (row["rollout_finished_unix"], -1))
        )
        active = maximum = 0
        for _, delta in events:
            active += delta
            maximum = max(maximum, active)
        return maximum

    return {
        "peak_active_episodes": peak(episodes),
        "peak_same_task_episodes": {
            str(task): peak([row for row in episodes if row["global_task_id"] == task]) for task in tasks
        },
    }


def run_episode_variant(settings, tasks, workers, output):
    jobs = make_jobs(tasks, settings["episodes_per_task"])
    if workers not in ALLOWED_WORKERS or len(jobs) < workers:
        raise ValueError("Invalid worker count or too few episode jobs to fill workers")
    output.mkdir(exist_ok=False)
    (output / "actions").mkdir()
    settings = dict(settings, variant_output=str(output))
    context = mp.get_context("spawn")
    ready = context.Queue()
    start, start_unix = time.perf_counter(), time.time()
    pool = context.Pool(workers, initialize_worker, (settings, ready))
    try:
        workers_info = []
        for _ in range(workers):
            try:
                info = ready.get(timeout=240)
            except queue.Empty as exc:
                raise TimeoutError("Worker CUDA/model initialization timed out") from exc
            if info["status"] != "ready":
                raise RuntimeError(f"Worker initialization failed: {info}")
            workers_info.append(info)
        if len({info["model_sha256"] for info in workers_info}) != 1:
            raise ValueError("Workers loaded different models")
        ready_time, ready_unix = time.perf_counter(), time.time()
        base.atomic(output / "workers.json", workers_info)
        episodes = []
        for row in pool.imap_unordered(evaluate_episode, jobs, chunksize=1):
            episodes.append(row)
            status = {
                "status": "running",
                "scheduler": "episode",
                "workers": workers,
                "completed_episodes": len(episodes),
                "total_episodes": len(jobs),
                "elapsed_seconds": time.perf_counter() - start,
            }
            base.atomic(output / "status.json", status)
            print(json.dumps(dict(status, phase=output.name)), flush=True)
        finished, finished_unix = time.perf_counter(), time.time()
        pool.close()
        pool.join()
        pool = None
        cleanup_finished_unix = time.time()
        work_by_pid = {
            str(info["pid"]): [
                [row["global_task_id"], row["episode_index"]]
                for row in episodes
                if row["worker_pid"] == info["pid"]
            ]
            for info in workers_info
        }
        episodes.sort(key=lambda row: (row["global_task_id"], row["episode_index"]))
        result = {
            "status": "completed",
            "scheduler": "episode",
            "formal_result_eligible": False,
            "workers": workers,
            "actual_busy_workers": sum(bool(jobs) for jobs in work_by_pid.values()),
            "episode_jobs_by_worker": work_by_pid,
            "start_unix": start_unix,
            "ready_unix": ready_unix,
            "finished_unix": finished_unix,
            "cleanup_finished_unix": cleanup_finished_unix,
            "initializer_seconds": ready_time - start,
            "rollout_seconds": finished - ready_time,
            "end_to_end_seconds": finished - start,
            "with_cleanup_seconds": time.perf_counter() - start,
            "episode_count": len(episodes),
            "total_control_steps": sum(row["control_steps"] for row in episodes),
            "successes": sum(row["success"] for row in episodes),
            "environment_creations": sum(row["environment_created"] for row in episodes),
            "replayed_resets": sum(row["replayed_resets"] for row in episodes),
            "sum_replayed_reset_seconds": sum(row["replayed_reset_seconds"] for row in episodes),
            "sum_environment_setup_seconds": sum(row["environment_setup_seconds"] for row in episodes),
            "task_ids": [task["global_task_id"] for task in tasks],
            "worker_initialization": workers_info,
            "episodes": episodes,
            **peak_concurrency(episodes),
        }
        if result["actual_busy_workers"] != workers:
            raise ValueError("Not all requested workers executed episode jobs")
        base.atomic(output / "result.json", result)
        base.atomic(output / "status.json", summarize(result))
        return result
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        ready.close()


def summarize(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"episodes", "worker_initialization", "episode_jobs_by_worker", "task_jobs_by_worker"}
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(base.TASK_IDS))
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--schedule-seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    try:
        tasks = [{"global_task_id": task_id} for task_id in args.task_ids]
        jobs = make_jobs(tasks, args.episodes_per_task)
        if len(tasks) < 8 or len(jobs) < 32:
            raise ValueError("Need at least 8 task jobs and 32 distinct episode jobs")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Set explicit exclusive CUDA UUID before spawning workers")
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((args.checkpoint / "task_catalog.json").read_text())
    tasks = base.select_tasks(catalog, args.task_ids)
    settings = {
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu_uuid": args.gpu_uuid,
        "episodes_per_task": args.episodes_per_task,
    }
    phases = phase_plan(args.schedule_seed)
    protocol = dict(
        settings,
        task_ids=args.task_ids,
        phase_plan=phases,
        schedule_seed=args.schedule_seed,
        formal_result_eligible=False,
        start_method="spawn",
        hard_reset=True,
        full_control_horizons=True,
        original_plan_episodes=10,
        seed_namespace="validation",
        seed=42,
        job_order="task-major; dynamic imap_unordered chunksize=1; no task barriers",
        environment_cache="one most-recent task per process, hard reset every episode",
        reset_lifecycle="replay skipped prior episode resets/seeds/settle to preserve serial reset ordinal",
        model_sha256=base.file_hash(args.checkpoint / "model.safetensors"),
        helper_sha256=base.file_hash(Path(base.__file__)),
        driver_sha256=base.file_hash(Path(__file__)),
        inference="one independent policy replica per worker, no central GPU batching",
    )
    base.atomic(args.output / "protocol.json", protocol)
    baseline, summaries = None, []
    try:
        for index, (scheduler, workers) in enumerate(phases):
            output = args.output / f"phase_{index:02d}_{scheduler}_w{workers}"
            base.atomic(
                args.output / "status.json",
                {"status": "running", "phase": index, "scheduler": scheduler, "workers": workers},
            )
            if scheduler == "task":
                result = base.run_variant(settings, tasks, workers, output)
                result.update(scheduler="task", environment_creations=len(tasks))
                base.atomic(output / "result.json", result)
            else:
                result = run_episode_variant(settings, tasks, workers, output)
            baseline = baseline if baseline is not None else result
            parity = base.compare_results(baseline, result)
            summary = dict(summarize(result), phase=index, **parity)
            summaries.append(summary)
            base.atomic(args.output / "summary.json", {"status": "running", "phases": summaries})
            print(json.dumps(summary), flush=True)
            if not parity["bitwise_trajectory_parity"]:
                raise ValueError("Episode queue changed trajectories; results ineligible for speedup claims")
        if base.file_hash(args.checkpoint / "model.safetensors") != protocol["model_sha256"]:
            raise ValueError("Checkpoint changed during benchmark")
        base.atomic(args.output / "summary.json", {"status": "completed", "phases": summaries})
        base.atomic(args.output / "status.json", {"status": "completed", "phase_count": len(summaries)})
    except BaseException as exc:
        base.atomic(args.output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
