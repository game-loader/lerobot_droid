"""Nonformal native LIBERO process-parallel benchmark; no edits to production runs."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import time
from pathlib import Path

TASK_IDS = (0, 5, 10, 15, 20, 25, 30, 41, 52, 63, 74, 85, 96, 107, 120, 125)
_BUNDLE = None
_SETTINGS = None
ALLOWED_WORKERS = (1, 2, 4, 8, 16, 32)


def expanded_task_ids(seed=20260924):
    """64 fixed stratified task jobs, including the original16, chosen without outcomes."""
    rng = random.Random(seed)
    selected = []
    for offset, count, wanted in ((0, 10, 8), (10, 10, 8), (20, 10, 8), (30, 90, 32), (120, 10, 8)):
        available = set(range(offset, offset + count))
        keep = available.intersection(TASK_IDS)
        selected.extend(sorted(keep | set(rng.sample(sorted(available - keep), wanted - len(keep)))))
    return selected


def validate_worker_plan(task_ids, worker_order):
    if not worker_order or any(workers not in ALLOWED_WORKERS for workers in worker_order):
        raise ValueError("Worker counts must be1/2/4/8/16/32")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate task jobs would bias comparison")
    if len(task_ids) < max(worker_order):
        raise ValueError(
            "Need at least as many distinct task jobs as workers; use --expanded-panel for32workers"
        )


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_tasks(catalog, task_ids):
    from scripts.lpwm_full.evaluate import catalog_suites

    if sum(catalog_suites(catalog).values()) != 130:
        raise ValueError("Benchmark requires the existing canonical full130 checkpoint")
    if len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(range(130)):
        raise ValueError("Task IDs must be unique members of the full130 catalog")
    return [catalog["tasks"][i] for i in task_ids]


def observation_digest(hasher, value):
    """Hash CPU observations without copying/retaining RGB payloads."""
    import numpy as np

    if isinstance(value, dict):
        for key in sorted(value):
            hasher.update(str(key).encode())
            observation_digest(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        for child in value:
            observation_digest(hasher, child)
    elif isinstance(value, np.ndarray):
        hasher.update(str((value.shape, value.dtype.str)).encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    else:
        hasher.update(repr(value).encode())


class AuditEnv:
    def __init__(self, env):
        self.env = env
        self.actions = []
        self.observations = hashlib.sha256()

    @property
    def init_state_id(self):
        return self.env.init_state_id

    @init_state_id.setter
    def init_state_id(self, value):
        self.env.init_state_id = value

    def reset(self, **kwargs):
        self.actions = []
        self.observations = hashlib.sha256()
        result = self.env.reset(**kwargs)
        observation_digest(self.observations, result[0])
        return result

    def step(self, action):
        self.actions.append(action.copy())
        result = self.env.step(action)
        observation_digest(self.observations, result[0])
        return result


def initialize_worker(settings, ready):
    global _BUNDLE, _SETTINGS
    start = time.perf_counter()
    try:
        import torch

        from scripts.lpwm_ab import evaluate as native
        from scripts.lpwm_full.evaluate import ExactLanguageCache

        _SETTINGS = settings
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        if torch.cuda.device_count() != 1:
            raise ValueError("Benchmark must see only its selected GPU")
        prop = torch.cuda.get_device_properties(0)
        if str(prop.uuid).removeprefix("GPU-") != settings["gpu_uuid"].removeprefix("GPU-"):
            raise ValueError("Wrong GPU; refusing to run")
        native.LanguageCache = ExactLanguageCache
        _BUNDLE = native.load_checkpoint(Path(settings["checkpoint"]), torch.device("cuda"))
        ready.put(
            {
                "status": "ready",
                "pid": os.getpid(),
                "seconds": time.perf_counter() - start,
                "gpu_uuid": str(prop.uuid),
                "model_sha256": _BUNDLE["checkpoint"]["model_sha256"],
                "torch_threads": torch.get_num_threads(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
        )
    except BaseException as exc:
        ready.put({"status": "failed", "pid": os.getpid(), "error_type": type(exc).__name__})
        raise


def evaluate_task(item):
    import numpy as np
    import torch

    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
    from scripts.lpwm_ab import evaluate as native

    bundle, settings = _BUNDLE, _SETTINGS
    suite = _get_suite(item["suite"])
    task = suite.get_task(item["task_id"])
    if task.name != item["name"] or native.normalize_task_description(
        task.language
    ) != native.normalize_task_description(item["language"]):
        raise ValueError("Task identity differs from immutable checkpoint catalog")
    limit = TASK_SUITE_MAX_STEPS[item["suite"]]
    start = time.perf_counter()
    env = LiberoEnv(
        task_suite=suite,
        task_id=item["task_id"],
        task_suite_name=item["suite"],
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
    audit = AuditEnv(env)
    language, _ = bundle["language_cache"].select(task.language, torch.device("cuda"))
    # Slice the ORIGINAL ten-episode plan, never resample a smaller initial-state pool.
    plans = native.episode_plan(len(env._init_states), 10, 42, "validation", item["global_task_id"])
    rows = []
    try:
        for plan in plans[: settings["episodes_per_task"]]:
            row = native.rollout_episode(
                audit,
                bundle["policy"],
                plan,
                bundle["mapping"],
                bundle["state_stats"],
                language,
                torch.device("cuda"),
                limit,
                128,
            )
            actions = np.stack(audit.actions)
            filename = f"task_{item['global_task_id']:03d}_episode_{plan['episode_index']:02d}.npy"
            np.save(Path(settings["variant_output"]) / "actions" / filename, actions, allow_pickle=False)
            row.update(
                global_task_id=item["global_task_id"],
                suite=item["suite"],
                suite_task_id=item["task_id"],
                max_control_steps=limit,
                action_trace=filename,
                action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                observation_sha256=audit.observations.hexdigest(),
            )
            rows.append(row)
    finally:
        env.close()
    result = {
        "task_id": item["global_task_id"],
        "suite": item["suite"],
        "worker_pid": os.getpid(),
        "seconds": time.perf_counter() - start,
        "episodes": rows,
    }
    atomic(Path(settings["variant_output"]) / f"task_{item['global_task_id']:03d}.json", result)
    return result


def run_variant(settings, tasks, workers, output):
    validate_worker_plan([task["global_task_id"] for task in tasks], [workers])
    output.mkdir(exist_ok=False)
    (output / "actions").mkdir()
    settings = dict(settings, variant_output=str(output))
    context = mp.get_context("spawn")
    ready = context.Queue()
    start = time.perf_counter()
    start_unix = time.time()
    pool = context.Pool(workers, initialize_worker, (settings, ready))
    try:
        workers_info = []
        for _ in range(workers):
            try:
                item = ready.get(timeout=240)
            except queue.Empty as exc:
                raise TimeoutError("Worker CUDA/model initialization timed out") from exc
            if item["status"] != "ready":
                raise RuntimeError(f"Worker initialization failed: {item}")
            workers_info.append(item)
        if len({x["model_sha256"] for x in workers_info}) != 1:
            raise ValueError("Workers did not load the same model")
        ready_time = time.perf_counter()
        ready_unix = time.time()
        atomic(output / "workers.json", workers_info)
        rows = []
        for row in pool.imap_unordered(evaluate_task, tasks, chunksize=1):
            rows.append(row)
            atomic(
                output / "status.json",
                {
                    "status": "running",
                    "workers": workers,
                    "completed_tasks": len(rows),
                    "total_tasks": len(tasks),
                    "elapsed_seconds": time.perf_counter() - start,
                },
            )
            print(
                json.dumps(
                    {
                        "variant": output.name,
                        "completed_tasks": len(rows),
                        "total_tasks": len(tasks),
                        "task_id": row["task_id"],
                    }
                ),
                flush=True,
            )
        finished = time.perf_counter()
        finished_unix = time.time()
        pool.close()
        pool.join()
        pool = None
        cleanup_finished_unix = time.time()
        work_by_pid = {
            str(w["pid"]): [row["task_id"] for row in rows if row["worker_pid"] == w["pid"]]
            for w in workers_info
        }
        episodes = [ep for task in sorted(rows, key=lambda x: x["task_id"]) for ep in task["episodes"]]
        result = {
            "status": "completed",
            "formal_result_eligible": False,
            "workers": workers,
            "start_unix": start_unix,
            "ready_unix": ready_unix,
            "finished_unix": finished_unix,
            "cleanup_finished_unix": cleanup_finished_unix,
            "actual_busy_workers": sum(bool(jobs) for jobs in work_by_pid.values()),
            "task_jobs_by_worker": work_by_pid,
            "initializer_seconds": ready_time - start,
            "rollout_seconds": finished - ready_time,
            "end_to_end_seconds": finished - start,
            "with_cleanup_seconds": time.perf_counter() - start,
            "episode_count": len(episodes),
            "total_control_steps": sum(x["control_steps"] for x in episodes),
            "successes": sum(x["success"] for x in episodes),
            "episodes": episodes,
            "task_ids": [x["global_task_id"] for x in tasks],
            "worker_initialization": workers_info,
        }
        atomic(output / "result.json", result)
        atomic(
            output / "status.json",
            {k: v for k, v in result.items() if k not in ["episodes", "worker_initialization"]},
        )
        return result
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        ready.close()


PARITY_KEYS = (
    "seed",
    "init_state_index",
    "episode_index",
    "success",
    "control_steps",
    "terminated",
    "truncated",
    "reached_step_limit",
    "clipped_action_components",
    "executed_action_components",
)


def compare_results(baseline, candidate):
    def keyed(result):
        rows = {(r["global_task_id"], r["episode_index"]): r for r in result["episodes"]}
        if len(rows) != len(result["episodes"]):
            raise ValueError("Duplicate episode identity")
        return rows

    left, right = keyed(baseline), keyed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Different episode coverage")
    outcome, action, observation = [], [], []
    for key, row in left.items():
        other = right[key]
        changed = [field for field in PARITY_KEYS if row[field] != other[field]]
        if changed:
            outcome.append({"episode": list(key), "fields": changed})
        if row["action_sha256"] != other["action_sha256"]:
            action.append(list(key))
        if row["observation_sha256"] != other["observation_sha256"]:
            observation.append(list(key))
    return {
        "episode_count": len(left),
        "outcome_mismatches": outcome,
        "action_trace_mismatches": action,
        "observation_trace_mismatches": observation,
        "bitwise_trajectory_parity": not (outcome or action or observation),
        "outcome_parity": not outcome,
        "end_to_end_speedup": baseline["end_to_end_seconds"] / candidate["end_to_end_seconds"],
        "rollout_speedup": baseline["rollout_seconds"] / candidate["rollout_seconds"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument(
        "--expanded-panel",
        action="store_true",
        help="64task jobs, including original16; suitable for32workers",
    )
    parser.add_argument("--panel-seed", type=int, default=20260924)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--worker-order", type=int, nargs="+", default=[1, 4, 8, 1])
    args = parser.parse_args(argv)
    if args.expanded_panel and args.task_ids is not None:
        parser.error("Choose explicit task IDs or expanded panel, not both")
    args.task_ids = (
        args.task_ids
        if args.task_ids is not None
        else (expanded_task_ids(args.panel_seed) if args.expanded_panel else list(TASK_IDS))
    )
    if not 1 <= args.episodes_per_task <= 10:
        parser.error("Use1..10original episodes")
    try:
        validate_worker_plan(args.task_ids, args.worker_order)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Set the explicit exclusive CUDA UUID before spawning workers")
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((args.checkpoint / "task_catalog.json").read_text())
    tasks = select_tasks(catalog, args.task_ids)
    settings = {
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu_uuid": args.gpu_uuid,
        "episodes_per_task": args.episodes_per_task,
    }
    protocol = dict(
        settings,
        task_ids=list(args.task_ids),
        worker_order=args.worker_order,
        reference_workers=args.worker_order[0],
        panel_seed=args.panel_seed if args.expanded_panel else None,
        panel_tasks=len(tasks),
        start_method="spawn",
        formal_result_eligible=False,
        hard_reset=True,
        full_control_horizons=True,
        original_plan_episodes=10,
        seed_namespace="validation",
        model_sha256=file_hash(args.checkpoint / "model.safetensors"),
    )
    atomic(args.output / "protocol.json", protocol)
    baseline, summaries = None, []
    try:
        for index, workers in enumerate(args.worker_order):
            variant = args.output / f"phase_{index:02d}_w{workers}"
            atomic(args.output / "status.json", {"status": "running", "phase": index, "workers": workers})
            result = run_variant(settings, tasks, workers, variant)
            baseline = result if baseline is None else baseline
            summary = {k: v for k, v in result.items() if k not in ["episodes", "worker_initialization"]}
            summary.update(phase=index, **compare_results(baseline, result))
            summaries.append(summary)
            atomic(
                args.output / "summary.json",
                {"status": "running", "formal_result_eligible": False, "phases": summaries},
            )
            print(json.dumps(summary), flush=True)
        if file_hash(args.checkpoint / "model.safetensors") != protocol["model_sha256"]:
            raise ValueError("Checkpoint changed during benchmark")
        atomic(
            args.output / "summary.json",
            {"status": "completed", "formal_result_eligible": False, "phases": summaries},
        )
        atomic(args.output / "status.json", {"status": "completed", "phase_count": len(summaries)})
    except BaseException as exc:
        atomic(args.output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
