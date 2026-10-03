"""Common40 or all130 LIBERO tasks, exact language, real simulator success only."""

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path

import numpy as np
import torch

from scripts.lpwm_ab import evaluate as native

SUITES = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_90": 90, "libero_10": 10}
OriginalLanguageCache = native.LanguageCache


COMMON_SUITES = {name: count for name, count in SUITES.items() if name != "libero_90"}


def suites_for_count(task_count=130):
    if task_count == 40:
        return dict(COMMON_SUITES)
    if task_count == 130:
        return dict(SUITES)
    raise ValueError("Only common40 or all130 task coverage is supported")


def catalog_suites(catalog):
    """Validate the canonical suite/local-ID/global-ID mapping, not just its length."""
    tasks = catalog["tasks"]
    suites = suites_for_count(len(tasks))
    target = [(name, i) for name, count in suites.items() for i in range(count)]
    if [(t["suite"], t["task_id"]) for t in tasks] != target or [t["global_task_id"] for t in tasks] != list(
        range(len(tasks))
    ):
        raise ValueError("Catalog suite/task identity mismatch or non-contiguous global IDs")
    return suites


def checkpoint_catalog(checkpoint, preflight=False):
    path = checkpoint / "task_catalog.json"
    if not path.is_file() and preflight:
        path = checkpoint.parent.parent / "task_catalog.json"
    catalog = json.loads(path.read_text())
    return catalog, catalog_suites(catalog)


def protocol_suites(protocol):
    # Older all130 result fixtures only carried the preflight marker.
    declared = protocol.get("suites", SUITES)
    if not isinstance(declared, dict) or any(type(n) is not int for n in declared.values()):
        raise ValueError("Invalid protocol suites")
    suites = suites_for_count(sum(declared.values()))
    if declared != suites:
        raise ValueError("Wrong protocol suite coverage")
    return suites


class ExactLanguageCache(OriginalLanguageCache):
    """Duplicate instruction texts across scenes may share only IDENTICAL frozen tokens."""

    def __init__(self, metadata, embeddings, masks, language_dim):
        seen = {}
        rows = []
        ids = []
        for row, task_id in enumerate(metadata["language"]["task_ids"]):
            text = native.normalize_task_description(metadata["tasks"][str(task_id)])
            if text in seen:
                old = seen[text]
                if not np.array_equal(embeddings[row], embeddings[old]) or not np.array_equal(
                    masks[row], masks[old]
                ):
                    raise ValueError("Conflicting embeddings for duplicate exact language")
            else:
                seen[text] = row
                rows.append(row)
                ids.append(task_id)
        reduced = {**metadata, "language": {**metadata["language"], "task_ids": ids}}
        super().__init__(reduced, np.asarray(embeddings)[rows], np.asarray(masks)[rows], language_dim)


def validate_result(result, preflight=False):
    protocol = result["protocol"]
    suites = protocol_suites(protocol)
    scope_count = sum(suites.values())
    expected = len(suites) if preflight else scope_count
    count = 1 if preflight else 10
    if protocol.get("preflight") is not preflight:
        raise ValueError("Wrong preflight marker")
    required = {
        "task_count": expected,
        "episodes_per_task": count,
        "scope_task_count": scope_count,
        "formal_result_eligible": not preflight,
        "actual_preflight_max_steps": 2 if preflight else None,
    }
    for key, value in required.items():
        if (
            (scope_count == 40 and key in ("task_count", "episodes_per_task")) or key in protocol
        ) and protocol.get(key) != value:
            raise ValueError(f"Wrong protocol {key}")
    if result.get("status") != "complete" or result["num_episodes"] != expected * count:
        raise ValueError("Incomplete all-suite episode denominator")
    tasks = result["per_task"]
    if len(tasks) != expected:
        raise ValueError("Missing fulltask coverage")
    names = [(x["suite"], x["suite_task_id"]) for x in tasks]
    target = (
        [(name, 0) for name in suites]
        if preflight
        else [(name, i) for name, n in suites.items() for i in range(n)]
    )
    if names != target:
        raise ValueError("Fullsuite/task identity mismatch")
    global_ids = {
        (name, i): gid
        for gid, (name, i) in enumerate((name, i) for name, n in suites.items() for i in range(n))
    }
    total = 0
    for task in tasks:
        if (scope_count == 40 or "task_id" in task) and task.get("task_id") != global_ids[
            task["suite"], task["suite_task_id"]
        ]:
            raise ValueError("Global task identity mismatch")
        episodes = task["episodes"]
        if len(episodes) != count or [x["episode_index"] for x in episodes] != list(range(count)):
            raise ValueError("Incomplete or duplicate episode rows")
        if any(
            type(x["success"]) is not bool
            or type(x["control_steps"]) is not int
            or x["control_steps"] < 1
            or (preflight and x["control_steps"] > 2)
            for x in episodes
        ):
            raise ValueError("Invalid simulator outcomes")
        successes = sum(x["success"] for x in episodes)
        if task["successes"] != successes or not np.isclose(task["success_rate"], successes / count):
            raise ValueError("Per-task count mismatch")
        total += successes
    if total != result["successes"] or not np.isclose(total / (expected * count), result["success_rate"]):
        raise ValueError("Aggregate success mismatch")
    if set(result["per_suite"]) != set(suites):
        raise ValueError("Missing suite summaries")
    suite_rates = []
    for name in suites:
        selected = [t for t in tasks if t["suite"] == name]
        successes = sum(t["successes"] for t in selected)
        episodes = len(selected) * count
        actual = result["per_suite"][name]
        if (
            actual["num_episodes"] != episodes
            or actual["successes"] != successes
            or not np.isclose(actual["success_rate"], successes / episodes)
        ):
            raise ValueError("Suite denominator/count mismatch")
        suite_rates.append(successes / episodes)
    if not np.isclose(result["suite_macro_success_rate"], np.mean(suite_rates)):
        raise ValueError("Wrong suite macro success")
    return {
        "eval/successes": total,
        "eval/num_episodes": expected * count,
        "eval/success_rate": total / (expected * count),
        "eval/pc_success": 100 * total / (expected * count),
        "eval/suite_macro_success_rate": result["suite_macro_success_rate"],
        **{f"eval/{t['suite']}/task_{t['suite_task_id']:02d}/success_rate": t["success_rate"] for t in tasks},
        **{f"eval/{name}/success_rate": v["success_rate"] for name, v in result["per_suite"].items()},
    }


_WORKER_BUNDLE = None
_WORKER_OPTIONS = None


def evaluate_task(item, bundle, suites, preflight, namespace, device, video_dir=None):
    """One process owns all ordered episodes of a task, preserving reset history."""
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv

    policy = bundle["policy"]
    name = item["suite"]
    tid = item["task_id"]
    task = suites[name].get_task(tid)
    max_steps = 2 if preflight else TASK_SUITE_MAX_STEPS[name]
    env = LiberoEnv(
        task_suite=suites[name],
        task_id=tid,
        task_suite_name=name,
        episode_length=max_steps,
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
    language, language_info = bundle["language_cache"].select(task.language, device)
    rows = []
    try:
        plan = native.episode_plan(
            len(env._init_states),
            1 if preflight else 10,
            42,
            namespace,
            item["global_task_id"],
        )
        for ep in plan:
            writer = None
            video_path = None
            try:
                if video_dir is not None and ep["episode_index"] == 0:
                    video_path = Path(video_dir) / f"task_{item['global_task_id']:03d}_episode_000.mp4"
                    writer = native.OptionalVideoWriter(video_path)
                video_kwargs = {"video_writer": writer} if writer is not None else {}
                row = native.rollout_episode(
                    env,
                    policy,
                    ep,
                    bundle["mapping"],
                    bundle["state_stats"],
                    language,
                    device,
                    max_steps,
                    128,
                    **video_kwargs,
                )
            finally:
                if writer is not None:
                    writer.close()
            if writer is not None:
                if writer.error is None:
                    row["video"] = str(video_path)
                else:
                    row["video_error"] = writer.error
            rows.append(row)
    finally:
        env.close()
    summary = {
        **native.summarize_episodes(rows),
        "task_id": item["global_task_id"],
        "suite": name,
        "suite_task_id": tid,
        "task_name": task.name,
        "language": task.language,
        "language_cache": language_info,
        "episodes": rows,
        "max_control_steps": max_steps,
    }
    summary["worker_pid"] = os.getpid()
    return summary


def initialize_eval_worker(checkpoint, device, preflight, namespace, ready, video_dir=None):
    global _WORKER_BUNDLE, _WORKER_OPTIONS
    try:
        torch.set_num_threads(4)
        if device == "cuda" and torch.cuda.device_count() != 1:
            raise ValueError("Parallel evaluation must see exactly one selected GPU")
        native.LanguageCache = ExactLanguageCache
        _WORKER_BUNDLE = native.load_checkpoint(Path(checkpoint), torch.device(device))
        _WORKER_OPTIONS = (preflight, namespace, torch.device(device), video_dir)
        ready.put(
            {
                "status": "ready",
                "pid": os.getpid(),
                "model_sha256": _WORKER_BUNDLE["checkpoint"]["model_sha256"],
            }
        )
    except BaseException as exc:
        ready.put({"status": "failed", "pid": os.getpid(), "error_type": type(exc).__name__})


def evaluate_worker_task(item):
    from lerobot.envs.libero import _get_suite

    preflight, namespace, device, video_dir = _WORKER_OPTIONS
    return evaluate_task(
        item,
        _WORKER_BUNDLE,
        {item["suite"]: _get_suite(item["suite"])},
        preflight,
        namespace,
        device,
        video_dir,
    )


def task_results(args, selected, bundle, suites, device):
    workers = getattr(args, "workers", 1)
    video_dir = args.output.parent / f"{args.output.stem}_videos" if getattr(args, "video", False) else None
    if workers == 1:
        for item in selected:
            yield evaluate_task(item, bundle, suites, args.preflight, args.seed_namespace, device, video_dir)
        return
    if workers != 8:
        raise ValueError("Formal evaluator supports only serial or authorized task8")
    context = mp.get_context("spawn")
    ready = context.Queue()
    pool = context.Pool(
        workers,
        initialize_eval_worker,
        (str(args.checkpoint), args.device, args.preflight, args.seed_namespace, ready, video_dir),
    )
    try:
        records = []
        for _ in range(workers):
            try:
                record = ready.get(timeout=240)
            except queue.Empty as exc:
                raise TimeoutError("Parallel evaluator initialization timed out") from exc
            if record["status"] != "ready":
                raise RuntimeError(f"Evaluator worker failed: {record}")
            records.append(record)
        if any(row["model_sha256"] != bundle["checkpoint"]["model_sha256"] for row in records):
            raise ValueError("Parallel evaluators loaded different checkpoints")
        yield from pool.imap_unordered(evaluate_worker_task, selected, chunksize=1)
        pool.close()
        pool.join()
        pool = None
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        ready.close()


def verify_online_resilient(api, run_path, metrics, attempts=24, delay=5, chunk_size=40):
    """Bounded delayed readback, verify EVERY metric; never log provider secrets."""
    if attempts < 1 or chunk_size < 1 or delay < 0:
        raise ValueError("Invalid upload verification retry settings")
    last_error = None
    for attempt in range(attempts):
        try:
            remote = api.run(run_path)
            if remote.state != "FINISHED":
                raise RuntimeError("Remote run not finished")
            keys = list(metrics)
            received = {}
            for offset in range(0, len(keys), chunk_size):
                payload = remote.metrics(keys=keys[offset : offset + chunk_size], sample=10)
                received.update({row["key"]: row["metrics"] for row in payload["list"]})
            for key, value in metrics.items():
                if not any(
                    point.get("step", point.get("index")) == 1
                    and type(point.get("value", point.get("data"))) in (int, float)
                    and math.isclose(point.get("value", point.get("data")), value, rel_tol=1e-6, abs_tol=1e-6)
                    for point in received.get(key, [])
                ):
                    raise ValueError("Remote metric missing or mismatched")
            return {"attempts": attempt + 1, "metrics_verified": len(metrics)}
        except Exception as exc:
            last_error = type(exc).__name__
            print(json.dumps({"telemetry_verify_attempt": attempt + 1, "error_type": last_error}), flush=True)
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError(f"Online upload not verified after {attempts} attempts; last error type: {last_error}")


def run_evaluation(args):
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/root/lpwm_ab/libero_config")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite

    native.LanguageCache = ExactLanguageCache
    device = torch.device(args.device)
    load_device = torch.device("cpu") if getattr(args, "workers", 1) > 1 else device
    bundle = native.load_checkpoint(args.checkpoint, load_device)
    catalog, suite_counts = checkpoint_catalog(args.checkpoint, args.preflight)
    suites = {name: _get_suite(name) for name in suite_counts}
    for item in catalog["tasks"]:
        task = suites[item["suite"]].get_task(item["task_id"])
        if task.name != item["name"] or native.normalize_task_description(
            task.language
        ) != native.normalize_task_description(item["language"]):
            raise ValueError("Installed benchmark task changed")
        bundle["language_cache"].select(task.language, load_device)
    selected = [t for t in catalog["tasks"] if not args.preflight or t["task_id"] == 0]
    allrows = []
    tasks = []
    per_suite = {}
    start = time.time()
    for summary in task_results(args, selected, bundle, suites, device):
        name = summary["suite"]
        tid = summary["suite_task_id"]
        rows = summary["episodes"]
        tasks.append(summary)
        tasks.sort(key=lambda task: task["task_id"])
        allrows += rows
        suite_rows = [row for t in tasks if t["suite"] == name for row in t["episodes"]]
        per_suite[name] = native.summarize_episodes(suite_rows)
        native.atomic_json(
            args.output.with_suffix(".partial.json"),
            {
                "status": "running",
                "completed_tasks": len(tasks),
                "expected_tasks": len(selected),
                "per_task": tasks,
            },
        )
        print(
            json.dumps(
                {
                    "completed_tasks": len(tasks),
                    "suite": name,
                    "task": tid,
                    "successes": summary["successes"],
                    "episodes": len(rows),
                }
            ),
            flush=True,
        )
    protocol = {
        "parallel_workers": getattr(args, "workers", 1),
        "parallel_unit": "task",
        "within_task_episode_order": "serial original plan",
        "worker_start_method": "spawn" if getattr(args, "workers", 1) > 1 else None,
        "actual_busy_workers": len({task["worker_pid"] for task in tasks}),
        "name": f"lpwm-libero-{'common40' if len(catalog['tasks']) == 40 else 'all130'}-rollout-v1",
        "suites": suite_counts,
        "scope_task_count": len(catalog["tasks"]),
        "formal_result_eligible": not args.preflight,
        "seed": 42,
        "seed_namespace": args.seed_namespace,
        "preflight": args.preflight,
        "episodes_per_task": 1 if args.preflight else 10,
        "task_count": len(selected),
        "control_freq_hz": 20,
        "suite_max_steps": {name: TASK_SUITE_MAX_STEPS[name] for name in suite_counts},
        "actual_preflight_max_steps": 2 if args.preflight else None,
        "state_pool": "validation firsthalf;final secondhalf of seeded no-replacement initstates; task seed uses globalcatalogid",
        "image_preprocessing": "raw OpenGL128 RGB rotate180 ONCE;PILbilinear128;CHWfloat/255",
        "state_preprocessing": "eef_pos3+xyzwquat_to_axisangle3+gripperqpos2; savedtrainmean/std",
        "action_execution": "native relativeOSC_POSE7,clip[-1,1],no gripperinversion",
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "flow_inference_steps": 10,
        "language": "exact normalized text; duplicatedtext tokens must be byteidentical",
        "training_includes_all_suites": "NOT a held-out-task LIBERO90-to10 generalization experiment",
    }
    if getattr(args, "video", False):
        protocol["video"] = True
        protocol["video_policy"] = "first episode/task; all outcomes retained; best effort errors per episode"
        protocol["video_fps"] = 20
        protocol["video_camera"] = "agentview_image (policy input), 128x128"
    result = {
        "status": "complete",
        "checkpoint": bundle["checkpoint"],
        "protocol": protocol,
        "protocol_sha256": native.hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        **native.summarize_episodes(allrows),
        "per_suite": per_suite,
        "suite_macro_success_rate": float(np.mean([v["success_rate"] for v in per_suite.values()])),
        "per_task": tasks,
        "duration_seconds": time.time() - start,
    }
    validate_result(result, args.preflight)
    native.atomic_json(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, choices=(1, 8), default=1)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--video", action="store_true", help="Record the first episode of every task as MP4")
    p.add_argument("--seed-namespace", choices=("validation", "final"), default="validation")
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, suite_counts = checkpoint_catalog(args.checkpoint, args.preflight)
    import swanlab

    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.output.with_suffix(".swanlab.json")
    if not swanlab.login(api_key=args.credential_file.read_text().strip(), save=False):
        raise RuntimeError("SwanLab authentication failed")
    run = swanlab.init(
        project=args.project,
        name=args.run_name,
        mode="online",
        public=False,
        log_dir=str(args.output.parent / (args.output.stem + "_swanlog")),
        config={
            "preflight": args.preflight,
            "task_count": len(suite_counts) if args.preflight else sum(suite_counts.values()),
            "suites": suite_counts,
            "formal_result_eligible": not args.preflight,
            "episodes_per_task": 1 if args.preflight else 10,
            "namespace": args.seed_namespace,
            "video": args.video,
            "parallel_workers": args.workers,
            "parallel_unit": "task",
        },
    )
    if run.mode != "online":
        raise RuntimeError("Online logging mandatory")
    api = swanlab.Api()
    project_path = f"{api.username}/{args.project}"
    if api.project(project_path).visibility != "PRIVATE":
        raise RuntimeError("Project must beprivate")
    native.atomic_json(receipt, {"status": "running", "id": run.id, "url": run.url})
    try:
        result = run_evaluation(args)
        metrics = validate_result(result, args.preflight)
        swanlab.log(metrics, step=1)
        if swanlab.finish() is False:
            raise RuntimeError("Upload failed")
        verification = verify_online_resilient(api, f"{project_path}/{run.id}", metrics)
        native.atomic_json(
            receipt,
            {
                "status": "complete",
                "id": run.id,
                "url": run.url,
                "verified_upload": True,
                "verification": verification,
                "uploaded": True,
                "formal_result_eligible": not args.preflight,
                "result_sha256": native.file_sha256(args.output),
                "protocol_role": args.seed_namespace,
            },
        )
    except BaseException:
        if swanlab.has_run():
            swanlab.finish(state="crashed")
        raise


if __name__ == "__main__":
    main()
