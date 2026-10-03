"""One fresh process per B checkpoint; unchanged LIBERO evaluation, private online logging.

Production: 10 tasks x 10 completed episodes (not 100 successful episodes).
Use --preflight for a smaller/altered protocol. Retry logging without a simulator:
    uv run python scripts/lpwm_ab/eval_b_sweep.py --upload-only result.json \
        --credential-file /external/swanlab-key --project NAME --run-name NAME

Credentials are a plain-text API key in a file outside the repository. Result JSON
is owned exclusively by evaluate.py; this wrapper only writes result.swanlab.json.
Upload success requires server read-back, not just a return from swanlab.finish().
Queue contract: <output stem>.swanlab.json has status running/complete/failed and
verified_upload (uploaded is a compatibility alias). Accept only exit code 0,
status complete, and verified_upload true. Formal results also require
formal_result_eligible true; a verified preflight is still NOT a formal result.
Requires the SwanLab Settings and Api.run/metrics interfaces (tested with 0.9.4).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_ATTEMPTS = 5
VERIFY_DELAY = 2
SUMMARY_KEYS = ("successes", "num_episodes", "success_rate", "pc_success")


def load_evaluator():
    """Import the sibling only for rollout; upload-only needs neither torch nor LIBERO."""
    spec = importlib.util.spec_from_file_location(
        "lpwm_b_sweep_evaluator", Path(__file__).with_name("evaluate.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, data):
    """Replace only the wrapper-owned sidecar, even after an interrupted attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require(condition, message):
    """Enforce a runtime invariant without relying on Python assertions."""
    if not condition:
        raise ValueError(message)


def integer(value):
    """Exclude booleans from JSON integer counters."""
    return type(value) is int


def validate_result(result, *, preflight=False):
    """Derive denominators from episode rows and reject stale/chunk-based summaries."""
    require(result["status"] == "complete", "Incomplete evaluation")
    require(result["checkpoint"]["variant"] == "B", "Only variant B is permitted")
    protocol = result["protocol"]
    require(protocol["name"] == "lpwm-libero-spatial-rollout-v1", "Unexpected protocol")
    require(protocol["suite"] == "libero_spatial", "Unexpected suite")
    require(protocol["seed_namespace"] in ("validation", "final"), "Invalid seed namespace")
    require(integer(protocol["seed"]), "Invalid seed")
    task_ids = protocol["task_ids"]
    require(isinstance(task_ids, list) and bool(task_ids), "Missing task IDs")
    require(all(integer(task) and 0 <= task < 10 for task in task_ids), "Invalid task IDs")
    require(len(task_ids) == len(set(task_ids)), "Duplicate task IDs")
    count = protocol["episodes_per_task"]
    require(integer(count) and 0 < count <= 10, "Invalid episode count")
    require(integer(protocol["init_state_offset"]) and protocol["init_state_offset"] >= 0, "Invalid offset")
    if not preflight:
        require(sorted(task_ids) == list(range(10)) and count == 10, "Production requires 100 episodes")
        require(protocol["init_state_offset"] == 0, "Production requires the default init-state pool")
        require(protocol["requested_max_steps"] is None, "Production requires the full horizon")
        require(protocol["max_control_steps"] == protocol["local_suite_max_steps"], "Truncated horizon")
    tasks = result["per_task"]
    require([task["task_id"] for task in tasks] == task_ids, "Missing, duplicate, or reordered tasks")
    metrics = {}
    all_rows = []

    def summary(rows, recorded, prefix):
        successes = sum(row["success"] for row in rows)
        expected = dict(
            zip(
                SUMMARY_KEYS,
                (successes, len(rows), successes / len(rows), 100 * successes / len(rows)),
                strict=True,
            )
        )
        for key, value in expected.items():
            actual = recorded[key]
            require(type(actual) in (int, float) and math.isfinite(actual), "Invalid summary")
            if key in ("successes", "num_episodes"):
                require(integer(actual) and actual == value, "Summary count is not an episode count")
            else:
                require(math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9), "Inconsistent success rate")
            metrics[f"{prefix}/{key}"] = value

    for task in tasks:
        rows = task["episodes"]
        require(len(rows) == count, "Missing completed episodes")
        require(all(integer(row["episode_index"]) for row in rows), "Invalid episode index")
        require([row["episode_index"] for row in rows] == list(range(count)), "Duplicate/missing episodes")
        require(all(type(row["success"]) is bool for row in rows), "Invalid episode success")
        require(
            all(integer(row["control_steps"]) and row["control_steps"] > 0 for row in rows), "Empty episode"
        )
        summary(rows, task, f"eval/task_{task['task_id']:02d}")
        all_rows.extend(rows)
    summary(all_rows, result, "eval")
    metrics["eval/preflight"] = int(preflight)
    metrics["eval/formal_result_eligible"] = int(not preflight)
    return metrics


def verify_online(api, run_path, metrics):
    """Fail closed on swallowed SDK errors, dropped metrics, or an offline fallback.

    SwanLab 0.9.x scalar read-back uses index/data; step/value is also accepted.
    Fresh IDs plus exact metric/step checks prevent a previous upload satisfying this.
    """
    for attempt in range(VERIFY_ATTEMPTS):
        try:
            remote = api.run(run_path)
            require(remote.state == "FINISHED", "Remote run not finished")
            payload = remote.metrics(keys=list(metrics), sample=10)
            received = {row["key"]: row["metrics"] for row in payload["list"]}
            for key, value in metrics.items():
                require(
                    any(
                        point.get("step", point.get("index")) == 1
                        and type(point.get("value", point.get("data"))) in (int, float)
                        and math.isclose(
                            point.get("value", point.get("data")), value, rel_tol=1e-6, abs_tol=1e-6
                        )
                        for point in received.get(key, [])
                    ),
                    "Remote metric missing or mismatched",
                )
            return
        except Exception:
            # Never include provider exception strings: they may contain credentials.
            if attempt + 1 < VERIFY_ATTEMPTS:
                time.sleep(VERIFY_DELAY)
    raise RuntimeError("Online upload could not be verified")


def parse_args(argv=None):
    """Parse one synchronous evaluation or one upload-only retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload-only", type=Path, metavar="RESULT_JSON")
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-namespace", choices=("validation", "final"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--init-state-offset", type=int, default=0)
    args = parser.parse_args(argv)
    if args.upload_only:
        if args.output and args.output.resolve() != args.upload_only.resolve():
            parser.error("--output must match --upload-only when both are supplied")
        args.output = args.upload_only
    elif args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required for rollout")
    if not 1 <= args.episodes_per_task <= 10:
        parser.error("--episodes-per-task must be between 1 and 10")
    if args.init_state_offset < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Invalid offset or max-steps")
    if args.task_ids is not None and (
        len(set(args.task_ids)) != len(args.task_ids) or any(task not in range(10) for task in args.task_ids)
    ):
        parser.error("--task-ids must be unique Spatial IDs in [0, 9]")
    if (
        not args.preflight
        and not args.upload_only
        and (
            args.episodes_per_task != 10
            or (args.task_ids is not None and sorted(args.task_ids) != list(range(10)))
            or args.max_steps is not None
            or args.init_state_offset != 0
        )
    ):
        parser.error("Altered/smaller protocols require --preflight")
    if args.credential_file.resolve().is_relative_to(REPO_ROOT):
        parser.error("Credential file must be outside the repository")
    if args.output.resolve() == args.credential_file.resolve():
        parser.error("Result and credential paths must differ")
    return args


def main(argv=None):
    """Run one job, preserving evaluator output on every online failure."""
    args = parse_args(argv)
    metadata_path = args.output.with_suffix(".swanlab.json")
    # No wrapper path may overwrite a credential or the authoritative result.
    if metadata_path.resolve() in (args.output.resolve(), args.credential_file.resolve()):
        print("Unsafe metadata path", file=sys.stderr)
        return 1
    if not args.upload_only and args.output.exists():
        print("Result exists; use --upload-only to retry without rerollout", file=sys.stderr)
        return 1
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # In particular, do NOT remap CUDA_VISIBLE_DEVICES or MUJOCO_EGL_DEVICE_ID.
    metadata = {
        "schema_version": 1,
        "status": "running",
        "uploaded": False,
        "verified_upload": False,
        "evaluation_status": "existing" if args.upload_only else "not_started",
        "upload_only": bool(args.upload_only),
        "preflight": args.preflight,
        "formal_result_eligible": not args.preflight,
        "protocol_role": "preflight" if args.preflight else args.seed_namespace,
        "project": args.project,
        "run_name": args.run_name,
        "mode": "online",
        "public": False,
        "attempt_id": uuid.uuid4().hex,
        "result": str(args.output),
    }
    swanlab = run = None
    finish_attempted = False
    phase = "prepare"
    try:
        if args.upload_only and metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get("preflight") and not args.preflight:
                print("Retry a preflight with --preflight; metadata retained", file=sys.stderr)
                return 1
        atomic_json(metadata_path, metadata)
        if args.upload_only:
            result = json.loads(args.output.read_text())
            metrics = validate_result(result, preflight=args.preflight)
            metadata["evaluation_status"] = "complete"
            protocol_config = {
                field: result["protocol"][field]
                for field in (
                    "seed",
                    "seed_namespace",
                    "task_ids",
                    "episodes_per_task",
                    "init_state_offset",
                    "requested_max_steps",
                )
            }
            experiment = result["checkpoint"]
        else:
            # Catch wrong-variant jobs before authenticating or allocating a simulator.
            experiment = json.loads((args.checkpoint / "experiment.json").read_text())
            require(experiment["variant"] == "B", "Only variant B is permitted")
            protocol_config = {
                field: getattr(args, field)
                for field in (
                    "seed",
                    "seed_namespace",
                    "episodes_per_task",
                    "init_state_offset",
                )
            }
            protocol_config.update(
                task_ids=args.task_ids or list(range(10)), requested_max_steps=args.max_steps
            )
        metadata["protocol_role"] = "preflight" if args.preflight else protocol_config["seed_namespace"]
        metadata["protocol"] = protocol_config
        phase = "authentication"
        import swanlab

        require(not swanlab.has_run(), "Invoke the wrapper in a fresh process")
        key = args.credential_file.read_text().strip()
        require(bool(key) and not any(char.isspace() for char in key), "Invalid credential file")
        # Do not expose provider authentication exception text or console output.
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            authenticated = swanlab.login(api_key=key, save=False, relogin=True)
        del key
        require(authenticated is not False, "Authentication failed")
        phase = "initialization"
        api = swanlab.Api()
        # An existing public project must never silently defeat public=False.
        project_path = f"{api.username}/{args.project}"
        settings = swanlab.Settings(
            interactive=False,
            terminal={"proxy_type": "none"},
            probe=dict.fromkeys(
                ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor"), False
            ),
        )
        run = swanlab.init(
            project=args.project,
            workspace=api.username,
            name=args.run_name,
            mode="online",
            public=False,
            resume="never",
            log_dir=str(args.output.parent / f"{args.output.stem}_swanlog"),
            settings=settings,
            config={
                "variant": "B",
                "preflight": args.preflight,
                "upload_only": bool(args.upload_only),
                "formal_result_eligible": not args.preflight,
                "protocol": protocol_config,
                "checkpoint_step": experiment.get("step"),
            },
        )
        require(run is not None and run.mode == "online", "SwanLab is not online")
        require(isinstance(run.id, str) and bool(run.id), "SwanLab did not allocate a new run ID")
        project = api.project(project_path)
        require(
            bool(project.project_id) and project.visibility == "PRIVATE", "Project is not private/visible"
        )
        metadata.update(id=run.id, url=run.url)
        atomic_json(metadata_path, metadata)
        phase = "start_logging"
        require(swanlab.log({"eval/started": 1}, step=0) is not False, "Start log failed")
        if not args.upload_only:
            phase = "rollout"
            metadata["evaluation_status"] = "running"
            atomic_json(metadata_path, metadata)
            returned = load_evaluator().run_evaluation(args)
            phase = "result_validation"
            result = json.loads(args.output.read_text())
            require(result == returned, "Persisted result differs from evaluator return")
            metrics = validate_result(result, preflight=args.preflight)
            protocol = result["protocol"]
            for field in ("seed", "seed_namespace", "episodes_per_task", "init_state_offset"):
                require(protocol[field] == getattr(args, field), "Result protocol differs from request")
            require(protocol["task_ids"] == (args.task_ids or list(range(10))), "Unexpected result tasks")
            require(protocol["requested_max_steps"] == args.max_steps, "Unexpected result horizon")
            metadata["evaluation_status"] = "complete"
        metadata["result_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        atomic_json(metadata_path, metadata)
        phase = "result_logging"
        require(swanlab.log(metrics, step=1) is not False, "Result log failed")
        phase = "finish"
        finish_attempted = True
        require(swanlab.finish() is not False, "Finish failed")
        phase = "verification"
        verify_online(api, f"{project_path}/{run.id}", metrics)
        metadata.update(status="complete", uploaded=True, verified_upload=True)
        atomic_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "verified_upload": True,
                    "metadata": str(metadata_path),
                    **{k: result[k] for k in SUMMARY_KEYS},
                }
            )
        )
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if run is not None and not finish_attempted:
            try:
                swanlab.finish(state="crashed", error=f"Evaluation wrapper failed during {phase}")
            except (Exception, KeyboardInterrupt):
                metadata["cleanup_failed"] = True
        if metadata["evaluation_status"] == "running":
            metadata["evaluation_status"] = "failed"
        metadata.update(
            status="failed",
            uploaded=False,
            verified_upload=False,
            failure_stage=phase,
            error_type=type(error).__name__,
        )
        try:
            atomic_json(metadata_path, metadata)
        except OSError:
            print("Could not write SwanLab failure metadata", file=sys.stderr)
        # Raw provider/evaluator exception messages can contain secrets; never persist them.
        print(
            f"Evaluation wrapper failed during {phase}; result retained if written; upload NOT confirmed",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
