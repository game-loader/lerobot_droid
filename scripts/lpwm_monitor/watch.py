"""Read-only experiment observer; never launches/stops training or changes frozen source.

Writes ONLY its own output directory. Existing run_b_sweep remains the authority
for advancing stages; a failed/stale run is an alert, not permission to restart it.
"""

import argparse
import csv
import fcntl
import importlib.util
import io
import json
import shutil
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def signature(paths):
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(paths)]


def recent_metrics(path):
    if not path.exists():
        return {}
    # Bounded read of a growing log; tolerate a partial last line.
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 131072))
        lines = f.read().decode(errors="replace").splitlines()
    latest = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for key in ("train/fm_loss", "validation/fm_loss", "gradient/world_to_fm_ratio"):
            if key in row:
                latest[key] = row
    return latest


def queue_alive(suite):
    try:
        pid = int((suite / "queue.pid").read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        return ("run_b_sweep.py" in cmd or "recover_queue.py" in cmd or "power_resume.py" in cmd) and str(
            suite
        ) in cmd
    except (OSError, ValueError):
        return False


def collect(suite, gate, cache, now=None):
    now = time.time() if now is None else now
    queue = read(suite / "queue_state.json")
    alerts, runs, evaluations = [], {}, []
    free = shutil.disk_usage(suite).free
    if free < 3 * 2**30:
        alerts.append("disk_free_below_3GiB")
    alive = queue_alive(suite)
    if queue["status"] not in ("completed", "failed", "interrupted") and not alive:
        alerts.append("queue_process_missing")
    recovery_path = suite / "monitor/recovery_status.json"
    if recovery_path.exists() and read(recovery_path).get("state") == "blocked":
        alerts.append("guarded_recovery_blocked")
    if queue["status"] in ("failed", "interrupted"):
        alerts.append("queue_" + queue["status"])
    for run_id, record in queue["runs"].items():
        root = suite / "runs" / run_id
        state_path = root / "status.json"
        if not state_path.exists():
            if now - record.get("started_unix", now) > 300:
                alerts.append(run_id + ":missing_status")
            continue
        state = read(state_path)
        age = now - state_path.stat().st_mtime
        limit = 7200 if state["status"] == "evaluating" else 1800
        if state["status"] in ("running", "evaluating") and age > limit:
            alerts.append(run_id + ":stale_" + state["status"])
        if state["status"] == "failed":
            alerts.append(run_id + ":trainer_failed")
        runs[run_id] = {
            "status": state,
            "status_age_seconds": age,
            "spec": record["spec"],
            "metrics": recent_metrics(root / "metrics.jsonl"),
        }
        if (root / "swanlab_run.json").exists():
            runs[run_id]["swanlab"] = read(root / "swanlab_run.json")
        experiment = read(root / "experiment.json") if (root / "experiment.json").exists() else None
        for result_path in sorted((root / "eval").glob("step_*.json")):
            if result_path.name.endswith(".swanlab.json"):
                continue
            step = int(result_path.stem.split("_")[1])
            receipt = result_path.with_suffix(".swanlab.json")
            # Evaluator writes the result before upload completes; do not call that corruption.
            if not receipt.exists() or read(receipt).get("status") != "complete":
                if receipt.exists() and read(receipt).get("status") == "failed":
                    alerts.append(f"{run_id}:{step}:upload_failed")
                continue
            cp = root / "checkpoints" / f"step_{step:06d}"
            paths = [result_path, receipt, *cp.glob("*")]
            sig = signature([p for p in paths if p.is_file()])
            key = f"{run_id}:{step}"
            try:
                if key not in cache or cache[key]["signature"] != sig:
                    verified = gate.verify_evaluation(result_path, root, step, experiment)
                    result = read(result_path)
                    cache[key] = {"signature": sig, "verified": verified, "result": result}
                entry = cache[key]
                evaluations.append(
                    {
                        "run": run_id,
                        "world_weight": record["spec"]["world_weight"],
                        "reconstruction_weight": record["spec"]["reconstruction_weight"],
                        "dynamics_weight": record["spec"]["dynamics_weight"],
                        "step": step,
                        "successes": entry["result"]["successes"],
                        "episodes": entry["result"]["num_episodes"],
                        "success_rate": entry["result"]["success_rate"],
                        "integrity_and_upload_verified": True,
                        "per_task": [
                            {
                                "task_id": t["task_id"],
                                "successes": t["successes"],
                                "episodes": t["num_episodes"],
                            }
                            for t in entry["result"]["per_task"]
                        ],
                    }
                )
            except Exception as error:
                alerts.append(f"{run_id}:{step}:verification_failed:{type(error).__name__}:{error}")
    return {
        "captured_unix": now,
        "suite": str(suite),
        "queue_status": queue["status"],
        "queue_process_alive": alive,
        "disk_free_gib": free / 2**30,
        "runs": runs,
        "evaluations": evaluations,
        "alerts": alerts,
        "next_phase_authority": "existing run_b_sweep.py queue; observer never mutates training",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=120)
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--gate", type=Path, help="Reviewed compatible queue verifier outside frozen training source"
    )
    args = p.parse_args()
    if args.interval < 10:
        p.error("interval must be >=10 seconds")
    args.suite = args.suite.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "observer.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = args.gate or args.suite / "repo/scripts/lpwm_ab/run_b_sweep.py"
    spec = importlib.util.spec_from_file_location("frozen_sweep_gate", source)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    cache, previous_alerts = {}, None
    while True:
        try:
            snapshot = collect(args.suite, gate, cache)
            atomic(args.output / "health.json", snapshot)
            atomic(args.output / "verified_results.json", {k: v["result"] for k, v in cache.items()})
            stream = io.StringIO()
            fields = [
                "run",
                "world_weight",
                "reconstruction_weight",
                "dynamics_weight",
                "step",
                "successes",
                "episodes",
                "success_rate",
                "integrity_and_upload_verified",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(snapshot["evaluations"])
            temp = args.output / "success_curve.csv.tmp"
            temp.write_text(stream.getvalue())
            temp.replace(args.output / "success_curve.csv")
            brief = {
                "captured_unix": snapshot["captured_unix"],
                "queue_status": snapshot["queue_status"],
                "steps": {k: v["status"] for k, v in snapshot["runs"].items()},
                "verified_evaluations": len(snapshot["evaluations"]),
                "alerts": snapshot["alerts"],
            }
            with (args.output / "history.jsonl").open("a") as f:
                f.write(json.dumps(brief) + "\n")
            print(json.dumps(brief), flush=True)
            if snapshot["alerts"] != previous_alerts:
                with (args.output / "alerts.jsonl").open("a") as f:
                    f.write(json.dumps(brief) + "\n")
                previous_alerts = snapshot["alerts"]
            if args.once or snapshot["queue_status"] == "completed":
                break
        except Exception as error:
            atomic(
                args.output / "observer_error.json",
                {"time": time.time(), "error_type": type(error).__name__, "error": str(error)},
            )
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
