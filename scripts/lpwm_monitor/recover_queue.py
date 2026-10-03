"""One-shot guarded continuation for the known SwanLab request/server-ID mismatch.

Waits for the OLD queue and ALL child processes to exit naturally. Never kills a
process or resumes partial training. Only full30k/six-eval runs whose sole failure
is the known identity predicate can be adopted after full corrected verification.
Frozen trainer/evaluator/model source remains unchanged. Backups precede migration.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

KNOWN_ERROR = "Upload receipt lacks fresh server run identity:"


def recovery_plan(state, verify, alive):
    if state.get("status") != "failed":
        raise ValueError("Only the known failed queue can be recovered")
    first_pair = {"phase1_world_0p03", "phase1_world_0p3"}
    if set(state["runs"]) != first_pair:
        raise ValueError("Recovery is limited to the original first pair")
    summaries = {}
    for name, record in state["runs"].items():
        if alive(record):
            raise ValueError("Child process group still alive")
        if record.get("exit_code") != 0:
            raise ValueError("Trainer did not exit successfully")
        if record["status"] != "failed" or not record.get("error", "").startswith(KNOWN_ERROR):
            raise ValueError("Unrelated or unknown failure; manual inspection required")
        summaries[name] = verify(record["spec"])
    return summaries


def process_alive(pid, suite):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        return str(suite) in cmd and "run_b_sweep.py" in cmd
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("corrected_queue", args.gate)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    while True:
        state = gate.read_json(args.suite / "queue_state.json")
        old_pid = int((args.suite / "queue.pid").read_text())
        gate.atomic_json(
            args.output / "recovery_status.json",
            {
                "state": "waiting_for_original_pair",
                "time": time.time(),
                "original_queue_pid": old_pid,
                "queue_status": state["status"],
            },
        )
        if state["status"] == "completed":
            return
        if not process_alive(old_pid, args.suite):
            break
        time.sleep(60)
    config = state["configuration"]
    opts = SimpleNamespace(
        **{k: Path(config[k]) for k in ("repo", "python", "eval_python", "root", "data", "uv")},
        credential_file=Path("/root/lpwm_ab/.swanlab_api_key"),
        project=config["project"],
        min_free_gib=config["min_free_gib"],
        estimated_run_gib=config["estimated_run_gib"],
        poll_seconds=5,
        resume=True,
        recover_stale=[],
    )
    os.environ["LIBERO_CONFIG_PATH"] = config["libero_config_path"]
    with gate.queue_lock(args.suite, True) as lock:
        manager = gate.SweepQueue(opts, lock)
        summaries = recovery_plan(manager.state, manager.verified, gate.group_alive)
        gate.verify_paired(list(summaries.values()))
        backup = args.output / "queue_before_identity_fix.json"
        with backup.open("x") as f:
            json.dump(manager.state, f, indent=2)
        for name, summary in summaries.items():
            record = manager.state["runs"][name]
            record["previous_verification_error"] = record.pop("error")
            record.update(status="completed", summary=summary, recovered=True)
        manager.event(
            "recovered_sdk_generated_run_identity",
            None,
            backup=str(backup),
            verifier=str(args.gate),
            no_training_restarted=True,
        )
        # Record actual owning PID; no unrelated process was terminated.
        (args.suite / "queue.pid").write_text(str(os.getpid()) + "\n")
        gate.atomic_json(
            args.output / "recovery_status.json",
            {
                "state": "continuing_queue",
                "pid": os.getpid(),
                "time": time.time(),
                "adopted_runs": list(summaries),
            },
        )
        manager.execute()
        gate.atomic_json(args.output / "recovery_status.json", {"state": "completed", "time": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / "recovery_status.json.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "state": "blocked",
                        "time": time.time(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            )
            temporary.replace(output / "recovery_status.json")
        raise
