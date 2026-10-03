"""Own the old queue lock, finish first-pair checkpoints after power loss, then continue.

No partial run is counted completed. Originals and interruption logs are retained.
Only validated25000->30000 recovery for the known initial pair is supported here.
"""

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    tools = suite / "monitor/tools"
    out = suite / "power_recovery_20260919"
    out.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("fixed_gate", tools / "run_b_sweep_fixed.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    state = gate.read_json(suite / "queue_state.json")
    c = state["configuration"]
    opts = SimpleNamespace(
        **{k: Path(c[k]) for k in ("repo", "python", "eval_python", "root", "data", "uv")},
        credential_file=Path("/root/lpwm_ab/.swanlab_api_key"),
        project=c["project"],
        min_free_gib=c["min_free_gib"],
        estimated_run_gib=c["estimated_run_gib"],
        poll_seconds=5,
        resume=True,
        recover_stale=[],
    )
    os.environ["LIBERO_CONFIG_PATH"] = c["libero_config_path"]
    with gate.queue_lock(suite, True) as lock:
        manager = gate.SweepQueue(opts, lock)
        pair = gate.phase1_rounds()[0]
        assert set(manager.state["runs"]) == {r["id"] for r in pair}
        for run in pair:
            record = manager.state["runs"][run["id"]]
            assert not gate.group_alive(record), "Live process group; no duplicate launch"
            path = suite / "runs" / run["id"]
            experiment = gate.read_json(path / "experiment.json")
            assert gate.read_json(path / "status.json")["step"] < 30000
            assert not (path / "checkpoints/step_030000").exists()
            # Original complete25k checkpoint, all prior evaluations & uploader gate.
            for step in (5000, 10000, 15000, 20000, 25000):
                gate.verify_evaluation(path / "eval" / f"step_{step:06d}.json", path, step, experiment)
            preflight = (out / f"preflight-gpu{run['gpu']}.log").read_text()
            assert "RESUME_PREFLIGHT_OK" in preflight and "Traceback" not in preflight
        with (out / "queue_before_power_recovery.json").open("x") as f:
            json.dump(manager.state, f, indent=2)
        manager.state.update(status="running", power_recovery="restoring_first_pair_from25000")
        manager.event("power_loss_recovery_authorized", None, checkpoint_step=25000, logs_preserved=True)
        (suite / "queue.pid").write_text(str(os.getpid()) + "\n")
        children = []
        try:
            for run in pair:
                command = [
                    str(opts.python),
                    "-u",
                    str(tools / "resume_train.py"),
                    "--suite",
                    str(suite),
                    "--run-id",
                    run["id"],
                ]
                env = gate.child_environment(opts, run)
                with (out / f"{run['id']}.log").open("x") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=opts.repo,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock,),
                    )
                children.append((child, run))
                record = manager.state["runs"][run["id"]]
                record["previous_power_loss_process"] = {
                    k: record.get(k) for k in ("pid", "pgid", "boot_id", "hostname")
                }
                record.update(
                    status="running",
                    pid=child.pid,
                    pgid=child.pid,
                    boot_id=gate.boot_id(),
                    hostname=socket.gethostname(),
                    resumed_from_step=25000,
                )
                manager.event("power_resume_started", run["id"], pid=child.pid)
            while children:
                for child, run in children[:]:
                    code = child.poll()
                    if code is None:
                        continue
                    child.wait()
                    record = manager.state["runs"][run["id"]]
                    record["exit_code"] = code
                    try:
                        gate.require(code == 0, "Recovered trainer exit failure")
                        gate.require(not gate.group_alive(record), "Recovered trainer has live children")
                        summary = manager.verified(run)
                        record.update(status="completed", summary=summary, finished_unix=time.time())
                    except Exception as error:
                        record.update(status="failed", error=str(error))
                    manager.event("power_resume_" + record["status"], run["id"], exit_code=code)
                    children.remove((child, run))
                if children:
                    time.sleep(5)
            gate.require(
                all(manager.state["runs"][r["id"]]["status"] == "completed" for r in pair),
                "Recovery failed; later stages blocked",
            )
            gate.verify_paired([manager.state["runs"][r["id"]]["summary"] for r in pair])
            manager.event("power_resumed_pair_verified", None)
            manager.execute()
        except BaseException as error:
            manager.state.update(status="failed", error_type=type(error).__name__, error=str(error))
            manager.save()
            raise


if __name__ == "__main__":
    main()
