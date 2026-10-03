"""Run one predeclared treatment queue on one GPU, keeping original training untouched."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Supervise sequential experiments; no parameter changes or retries on failures."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan/plan.json").read_text())
    queue = plan["queues"][str(args.gpu)]
    (root / "logs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="8",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = f"{root}/source:{root}/source/src:{root.parent}/python_deps"
    results = []
    with (root / f"gpu{args.gpu}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in queue:
            command = [
                sys.executable,
                "-u",
                "-m",
                "RL.cli.validate_fastwam_q",
                "--case",
                case,
                "--checkpoint",
                str(root.parent / "run/step_005000"),
                "--cache",
                "/data/workspace/droid_franka4_20260929/cache/franka4_lossless",
                "--plan",
                str(root / "plan"),
                "--output",
                str(root / "results"),
                "--credential-file",
                "/data/workspace/droid_franka4_20260929/.secrets/swanlab.key",
                "--workers",
                "4",
                "--seed",
                str(plan["seed"]),
            ]
            status_path = root / f"queue_gpu{args.gpu}.json"
            status_path.write_text(
                json.dumps({"status": "running", "case": case, "finished": results, "queue": queue}, indent=2)
            )
            with (root / "logs" / f"{case}.log").open("w") as log:
                code = subprocess.call(
                    command, cwd=root / "source", env=env, stdout=log, stderr=subprocess.STDOUT
                )
            summary = root / "results" / case / "summary.json"
            results.append(
                {
                    "case": case,
                    "exit_code": code,
                    "summary": json.loads(summary.read_text()) if summary.exists() else None,
                }
            )
            status_path.write_text(
                json.dumps({"status": "between_cases", "finished": results, "queue": queue}, indent=2)
            )
        status_path.write_text(
            json.dumps({"status": "complete", "finished": results, "queue": queue}, indent=2)
        )


if __name__ == "__main__":
    main()
