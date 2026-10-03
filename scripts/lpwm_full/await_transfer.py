"""Wait for a verified common40 cache transfer, then delegate to the strict supervisor."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def transferred(cache):
    """Manifest is uploaded last; require exact verified bytes, never just file existence."""
    try:
        verification = json.loads((cache / "verification.json").read_text())
        with (cache / "manifest.json").open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        return (
            verification["status"] == "complete"
            and verification["tasks"] == 40
            and verification["frames"] == 273465
            and verification["episodes"] == 1693
            and verification["all_source_scalars_exact"] is True
            and verification["all_pixels_verified_via_PackedImages"] is True
            and verification["manifest_sha256"] == actual
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("Positive timeout required")
    root = args.root.resolve()
    with (root / "transfer_launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.timeout
        while not transferred(root / "cache"):
            timed_out = time.monotonic() >= deadline
            state = {
                "status": "failed_transfer_timeout" if timed_out else "awaiting_verified_cache_transfer",
                "training_started": False,
                "launcher_pid": os.getpid(),
                "task_count": 40,
                "updated_unix": time.time(),
                "received_task_markers": len(list((root / "cache").glob("task_*/complete.json"))),
            }
            target = root / "pipeline_status.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(target)
            if timed_out:
                raise TimeoutError("Common40 transfer did not finish; no training launched")
            time.sleep(15)

        def run_stage(stage, command):
            target = root / "pipeline_status.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(
                json.dumps(
                    {"status": stage, "training_started": False, "updated_unix": time.time()}, indent=2
                )
            )
            temp.replace(target)
            try:
                subprocess.run(command, check=True, cwd=root / "repo")
            except BaseException as error:
                temp.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "stage": stage,
                            "error": str(error),
                            "updated_unix": time.time(),
                        },
                        indent=2,
                    )
                )
                temp.replace(target)
                raise

        run_stage(
            "profiling_common40_cache",
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.lpwm_full.profile_batch",
                "--data",
                str(root / "cache"),
                "--output",
                str(root / "profiles_common40/batch_8.json"),
                "--batch-size",
                "8",
            ],
        )
        run_stage(
            "preflight_common40_cache",
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.lpwm_full.train",
                "--data",
                str(root / "cache"),
                "--output",
                str(root / "preflight_common40_run"),
                "--steps",
                "2",
                "--preflight",
                "--world-weight",
                "1",
                "--rec-weight",
                "1",
                "--dyn-weight",
                "1",
                "--prior-weight",
                "0.001",
                "--workers",
                "4",
                "--validation-batches",
                "5",
                "--project",
                "lpwm-fm-b-full-libero",
                "--run-name",
                "common40-realcache-preflight",
                "--credential-file",
                "/root/lpwm_ab/.swanlab_api_key",
                "--eval-python",
                "/root/lpwm_ab/.venv-eval/bin/python",
                "--eval-runner",
                str(root / "repo/scripts/lpwm_full/evaluate.py"),
                "--expected-init-sha256",
                "3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e",
            ],
        )
        run_stage(
            "starting_supervisor",
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.lpwm_full.supervise",
                "--root",
                str(root),
                "--task-count",
                "40",
                "--batch-size",
                "8",
                "--grad-accumulation",
                "4",
                "--preflight-run",
                "preflight_common40_run",
                "--profile-gate",
                "profiles_common40/batch_8.json",
            ],
        )


if __name__ == "__main__":
    main()
