"""Complete pinned all130 cache without deleting historical artifacts; two explicit storage tiers."""

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path

from scripts.lpwm_full.cache import assemble, atomic, convert_task, digest

GIB = 2**30


def run_task(catalog, task, cache, scratch, reserve):
    os.environ["LPWM_DOWNLOAD_SCRATCH"] = str(scratch)
    return convert_task(catalog, task, cache, "https://hf-mirror.com", cache_reserve_gib=reserve)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reuse", type=Path, required=True)
    p.add_argument("--overflow", type=Path, required=True)
    p.add_argument("--primary-new-limit-gib", type=float, default=9.5)
    args = p.parse_args()
    root, old, overflow = args.root.resolve(), args.reuse.resolve(), args.overflow.resolve()
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    overflow.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((root / "task_catalog.json").read_text())
    if len(catalog["tasks"]) != 130:
        raise ValueError("Full130 catalog required")
    reused, complete = [], []
    for task in catalog["tasks"]:
        tid = task["global_task_id"]
        folder = f"task_{tid:03d}"
        previous = old / folder
        dest = cache / folder
        if (previous / "complete.json").is_file():
            record = json.loads((previous / "complete.json").read_text())
            if record["source_sha256"] != task["sha256"] or record["task"] != task:
                raise ValueError("Wrong original task identity")
            for name, expected in record["files_sha256"].items():
                if digest(previous / name) != expected:
                    raise ValueError(f"Corrupt reusable cache {folder}/{name}")
            if not dest.exists():
                shutil.copytree(previous, dest, copy_function=os.link)
            reused.append(tid)
            complete.append(tid)
        elif (dest / "complete.json").is_file():
            record = json.loads((dest / "complete.json").read_text())
            if record["source_sha256"] != task["sha256"]:
                raise ValueError("Wrong resumed source")
            for name, expected in record["files_sha256"].items():
                if digest(dest / name) != expected:
                    raise ValueError("Corrupt completed new cache")
            complete.append(tid)
    atomic(
        root / "cache_reuse.json",
        {"source": str(old), "verified_hardlinked_tasks": reused, "overflow_root": str(overflow)},
    )
    primary_scratch = root / "download_scratch"
    system_scratch = overflow.parent / "download_scratch"
    primary_scratch.mkdir(exist_ok=True)
    system_scratch.mkdir(exist_ok=True)

    def progress(state, **kwargs):
        d = {
            "status": state,
            "complete_tasks": len(complete),
            "total_tasks": 130,
            "verified_reused_tasks": len(reused),
            "time": time.time(),
            **kwargs,
        }
        atomic(cache / "preparation_status.json", d)
        print(json.dumps(d), flush=True)

    def primary_new_bytes():
        return sum(
            (cache / f"task_{tid:03d}" / "images.bin").stat().st_size
            for tid in complete
            if tid not in reused and not (cache / f"task_{tid:03d}").is_symlink()
        )

    try:
        progress("preparing")
        # Process primary tier in pairs; when full, overflow tier serially keeps ample raw scratch space.
        todo = [t for t in catalog["tasks"] if t["global_task_id"] not in complete]
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            while todo:
                first = todo[0]
                estimate = first["bytes"] * 0.45
                primary = (
                    primary_new_bytes() + estimate < args.primary_new_limit_gib * GIB
                    and shutil.disk_usage(cache).free > 6 * GIB + estimate
                )
                count = 1
                if primary and len(todo) >= 2:
                    estimate2 = sum(t["bytes"] for t in todo[:2]) * 0.45
                    if (
                        primary_new_bytes() + estimate2 < args.primary_new_limit_gib * GIB
                        and shutil.disk_usage(cache).free > 6 * GIB + estimate2
                        and shutil.disk_usage(system_scratch).free
                        > sum(t["bytes"] for t in todo[:2]) + 3.5 * GIB
                    ):
                        count = 2
                selected, todo = todo[:count], todo[count:]
                jobs = []
                for task in selected:
                    tid = task["global_task_id"]
                    folder = cache / f"task_{tid:03d}"
                    if not primary and not folder.exists():
                        target = overflow / folder.name
                        target.mkdir(exist_ok=True)
                        folder.symlink_to(target, target_is_directory=True)
                    primary_task = not folder.is_symlink() if folder.exists() else primary
                    scratch = system_scratch if primary_task else primary_scratch
                    reserve = 5.5 if primary_task else 2.5
                    progress("preparing", current_task=tid, tier="data" if primary_task else "system")
                    jobs.append(pool.submit(run_task, catalog, task, cache, scratch, reserve))
                for future in concurrent.futures.as_completed(jobs):
                    record = future.result()
                    complete.append(record["task"]["global_task_id"])
                    progress(
                        "preparing",
                        last_task=record["task"]["global_task_id"],
                        primary_new_gib=primary_new_bytes() / GIB,
                    )
        progress("assembling")
        manifest = assemble(cache, catalog)
        if len(manifest["episodes"]) != 6500:
            raise ValueError("All official50demonstrations/task are required")
        progress("complete", tasks=130, episodes=len(manifest["episodes"]), frames=manifest["num_frames"])
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
