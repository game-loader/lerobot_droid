"""Build a common40 view of an existing official130 cache without downloading or copying images."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.lpwm_full.evaluate import COMMON_SUITES, catalog_suites
from scripts.lpwm_full.supervise import verify_cache


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def subset_cache(source: Path, output: Path):
    """Remap canonical IDs/offsets, hardlink immutable task payloads, subset frozen language rows."""
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_symlink() or output.is_relative_to(source):
        raise ValueError("Use a new output outside the source cache; never modify the running dataset")
    manifest_path = source / "manifest.json"
    source_hash = digest(manifest_path)
    original = json.loads(manifest_path.read_text())
    if sum(catalog_suites(original["task_catalog"]).values()) != 130:
        raise ValueError("Expected the canonical all130 source cache")
    if original["schema"] != "lpwm_libero_full_zlib_v1" or original.get("preflight_only"):
        raise ValueError("A complete official cache is required")
    for name, expected in original["scalar_sha256"].items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError(f"Changed source scalar: {name}")
    tasks = [t for t in original["task_catalog"]["tasks"] if t["suite"] in COMMON_SUITES]
    catalog = copy.deepcopy(original["task_catalog"])
    catalog["tasks"] = [
        dict(t, global_task_id=i, source_global_task_id=t["global_task_id"]) for i, t in enumerate(tasks)
    ]
    catalog["suites"] = list(COMMON_SUITES)
    catalog["total_source_bytes"] = sum(t["bytes"] for t in tasks)
    if catalog_suites(catalog) != COMMON_SUITES:
        raise ValueError("Incorrect common40 mapping")
    language = original["language"]
    if len(set(language["task_ids"])) != len(language["task_ids"]):
        raise ValueError("Duplicate source language IDs")
    lookup = {tid: row for row, tid in enumerate(language["task_ids"])}
    rows = [lookup[t["global_task_id"]] for t in tasks]
    embeddings = np.load(source / "language_embeddings.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(source / "language_masks.npy", mmap_mode="r", allow_pickle=False)
    if embeddings.shape[0] != len(lookup) or masks.shape != embeddings.shape[:2]:
        raise ValueError("Language dimensions do not match metadata")
    global_states = np.load(source / "states.npy", mmap_mode="r", allow_pickle=False)
    global_actions = np.load(source / "actions.npy", mmap_mode="r", allow_pickle=False)
    global_tasks = np.load(source / "task_index.npy", mmap_mode="r", allow_pickle=False)
    if global_states.shape != (original["num_frames"], 8) or global_actions.shape != (
        original["num_frames"],
        7,
    ):
        raise ValueError("Expected native state8/action7")
    if global_tasks.shape != (original["num_frames"],):
        raise ValueError("Incorrect global task array")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preparation_status.json", {"status": "preparing", "network_downloads": 0})
    states, actions, task_rows, episodes, shards, links = [], [], [], [], [], []
    cursor = 0
    try:
        for new_task in catalog["tasks"]:
            new_id, old_id = new_task["global_task_id"], new_task["source_global_task_id"]
            old_task = original["task_catalog"]["tasks"][old_id]
            folder = source / f"task_{old_id:03d}"
            record = json.loads((folder / "complete.json").read_text())
            if record["task"] != old_task or record["source_sha256"] != old_task["sha256"]:
                raise ValueError("Wrong source task completion identity")
            if record.get("all_frame_lossless_roundtrip") is not True:
                raise ValueError("Source images lack complete roundtrip verification")
            old_shards = [s for s in original["image_shards"] if s["path"] == f"task_{old_id:03d}/images.bin"]
            if len(old_shards) != 1:
                raise ValueError("Expected one native shard per task")
            old_shard = old_shards[0]
            start, end = old_shard["start"], old_shard["end"]
            if end - start != record["frames"]:
                raise ValueError("Task frame offset mismatch")
            state = np.load(folder / "states.npy", allow_pickle=False)
            action = np.load(folder / "actions.npy", allow_pickle=False)
            if not (
                np.array_equal(state, global_states[start:end])
                and np.array_equal(action, global_actions[start:end])
                and np.all(global_tasks[start:end] == old_id)
                and state.shape == (record["frames"], 8)
                and action.shape == (record["frames"], 7)
            ):
                raise ValueError("Task scalar rows do not match global cache")
            dest = output / f"task_{new_id:03d}"
            dest.mkdir()
            hashes = record["files_sha256"]
            if set(hashes) != {"images.bin", "index.npz", "states.npy", "actions.npy"}:
                raise ValueError("Unexpected native task files")
            for name, expected in hashes.items():
                old_path = (folder / name).resolve(strict=True)
                if not old_path.is_relative_to(source) or digest(old_path) != expected:
                    raise ValueError(f"Changed source payload: task{old_id}/{name}")
                # Deliberately fail on EXDEV, instead of silently copying image gigabytes.
                os.link(old_path, dest / name)
                if not os.path.samefile(old_path, dest / name):
                    raise ValueError("Hardlink identity mismatch")
                links.append(
                    {"source": str(old_path), "target": str(dest / name), "bytes": old_path.stat().st_size}
                )
            local_episodes = []
            old_episodes = [ep for ep in original["episodes"] if ep["task_index"] == old_id]
            if len(old_episodes) != len(record["episodes"]):
                raise ValueError("Incomplete source episode metadata")
            relative = 0
            for ep, full_ep in zip(record["episodes"], old_episodes, strict=True):
                if not (
                    ep["start"] == relative
                    and ep["end"] > relative
                    and full_ep["start"] == start + ep["start"]
                    and full_ep["end"] == start + ep["end"]
                    and full_ep["source_demo"] == ep["source_demo"]
                    and full_ep["suite"] == new_task["suite"]
                    and full_ep["suite_task_id"] == new_task["task_id"]
                ):
                    raise ValueError("Episode/image/scalar alignment mismatch")
                local_episodes.append(dict(ep, task_index=new_id))
                episodes.append(
                    dict(
                        full_ep,
                        episode_index=len(episodes),
                        task_index=new_id,
                        start=cursor + ep["start"],
                        end=cursor + ep["end"],
                        source_full_episode_index=full_ep["episode_index"],
                        source_global_task_id=old_id,
                    )
                )
                relative = ep["end"]
            if relative != record["frames"]:
                raise ValueError("Episodes do not cover all task rows")
            write_json(
                dest / "complete.json",
                dict(record, task=new_task, episodes=local_episodes, source_full_cache=str(source)),
            )
            states.append(state)
            actions.append(action)
            task_rows.append(np.full(record["frames"], new_id, dtype=np.int64))
            shards.append(
                {
                    "start": cursor,
                    "end": cursor + record["frames"],
                    "path": f"{dest.name}/images.bin",
                    "index": f"{dest.name}/index.npz",
                    "files_sha256": hashes,
                }
            )
            cursor += record["frames"]
        for name, value in (
            ("states", np.concatenate(states)),
            ("actions", np.concatenate(actions)),
            ("task_index", np.concatenate(task_rows)),
            ("language_embeddings", embeddings[rows]),
            ("language_masks", masks[rows]),
        ):
            np.save(output / f"{name}.npy", value, allow_pickle=False)
        language = dict(
            language,
            task_ids=list(range(40)),
            task_count=40,
            unique_texts=len({t["language"] for t in tasks}),
            source_task_ids=[t["global_task_id"] for t in tasks],
            source_manifest_sha256=source_hash,
        )
        write_json(output / "language_manifest.json", language)
        result = copy.deepcopy(original)
        result.update(
            num_frames=cursor,
            episodes=episodes,
            task_catalog=catalog,
            tasks={str(t["global_task_id"]): t["language"] for t in catalog["tasks"]},
            image_shards=shards,
            language=language,
            scalar_sha256={
                f"{name}.npy": digest(output / f"{name}.npy")
                for name in ("states", "actions", "task_index", "language_embeddings", "language_masks")
            },
        )
        result["subset_provenance"] = {
            "source": str(source),
            "source_manifest_sha256": source_hash,
            "excluded_suites": ["libero_90"],
            "image_storage": "same-inode hardlinks",
            "source_task_ids": [t["global_task_id"] for t in tasks],
        }
        verify_cache(output, result, 40)
        if digest(manifest_path) != source_hash:
            raise ValueError("Source manifest changed during subsetting")
        report = {
            "status": "complete",
            "tasks": 40,
            "episodes": len(episodes),
            "frames": cursor,
            "network_downloads": 0,
            "image_bytes_copied": 0,
            "hardlinked_files": len(links),
            "hardlinked_logical_bytes": sum(x["bytes"] for x in links),
            "links": links,
            "source_manifest_sha256": source_hash,
            "original_source_unchanged": True,
        }
        write_json(output / "reuse_verification.json", report)
        write_json(output / "manifest.json", result)
        write_json(output / "preparation_status.json", {k: v for k, v in report.items() if k != "links"})
        return result
    except BaseException as error:
        write_json(
            output / "preparation_status.json", {"status": "failed", "error_type": type(error).__name__}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = subset_cache(args.source, args.output)
    print(
        json.dumps(
            {
                "tasks": 40,
                "frames": manifest["num_frames"],
                "episodes": len(manifest["episodes"]),
                "image_bytes_copied": 0,
                "network_downloads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
