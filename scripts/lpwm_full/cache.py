"""Bounded official130-task LIBERO cache: all frames, lossless RGB records, one raw task at a time."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

CAMERAS = ["observation.images.image", "observation.images.image2"]
SUITES = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_90": 90, "libero_10": 10}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def rgb_pair(agent, wrist):
    # Match evaluate.py: simulator OpenGL camera is rotated180 exactly once.
    return np.stack(
        [
            np.asarray(
                Image.fromarray(np.ascontiguousarray(image[::-1, ::-1])).resize(
                    (128, 128), Image.Resampling.BILINEAR
                )
            ).transpose(2, 0, 1)
            for image in (agent, wrist)
        ]
    )


def download(catalog, task, path, endpoint):
    if path.exists() and path.stat().st_size == task["bytes"] and digest(path) == task["sha256"]:
        return
    url = f"{endpoint}/datasets/{catalog['repo_id']}/resolve/{catalog['revision']}/{task['demo']}"
    part = path.with_suffix(".partial")
    for attempt in range(6):
        if shutil.disk_usage(path.parent).free < task["bytes"] + 3 * 2**30:
            raise OSError("Preserve3GiB scratch filesystem reserve; no automatic deletion")
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with part.open("wb") as f:
                    for block in response.iter_content(4 * 1024**2):
                        f.write(block)
            if part.stat().st_size != task["bytes"] or digest(part) != task["sha256"]:
                raise ValueError("Official source size/SHA256 mismatch")
            part.replace(path)
            return
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(60, 2**attempt))


def convert_task(catalog, task, root, endpoint, *, cache_reserve_gib=8):
    tid = task["global_task_id"]
    output = root / f"task_{tid:03d}"
    output.mkdir(exist_ok=True)
    marker = output / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["source_sha256"] != task["sha256"]:
            raise ValueError("Completed cache has wrong source")
        for name, value in record["files_sha256"].items():
            if digest(output / name) != value:
                raise ValueError("Completed task cache is corrupt")
        return record
    scratch = Path(os.environ.get("LPWM_DOWNLOAD_SCRATCH", str(root)))
    scratch.mkdir(parents=True, exist_ok=True)
    source = scratch / f"task_{tid:03d}.source.hdf5"
    download(catalog, task, source, endpoint)
    # Only our newly downloaded scratch source is removed, after verifying committed cache.
    offsets, lengths, states, actions, episodes = [], [], [], [], []
    frame_in_block, block_frames = [], []
    cursor = 0
    with h5py.File(source, "r") as f, (output / "images.bin.partial").open("wb") as packed:
        env = json.loads(f["data"].attrs["env_args"])
        if env["env_kwargs"]["control_freq"] != 20 or f["data"].attrs["macros_image_convention"] != "opengl":
            raise ValueError("Unsupported camera/control-rate source contract")
        if env["env_kwargs"]["controller_configs"]["type"] != "OSC_POSE":
            raise ValueError("Expected nativeOSC_POSE actions")
        problem = json.loads(f["data"].attrs["problem_info"])
        if problem["language_instruction"].strip().lower() != task["language"].strip().lower():
            raise ValueError("Official task language differs from installed simulator")
        demos = sorted(f["data"], key=lambda name: int(name.split("_")[-1]))
        for demo in demos:
            d = f["data"][demo]
            obs = d["obs"]
            a = np.asarray(d["actions"], dtype=np.float32)
            s = np.concatenate(
                [np.asarray(obs["ee_pos"]), np.asarray(obs["ee_ori"]), np.asarray(obs["gripper_states"])],
                axis=-1,
            ).astype(np.float32)
            if (
                a.shape != (len(s), 7)
                or s.shape[1] != 8
                or not np.isfinite(a).all()
                or not np.isfinite(s).all()
            ):
                raise ValueError("Bad state/action shape or values")
            if len(s) < 18:
                raise ValueError("Episode cannot provide history2/horizon16 windows")
            agent, wrist = obs["agentview_rgb"], obs["eye_in_hand_rgb"]
            if len(agent) != len(s) or len(wrist) != len(s):
                raise ValueError("Image/action temporal length mismatch")
            for start in range(0, len(s), 16):
                pixels = np.stack(
                    [rgb_pair(agent[index], wrist[index]) for index in range(start, min(start + 16, len(s)))]
                )
                residual = pixels.copy()
                residual[1:] = pixels[1:] ^ pixels[:-1]
                compressed = zlib.compress(residual.tobytes(), level=3)
                restored = np.bitwise_xor.accumulate(
                    np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(pixels.shape), axis=0
                )
                if not np.array_equal(restored, pixels):
                    raise ValueError("Lossless temporal block roundtrip failed")
                for frame in range(len(pixels)):
                    offsets.append(packed.tell())
                    lengths.append(len(compressed))
                    frame_in_block.append(frame)
                    block_frames.append(len(pixels))
                packed.write(compressed)
                if shutil.disk_usage(output).free < cache_reserve_gib * 2**30:
                    raise OSError("Cache exhausted checkpoint reserve; no history pruning")
            states.append(s)
            actions.append(a)
            episodes.append({"source_demo": demo, "start": cursor, "end": cursor + len(s), "task_index": tid})
            cursor += len(s)
    (output / "images.bin.partial").replace(output / "images.bin")
    np.save(output / "states.npy", np.concatenate(states))
    np.save(output / "actions.npy", np.concatenate(actions))
    np.savez(
        output / "index.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        frame_in_block=np.asarray(frame_in_block, dtype=np.int64),
        block_frames=np.asarray(block_frames, dtype=np.int64),
    )
    record = {
        "task": task,
        "source_sha256": digest(source),
        "frames": cursor,
        "episodes": episodes,
        "native_control_freq_hz": 20,
        "all_frame_lossless_roundtrip": True,
        "image_codec": "xor16_zlib_rgb128_v1",
        "files_sha256": {
            name: digest(output / name) for name in ("images.bin", "states.npy", "actions.npy", "index.npz")
        },
    }
    atomic(marker, record)
    source.unlink()
    return record


def assemble(root, catalog):
    records = [
        json.loads((root / f"task_{t['global_task_id']:03d}/complete.json").read_text())
        for t in catalog["tasks"]
    ]
    episodes = []
    state = []
    action = []
    tasks = []
    shards = []
    cursor = 0
    for record in records:
        tid = record["task"]["global_task_id"]
        folder = root / f"task_{tid:03d}"
        for name, value in record["files_sha256"].items():
            if digest(folder / name) != value:
                raise ValueError("Cache changed before finalassembly")
        state.append(np.load(folder / "states.npy"))
        action.append(np.load(folder / "actions.npy"))
        tasks.append(np.full(record["frames"], tid, dtype=np.int64))
        for ep in record["episodes"]:
            episodes.append(
                {
                    **ep,
                    "episode_index": len(episodes),
                    "start": cursor + ep["start"],
                    "end": cursor + ep["end"],
                    "suite": record["task"]["suite"],
                    "suite_task_id": record["task"]["task_id"],
                }
            )
        shards.append(
            {
                "start": cursor,
                "end": cursor + record["frames"],
                "path": str(folder.relative_to(root) / "images.bin"),
                "index": str(folder.relative_to(root) / "index.npz"),
                "files_sha256": record["files_sha256"],
            }
        )
        cursor += record["frames"]
    for name, array in (
        ("states", np.concatenate(state)),
        ("actions", np.concatenate(action)),
        ("task_index", np.concatenate(tasks)),
    ):
        np.save(root / f"{name}.npy", array)
    manifest = {
        "schema": "lpwm_libero_full_zlib_v1",
        "image_codec": "xor16_zlib_rgb128_v1",
        "cameras": CAMERAS,
        "image_size": 128,
        "num_frames": cursor,
        "episodes": episodes,
        "tasks": {str(t["global_task_id"]): t["language"] for t in catalog["tasks"]},
        "task_catalog": catalog,
        "image_shards": shards,
        "image_preprocessing": "native OpenGL RGB rotate180 ONCE; PIL bilinear128; lossless zlib of resized uint8",
        "temporal_semantics": "all official20Hz demo rows retained, no frame skipping/no-op filtering; a[t]conditions I[t]->I[t+1]",
        "action_normalization": "identity; nativeLIBERO7 OSC_POSE controls",
        "scalar_sha256": {
            f"{name}.npy": digest(root / f"{name}.npy") for name in ("states", "actions", "task_index")
        },
    }
    language = json.loads((root / "language_manifest.json").read_text())
    manifest["language"] = language
    if language["task_ids"] != list(range(130)):
        raise ValueError("Missing alltask language coverage")
    for name in ("language_embeddings.npy", "language_masks.npy"):
        manifest["scalar_sha256"][name] = digest(root / name)
    atomic(root / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    p.add_argument("--only-task", type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(a.catalog.read_text())
    assert len(catalog["tasks"]) == 130
    if a.only_task is not None:
        print(json.dumps(convert_task(catalog, catalog["tasks"][a.only_task], a.output, a.endpoint)))
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(convert_task, catalog, t, a.output, a.endpoint) for t in catalog["tasks"]]
        for completed, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            record = job.result()
            atomic(
                a.output / "preparation_status.json",
                {
                    "status": "preparing",
                    "complete_tasks": completed,
                    "total_tasks": 130,
                    "last_task": record["task"]["global_task_id"],
                    "time": time.time(),
                },
            )
            print(json.dumps({"complete_tasks": completed, "frames": record["frames"]}), flush=True)
    manifest = assemble(a.output, catalog)
    atomic(
        a.output / "preparation_status.json",
        {"status": "complete", "tasks": 130, "frames": manifest["num_frames"], "time": time.time()},
    )


if __name__ == "__main__":
    main()
