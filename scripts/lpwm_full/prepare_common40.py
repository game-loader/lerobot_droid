"""CPU-only, row-preserving LeRobot v3 common40 -> PackedImages cache.

Preserve the local dataset's RGB orientation, state8, action7 and every row.
The metadata says fps=10; no original physical rate or resampling is inferred.
Do NOT apply cache.rgb_pair's raw-OpenGL rotation.
Images live in per-task files; episode-local indices expose them in original
source row order to the unmodified PackedImages reader. No remote operations,
model downloads, simulator imports or GPU execution are performed here.

Example (reference130 was copied read-only from the existing remote cache)::

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      nice -n 19 ionice -c 3 uv run --no-sync python -m scripts.lpwm_full.prepare_common40 \
      --reference /data/lpwm_common40_cache/20260921/reference130 --workers 4
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import shutil
import time
import zlib
from collections import Counter, deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.lpwm_ab.data import build_split, training_state_statistics

from .data import FullCachedDataset, PackedImages

CAMERAS = ["observation.images.image", "observation.images.image2"]
SUITES = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10}
CODEC = "xor16_zlib_rgb128_v1"
SCHEMA = "lpwm_libero_full_zlib_v1"
CONVERTER = "lerobot_v3_common40_row_preserving_v1"
INDEX_KEYS = ("offsets", "lengths", "frame_in_block", "block_frames")
SCALARS = ("observation.state", "action", "index", "episode_index", "frame_index", "task_index")
DEFAULT_SOURCE = Path("/data/lerobot_datasets/HuggingFaceVLA/libero")
DEFAULT_OUTPUT = Path("/data/lpwm_common40_cache/20260921")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inspect_source(source, *, expected_counts=(1693, 273465, 40)):
    """Audit every scalar row before decoding images; fingerprint all source files."""
    source = Path(source)
    info = read_json(source / "meta/info.json")
    counts = tuple(info[k] for k in ("total_episodes", "total_frames", "total_tasks"))
    require(counts == expected_counts, f"Unexpected source counts: {counts} != {expected_counts}")
    require(info["codebase_version"] == "v3.0", "Expected LeRobot v3.0")
    require(info["fps"] == 10, "Expected metadata fps=10")
    for name, dim in (("observation.state", 8), ("action", 7)):
        feature = info["features"][name]
        require(feature["dtype"] == "float32" and feature["shape"] == [dim], f"Bad {name} schema")
    for name in CAMERAS:
        feature = info["features"][name]
        require(feature["dtype"] == "image" and feature["shape"] == [256, 256, 3], "Expected RGB256")
    tasks_table = pq.read_table(source / "meta/tasks.parquet").to_pydict()
    # LeRobot tasks.parquet serializes the pandas language index under this name.
    texts = tasks_table.get("__index_level_0__", tasks_table.get("task"))
    require(texts is not None, "Missing task language index")
    tasks = dict(zip(tasks_table["task_index"], texts, strict=True))
    require(len(texts) == len(tasks) == counts[2], "Duplicate/missing task IDs")
    require(set(tasks) == set(range(counts[2])), "Task IDs must be contiguous local IDs")
    require(len(set(texts)) == len(texts), "Ambiguous local task languages")
    episodes = pq.read_table(
        source / "meta/episodes",
        columns=["episode_index", "dataset_from_index", "dataset_to_index", "length", "tasks"],
    ).to_pylist()
    episodes.sort(key=lambda ep: ep["episode_index"])
    require([e["episode_index"] for e in episodes] == list(range(counts[0])), "Bad episode IDs")
    files = sorted(source.glob("data/chunk-*/*.parquet"))
    require(bool(files), "No source Parquet files")
    values = {k: [] for k in SCALARS}
    fingerprints = {}
    for file in files:
        table = pq.read_table(file, columns=list(SCALARS), use_threads=False)
        for key in SCALARS:
            column = table[key].combine_chunks()
            if key in ("observation.state", "action"):
                dim = 8 if key == "observation.state" else 7
                lengths = np.diff(column.offsets.to_numpy())
                require(np.all(lengths == dim) and column.null_count == 0, f"Bad {key} row width")
                array = column.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
                require(array.dtype == np.float32 and np.isfinite(array).all(), f"Bad {key} values")
            else:
                require(column.null_count == 0, f"Null {key}")
                array = column.to_numpy(zero_copy_only=False)
                require(array.dtype == np.int64, f"Bad {key} dtype")
            values[key].append(array)
        fingerprints[str(file.relative_to(source))] = {"bytes": file.stat().st_size, "sha256": digest(file)}
    arrays = {key: np.concatenate(value) for key, value in values.items()}
    n = counts[1]
    require(np.array_equal(arrays["index"], np.arange(n)), "Source global row index is not contiguous")
    cursor = 0
    for ep in episodes:
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        require(start == cursor and end - start == ep["length"] and end <= n, "Episode coverage mismatch")
        require(ep["length"] >= 18, "Episode too short for history2/horizon16")
        tid = int(arrays["task_index"][start])
        require(tid in tasks and ep["tasks"] == [tasks[tid]], "Episode language/task mismatch")
        require(np.all(arrays["task_index"][start:end] == tid), "Task changed inside episode")
        require(np.all(arrays["episode_index"][start:end] == ep["episode_index"]), "Episode rows shifted")
        require(
            np.array_equal(arrays["frame_index"][start:end], np.arange(end - start)), "Frame rows shifted"
        )
        ep.update(start=start, end=end, task_index=tid, source_episode_index=ep["episode_index"])
        cursor = end
    require(cursor == n and set(arrays["task_index"]) == set(tasks), "Incomplete source coverage")
    for file in sorted((source / "meta").rglob("*")):
        if file.is_file():
            fingerprints[str(file.relative_to(source))] = {
                "bytes": file.stat().st_size,
                "sha256": digest(file),
            }
    snapshot = {"root": str(source.resolve()), "info": info, "files": fingerprints}
    return info, tasks, episodes, arrays, files, snapshot


def match_catalog(tasks, reference, *, expected_suites=SUITES):
    """Use exact text, not local task ordering or numeric IDs, to recover identity."""
    result = []
    offsets = {}
    for suite in expected_suites:
        offsets[suite] = sum(expected_suites[s] for s in offsets)
    for tid, language in sorted(tasks.items()):
        matches = [
            t for t in reference["tasks"] if t["suite"] in expected_suites and t["language"] == language
        ]
        require(len(matches) == 1, f"Expected one exact four-suite match for task {tid}: {language!r}")
        match = matches[0]
        # Raw HDF5 provenance is reference-only, never provenance for local rows.
        task = {k: v for k, v in match.items() if k not in ("global_task_id", "bytes", "sha256", "demo")}
        task.update(
            global_task_id=offsets[match["suite"]] + match["task_id"],
            source_task_index=tid,
            official_global_task_id=match["global_task_id"],
            official_source_reference={k: match[k] for k in ("demo", "bytes", "sha256") if k in match},
        )
        result.append(task)
    require(dict(Counter(t["suite"] for t in result)) == expected_suites, "Incomplete four-suite catalog")
    for suite, count in expected_suites.items():
        require(
            {t["task_id"] for t in result if t["suite"] == suite} == set(range(count)), "Bad suite task IDs"
        )
    result.sort(key=lambda task: task["global_task_id"])
    return {
        "repo_id": "HuggingFaceVLA/libero",
        "suites": expected_suites,
        "tasks": result,
        "reference_catalog_sha256": object_digest(reference),
        "matching": "exact language text within common40 suites; canonical suite/task-ID order; source_task_index retains original local ID",
        "image_contract": "preserve converted RGB orientation; PIL bilinear128; no rotation/flip",
        "training_labels": "unchanged source-row float32 state8/action7 with metadata fps=10; no physical frequency/resampling inferred",
    }


def reuse_language(root, catalog, reference, language_root):
    """Copy frozen rows by exact language with ID->row lookup, never positional guessing."""
    language_root = Path(language_root)
    metadata = read_json(language_root / "language_manifest.json")
    embeddings = np.load(language_root / "language_embeddings.npy", allow_pickle=False)
    masks = np.load(language_root / "language_masks.npy", allow_pickle=False)
    ids = metadata["task_ids"]
    require(len(ids) == len(set(ids)), "Duplicate reference language task IDs")
    require(
        embeddings.shape == (len(ids), metadata["max_length"], metadata["hidden_dim"]), "Bad language shape"
    )
    require(masks.shape == embeddings.shape[:2] and masks.dtype == bool, "Bad language masks")
    require(embeddings.dtype == np.float32 and np.isfinite(embeddings).all(), "Bad language embeddings")
    require(np.all(masks.any(axis=1)), "Empty language mask")
    rows = {tid: row for row, tid in enumerate(ids)}
    selected = []
    for task in catalog["tasks"]:
        matches = [
            t for t in reference["tasks"] if t["language"] == task["language"] and t["global_task_id"] in rows
        ]
        require(bool(matches), f"Missing exact language embedding: {task['language']}")
        row = rows[matches[0]["global_task_id"]]
        for match in matches[1:]:
            other = rows[match["global_task_id"]]
            require(
                np.array_equal(embeddings[row], embeddings[other])
                and np.array_equal(masks[row], masks[other]),
                "Conflicting embeddings for identical exact language",
            )
        selected.append(row)
    np.save(root / "language_embeddings.npy", embeddings[selected])
    np.save(root / "language_masks.npy", masks[selected])
    result = {
        **metadata,
        "task_ids": [t["global_task_id"] for t in catalog["tasks"]],
        "task_count": len(selected),
        "unique_texts": len({t["language"] for t in catalog["tasks"]}),
        "reuse_method": "exact text; byte-identical frozen reference rows; no encoding or task-ID features",
        "reference_task_ids": [ids[row] for row in selected],
        "reference_files_sha256": {
            name: digest(language_root / name)
            for name in ("language_manifest.json", "language_embeddings.npy", "language_masks.npy")
        },
    }
    atomic(root / "language_manifest.json", result)
    return result


def rgb_pair_preserved(row):
    pixels = []
    for camera in CAMERAS:
        encoded = row[camera]["bytes"]
        require(bool(encoded), "Expected embedded image bytes, not an external path")
        with Image.open(io.BytesIO(encoded)) as image:
            require(image.size == (256, 256), "Unexpected source image size")
            pixels.append(
                np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)).transpose(
                    2, 0, 1
                )
            )
    return np.stack(pixels)


def encode_block(rows):
    require(0 < len(rows) <= 16, "Invalid XOR block length")
    require(len({row["episode_index"] for row in rows}) == 1, "XOR block crosses episode")
    pixels = np.stack([rgb_pair_preserved(row) for row in rows])
    residual = pixels.copy()
    residual[1:] = pixels[1:] ^ pixels[:-1]
    compressed = zlib.compress(residual.tobytes(), level=3)
    restored = np.bitwise_xor.accumulate(
        np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(pixels.shape), axis=0
    )
    require(np.array_equal(pixels, restored), "Lossless image roundtrip failed")
    return compressed, hashlib.sha256(pixels.tobytes()).hexdigest()


def image_blocks(files, skip_tasks=(), task_mapping=None):
    """Bounded read-ahead; carry partial blocks across Parquet files, not episodes."""
    block = []
    previous = None
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=32, columns=[*CAMERAS, "index", "episode_index", "task_index"], use_threads=False
        ):
            for row in batch.to_pylist():
                require(previous is None or row["index"] == previous + 1, "Image source rows shifted")
                previous = row["index"]
                if task_mapping is not None:
                    row["task_index"] = task_mapping[row["task_index"]]
                if block and (len(block) == 16 or row["episode_index"] != block[0]["episode_index"]):
                    yield block
                    block = []
                if row["task_index"] not in skip_tasks:
                    block.append(row)
    if block:
        yield block


def ordered_encode(blocks, workers):
    # Threaded PIL/zlib release the GIL; bounded queue avoids copying 33 GiB via IPC.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for rows in blocks:
            pending.append((rows[0], len(rows), pool.submit(encode_block, rows)))
            if len(pending) >= 2 * workers:
                row, n, job = pending.popleft()
                yield row, n, job.result()
        while pending:
            row, n, job = pending.popleft()
            yield row, n, job.result()


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        require(digest(root / name) == expected, f"Cache SHA256 mismatch: {root / name}")


def finish_task(root, task, episodes, arrays, index_lists, block_checks, cursor_count, source_sha256):
    tid = task["global_task_id"]
    folder = root / f"task_{tid:03d}"
    (folder / "images.bin.partial").replace(folder / "images.bin")
    selected = np.flatnonzero(arrays["task_index"] == tid)
    require(cursor_count == len(selected), "Missing task frames")
    np.save(folder / "states.npy", arrays["observation.state"][selected])
    np.save(folder / "actions.npy", arrays["action"][selected])
    np.save(folder / "source_indices.npy", selected)
    index = {key: np.asarray(values, dtype=np.int64) for key, values in index_lists.items()}
    np.savez(folder / "index.npz", **index)
    atomic(folder / "blocks.json", block_checks)
    names = ["images.bin", "states.npy", "actions.npy", "source_indices.npy", "index.npz", "blocks.json"]
    cursor = 0
    task_episodes = []
    for ep in episodes:
        if ep["task_index"] != tid:
            continue
        length = ep["end"] - ep["start"]
        name = f"episode_{ep['episode_index']:06d}.npz"
        np.savez(folder / name, **{key: values[cursor : cursor + length] for key, values in index.items()})
        names.append(name)
        task_episodes.append(
            {
                "source_demo": f"episode_{ep['episode_index']:06d}",
                "episode_index": ep["episode_index"],
                "source_episode_index": ep["episode_index"],
                "source_start": ep["start"],
                "source_end": ep["end"],
                "start": cursor,
                "end": cursor + length,
                "task_index": tid,
                "index_file": name,
            }
        )
        cursor += length
    record = {
        "converter": CONVERTER,
        "task": task,
        "source_sha256": source_sha256,
        "frames": len(selected),
        "episodes": task_episodes,
        "source_metadata_fps": 10,
        "all_frame_lossless_roundtrip": True,
        "image_codec": CODEC,
        "files_sha256": {name: digest(folder / name) for name in names},
    }
    atomic(folder / "complete.json", record)
    return record


def convert_images(root, catalog, episodes, arrays, files, source_sha256, workers, reserve_bytes):
    records, indices, checks, cursors = {}, {}, {}, {}
    task_lookup = {t["global_task_id"]: t for t in catalog["tasks"]}
    task_ends = {ep["task_index"]: ep["end"] for ep in episodes}
    with ExitStack() as stack:
        handles = {}
        for task in catalog["tasks"]:
            tid = task["global_task_id"]
            folder = root / f"task_{tid:03d}"
            folder.mkdir(exist_ok=True)
            marker = folder / "complete.json"
            if marker.exists():
                record = read_json(marker)
                require(
                    record["source_sha256"] == source_sha256
                    and record["task"] == task
                    and record.get("converter") == CONVERTER,
                    "Completed task belongs to another source/contract",
                )
                verify_hashes(folder, record["files_sha256"])
                records[tid] = record
            else:
                handles[tid] = stack.enter_context((folder / "images.bin.partial").open("wb"))
                indices[tid] = {key: [] for key in INDEX_KEYS}
                checks[tid] = []
                cursors[tid] = 0
        last_report = time.monotonic()
        for row, n, (compressed, pixel_hash) in ordered_encode(
            image_blocks(
                files, set(records), {t["source_task_index"]: t["global_task_id"] for t in catalog["tasks"]}
            ),
            workers,
        ):
            tid, eid, start = row["task_index"], row["episode_index"], row["index"]
            ep = episodes[eid]
            require(
                ep["start"] <= start < start + n <= ep["end"] and ep["task_index"] == tid,
                "Block row mismatch",
            )
            require(
                shutil.disk_usage(root).free >= reserve_bytes + len(compressed),
                "Preserve disk reserve; no automatic deletion",
            )
            handle = handles[tid]
            index = indices[tid]
            index["offsets"].extend([handle.tell()] * n)
            index["lengths"].extend([len(compressed)] * n)
            index["frame_in_block"].extend(range(n))
            index["block_frames"].extend([n] * n)
            handle.write(compressed)
            checks[tid].append(
                {"start": start, "frames": n, "episode_index": eid, "pixels_sha256": pixel_hash}
            )
            cursors[tid] += n
            if start + n == task_ends[tid]:
                handle.close()
                records[tid] = finish_task(
                    root,
                    task_lookup[tid],
                    episodes,
                    arrays,
                    indices.pop(tid),
                    checks.pop(tid),
                    cursors.pop(tid),
                    source_sha256,
                )
                print(
                    json.dumps(
                        {
                            "status": "task_complete",
                            "task": tid,
                            "frames": records[tid]["frames"],
                            "episodes": len(records[tid]["episodes"]),
                            "complete_tasks": len(records),
                        }
                    ),
                    flush=True,
                )
            if time.monotonic() - last_report >= 15:
                status = {
                    "status": "preparing",
                    "source_rows_visited": start + n,
                    "num_frames": len(arrays["index"]),
                    "complete_tasks": len(records),
                    "workers": workers,
                    "cpu_only": True,
                    "time": time.time(),
                }
                atomic(root / "preparation_status.json", status)
                print(json.dumps(status), flush=True)
                last_report = time.monotonic()
    return records


def assemble(root, info, catalog, episodes, arrays, records, source_sha256, language):
    task_lookup = {t["global_task_id"]: t for t in catalog["tasks"]}
    manifest_episodes, shards = [], []
    for ep in episodes:
        tid, eid = ep["task_index"], ep["episode_index"]
        task = task_lookup[tid]
        folder = Path(f"task_{tid:03d}")
        name = f"episode_{eid:06d}.npz"
        manifest_episodes.append(
            {
                "episode_index": eid,
                "source_episode_index": eid,
                "source_demo": f"episode_{eid:06d}",
                "start": ep["start"],
                "end": ep["end"],
                "task_index": tid,
                "suite": task["suite"],
                "suite_task_id": task["task_id"],
                "source_task_index": task["source_task_index"],
            }
        )
        shards.append(
            {
                "start": ep["start"],
                "end": ep["end"],
                "path": str(folder / "images.bin"),
                "index": str(folder / name),
                "task_index": tid,
                "files_sha256": {k: records[tid]["files_sha256"][k] for k in ("images.bin", name)},
            }
        )
    for name, source_key in (
        ("states", "observation.state"),
        ("actions", "action"),
        ("task_index", "task_index"),
        ("source_task_index", "source_task_index"),
    ):
        np.save(root / f"{name}.npy", arrays[source_key])
    manifest = {
        "schema": SCHEMA,
        "converter": CONVERTER,
        "image_codec": CODEC,
        "cameras": CAMERAS,
        "image_size": 128,
        "num_frames": info["total_frames"],
        "num_episodes": len(episodes),
        "num_tasks": len(catalog["tasks"]),
        "fps": info["fps"],
        "episodes": manifest_episodes,
        "tasks": {str(t["global_task_id"]): t["language"] for t in catalog["tasks"]},
        "task_catalog": catalog,
        "image_shards": shards,
        "language": language,
        "source": {
            "repo_id": "HuggingFaceVLA/libero",
            "codebase_version": "v3.0",
            "manifest": "source_manifest.json",
            "sha256": source_sha256,
            "rows": "all original global rows and episode IDs, unchanged order",
        },
        "image_preprocessing": "existing converted RGB orientation preserved; no flip/rotate; PIL bilinear128; lossless XOR16/zlib",
        "temporal_semantics": "all source rows retained, metadata fps=10; no physical frequency inference/shift/filter/resample; state[t], action[t], image[t] are same source row",
        "action_normalization": "identity; source float32 action7 unchanged",
    }
    split = build_split(manifest, seed=42, validation_fraction=0.1)
    atomic(root / "split.json", split)
    atomic(
        root / "state_stats.json",
        training_state_statistics(arrays["observation.state"], manifest_episodes, split["train_episode_ids"]),
    )
    manifest["split"] = {**split, "path": "split.json", "method": "scripts.lpwm_ab.data.build_split"}
    manifest["scalar_sha256"] = {
        name: digest(root / name)
        for name in (
            "states.npy",
            "actions.npy",
            "task_index.npy",
            "source_task_index.npy",
            "language_embeddings.npy",
            "language_masks.npy",
            "language_manifest.json",
            "task_catalog.json",
            "source_manifest.json",
            "split.json",
            "state_stats.json",
        )
    }
    return manifest


def verify_cache(root, manifest, arrays):
    """Independently read every committed frame through production PackedImages."""
    verify_hashes(root, manifest["scalar_sha256"])
    for name, key in (
        ("states", "observation.state"),
        ("actions", "action"),
        ("task_index", "task_index"),
        ("source_task_index", "source_task_index"),
    ):
        require(np.array_equal(np.load(root / f"{name}.npy"), arrays[key]), f"Source row mismatch: {name}")
    split = read_json(root / "split.json")
    require(split == build_split(manifest, seed=42, validation_fraction=0.1), "Noncanonical split")
    images = PackedImages(root, manifest)
    verified_frames, verified_blocks = 0, 0
    try:
        for task in manifest["task_catalog"]["tasks"]:
            tid = task["global_task_id"]
            folder = root / f"task_{tid:03d}"
            record = read_json(folder / "complete.json")
            require(
                record["task"] == task and record["source_sha256"] == manifest["source"]["sha256"],
                "Wrong completed task",
            )
            verify_hashes(folder, record["files_sha256"])
            source_indices = np.load(folder / "source_indices.npy")
            require(
                np.array_equal(source_indices, np.flatnonzero(arrays["task_index"] == tid)),
                "Bad task source indices",
            )
            for name, key in (("states", "observation.state"), ("actions", "action")):
                require(
                    np.array_equal(np.load(folder / f"{name}.npy"), arrays[key][source_indices]),
                    "Bad task scalar alignment",
                )
            for block in read_json(folder / "blocks.json"):
                start, n = block["start"], block["frames"]
                pixels = images[start : start + n]
                require(
                    hashlib.sha256(pixels.tobytes()).hexdigest() == block["pixels_sha256"],
                    "PackedImages pixel mismatch",
                )
                verified_frames += n
                verified_blocks += 1
    finally:
        for handle in images.files.values():
            handle.close()
    require(verified_frames == manifest["num_frames"], "Incomplete image verification")
    stats = read_json(root / "state_stats.json")
    counts = {}
    for label, key in (("train", "train_episode_ids"), ("validation", "validation_episode_ids")):
        ids = set(split[key])
        eps = [ep for ep in manifest["episodes"] if ep["episode_index"] in ids]
        # Same anchor formula as FullCachedDataset, history2/action_horizon16.
        expected_windows = sum(ep["end"] - ep["start"] - 16 for ep in eps)
        dataset = FullCachedDataset(root, split[key], stats)
        require(len(dataset) == expected_windows, "Window count mismatch")
        ends = np.array([ep["end"] for ep in manifest["episodes"]])
        ep_indices = np.searchsorted(ends, dataset.anchors, side="right")
        starts = np.array([ep["start"] for ep in manifest["episodes"]])[ep_indices]
        require(
            np.all(dataset.anchors - 1 >= starts) and np.all(dataset.anchors + 16 <= ends[ep_indices]),
            "Window crosses episode",
        )
        require(set(ep_indices).issubset(ids), "Window leaks across split")
        counts[label] = {
            "episodes": len(eps),
            "frames": sum(ep["end"] - ep["start"] for ep in eps),
            "windows_h2_a16": len(dataset),
        }
    per_task = []
    for task in manifest["task_catalog"]["tasks"]:
        eps = [ep for ep in manifest["episodes"] if ep["task_index"] == task["global_task_id"]]
        per_task.append(
            {
                "task_index": task["global_task_id"],
                "suite": task["suite"],
                "suite_task_id": task["task_id"],
                "episodes": len(eps),
                "frames": sum(ep["end"] - ep["start"] for ep in eps),
                "train_episodes": sum(ep["episode_index"] in split["train_episode_ids"] for ep in eps),
                "validation_episodes": sum(
                    ep["episode_index"] in split["validation_episode_ids"] for ep in eps
                ),
            }
        )
    return {
        "status": "complete",
        "cpu_only": True,
        "tasks": len(per_task),
        "episodes": len(manifest["episodes"]),
        "frames": verified_frames,
        "camera_images": verified_frames * 2,
        "verified_blocks": verified_blocks,
        "all_source_scalars_exact": True,
        "all_pixels_verified_via_PackedImages": True,
        "image_orientation": "unchanged from source",
        "source_metadata_fps": manifest["fps"],
        "split_sha256": split["sha256"],
        "split": counts,
        "suites": dict(Counter(t["suite"] for t in per_task)),
        "per_task": per_task,
        "manifest_sha256": digest(root / "manifest.json"),
    }


def prepare(
    source,
    root,
    reference_root,
    *,
    workers=2,
    reserve_bytes=8 * 2**30,
    expected_counts=(1693, 273465, 40),
    expected_suites=SUITES,
):
    source, root, reference_root = Path(source), Path(root), Path(reference_root)
    require(1 <= workers <= 8, "Use 1..8 CPU workers")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        print("Auditing source scalar rows and SHA256 fingerprints (CPU only)", flush=True)
        info, tasks, episodes, arrays, files, snapshot = inspect_source(
            source, expected_counts=expected_counts
        )
        source_sha256 = object_digest(snapshot)
        reference = read_json(reference_root / "task_catalog.json")
        catalog = match_catalog(tasks, reference, expected_suites=expected_suites)
        mapping = {t["source_task_index"]: t["global_task_id"] for t in catalog["tasks"]}
        arrays["source_task_index"] = arrays["task_index"]
        lookup = np.array([mapping[i] for i in range(len(tasks))], dtype=np.int64)
        arrays["task_index"] = lookup[arrays["source_task_index"]]
        for ep in episodes:
            ep["source_task_index"] = ep["task_index"]
            ep["task_index"] = mapping[ep["task_index"]]
        # Do not overwrite metadata of a cache that belongs to a different source.
        if (root / "source_manifest.json").exists():
            require(
                read_json(root / "source_manifest.json") == snapshot,
                "Output belongs to a different source snapshot",
            )
        if (root / "task_catalog.json").exists():
            require(read_json(root / "task_catalog.json") == catalog, "Output belongs to a different catalog")
        if (root / "manifest.json").exists():
            manifest = read_json(root / "manifest.json")
            require(
                manifest["converter"] == CONVERTER
                and manifest["task_catalog"] == catalog
                and manifest["source"]["sha256"] == source_sha256,
                "Existing manifest contract mismatch",
            )
            report = verify_cache(root, manifest, arrays)
        else:
            atomic(root / "source_manifest.json", snapshot)
            atomic(root / "task_catalog.json", catalog)
            language = reuse_language(root, catalog, reference, reference_root / "cache")
            print(
                json.dumps(
                    {
                        "status": "source_audited",
                        "episodes": len(episodes),
                        "frames": info["total_frames"],
                        "tasks": len(tasks),
                        "parquet_files": len(files),
                        "suites": catalog["suites"],
                    }
                ),
                flush=True,
            )
            records = convert_images(
                root, catalog, episodes, arrays, files, source_sha256, workers, reserve_bytes
            )
            manifest = assemble(root, info, catalog, episodes, arrays, records, source_sha256, language)
            # FullCachedDataset needs this path. Status remains verifying until exhaustive checks pass.
            atomic(root / "preparation_status.json", {"status": "verifying", "time": time.time()})
            atomic(root / "manifest.json", manifest)
            report = verify_cache(root, manifest, arrays)
        atomic(root / "verification.json", report)
        atomic(
            root / "preparation_status.json",
            {
                "status": "complete",
                "complete_tasks": len(tasks),
                "tasks": len(tasks),
                "episodes": len(episodes),
                "frames": info["total_frames"],
                "cpu_only": True,
                "time": time.time(),
            },
        )
        print(json.dumps({k: v for k, v in report.items() if k != "per_task"}), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Read-only copy: task_catalog.json and cache/language_*"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    prepare(args.source, args.output, args.reference, workers=args.workers)


if __name__ == "__main__":
    main()
