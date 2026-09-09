"""Raw RGB caches for the local Parquet reader, indexed by absolute frame index."""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .video_utils import decode_video_frames


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class DiskCameraFrameCache:
    """Picklable read-only cache; each process lazily opens its own memory maps."""

    def __init__(self, root: Path, manifest: dict):
        """Bind a completed cache to its validated manifest."""
        self.root = root
        self.manifest = manifest
        self._positions = {int(index): i for i, index in enumerate(manifest["indices"])}
        self._arrays = {}

    def __getstate__(self):
        """Do not pickle memory-map contents or inherited file handles."""
        return {**self.__dict__, "_arrays": {}}

    def __len__(self):
        """Return the number of distinct cached dataset rows."""
        return len(self._positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return private writable copies so transforms cannot mutate the cache."""
        position = self._positions[index]
        result = {}
        for key, spec in self.manifest["arrays"].items():
            if key not in self._arrays:
                self._arrays[key] = np.load(self.root / spec["file"], mmap_mode="r", allow_pickle=False)
            result[key] = torch.from_numpy(self._arrays[key][position].copy())
        return result


def _cache_inputs(reader):
    if reader.hf_dataset is None:
        reader.load_and_activate()
    rows = reader.hf_dataset.select_columns(["index", "episode_index", "timestamp"]).with_format("numpy")[:]
    keys = sorted(set(reader._meta.video_keys) - set(reader._meta.depth_keys))
    videos = []
    for episode in np.unique(rows["episode_index"]):
        ep = reader._meta.episodes[int(episode)]
        for key in keys:
            path = (reader.root / reader._meta.get_video_file_path(int(episode), key)).resolve()
            stat = path.stat()
            videos.append(
                {
                    "episode": int(episode),
                    "key": key,
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "offset": float(ep[f"videos/{key}/from_timestamp"]),
                    "from_index": int(ep["dataset_from_index"]),
                    "to_index": int(ep["dataset_to_index"]),
                }
            )
    row_hash = hashlib.sha256()
    for name in ("index", "episode_index", "timestamp"):
        row_hash.update(np.ascontiguousarray(rows[name]).tobytes())
    inputs = {
        "version": 1,
        "backend": reader._video_backend,
        "tolerance_s": reader._tolerance_s,
        "rows": row_hash.hexdigest(),
        "videos": videos,
        "features": {key: reader._meta.features[key] for key in keys},
    }
    inputs = json.loads(json.dumps(inputs, sort_keys=True))
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return rows, keys, inputs, fingerprint


def _decoded_batches(reader, rows, keys, batch_size=256):
    for episode in np.unique(rows["episode_index"]):
        positions = np.flatnonzero(rows["episode_index"] == episode)
        ep = reader._meta.episodes[int(episode)]
        for key in keys:
            offset = float(ep[f"videos/{key}/from_timestamp"])
            path = reader.root / reader._meta.get_video_file_path(int(episode), key)
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                timestamps = [offset + float(ts) for ts in rows["timestamp"][selected]]
                frames = decode_video_frames(
                    path, timestamps, reader._tolerance_s, reader._video_backend, return_uint8=True
                )
                if frames.dtype != torch.uint8 or frames.ndim != 4 or len(frames) != len(selected):
                    raise ValueError("RGB decoder returned an incompatible camera cache batch")
                yield key, selected, frames


def _valid_manifest(root, inputs, indices, keys):
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("arrays"), dict)
            or manifest["inputs"] != inputs
            or manifest["indices"] != indices
            or set(manifest["arrays"]) != set(keys)
        ):
            return None
        for spec in manifest["arrays"].values():
            path = root / spec["file"]
            if path.parent != root or _file_digest(path) != spec["sha256"]:
                return None
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                list(array.shape) != spec["shape"]
                or array.dtype != np.uint8
                or array.ndim != 4
                or array.shape[0] != len(indices)
                or array.shape[1] != 3
            ):
                return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def preload_camera_cache(reader, *, disk: bool = False):
    """Decode selected RGB rows with canonical video offsets and publish only complete caches."""
    rows, keys, inputs, fingerprint = _cache_inputs(reader)
    if not keys:
        return {}, {"camera_cache_frames": 0, "camera_cache_bytes": 0.0}
    indices = [int(index) for index in rows["index"]]
    if not disk:
        cache = {index: {} for index in indices}
        total_bytes = 0
        for key, positions, frames in _decoded_batches(reader, rows, keys):
            for position, frame in zip(positions, frames, strict=True):
                cache[indices[position]][key] = frame.clone()
            total_bytes += frames.numel()
        return cache, {"camera_cache_frames": len(cache), "camera_cache_bytes": float(total_bytes)}

    parent = (
        Path(os.environ.get("LEROBOT_CAMERA_CACHE_DIR", reader.root / ".camera_frame_cache"))
        .expanduser()
        .resolve()
    )
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / fingerprint
    with FileLock(str(parent / f"{fingerprint}.lock")):
        manifest = _valid_manifest(destination, inputs, indices, keys)
        if manifest is None:
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
            try:
                arrays = {}
                specifications = {}
                for key, positions, frames in _decoded_batches(reader, rows, keys):
                    if key not in arrays:
                        filename = f"camera-{len(arrays)}.npy"
                        shape = (len(indices), *frames.shape[1:])
                        arrays[key] = np.lib.format.open_memmap(
                            temporary / filename, mode="w+", dtype=np.uint8, shape=shape
                        )
                        specifications[key] = {"file": filename, "shape": list(shape)}
                    arrays[key][positions] = frames.numpy()
                for key, array in arrays.items():
                    array.flush()
                    specifications[key]["sha256"] = _file_digest(temporary / specifications[key]["file"])
                arrays.clear()
                manifest = {"inputs": inputs, "indices": indices, "arrays": specifications}
                (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
                if destination.exists():
                    destination.rename(parent / f"{fingerprint}.invalid-{uuid.uuid4().hex}")
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    size = sum(int(np.prod(spec["shape"])) for spec in manifest["arrays"].values())
    return DiskCameraFrameCache(destination, manifest), {
        "camera_cache_frames": len(indices),
        "camera_cache_bytes": float(size),
    }
