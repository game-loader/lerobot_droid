#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os

import datasets
import torch

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .video_utils import decode_video_frames


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        camera_frame_cache: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms
        self._return_uint8 = return_uint8
        # Optional decoded RGB frame cache.  The cache intentionally stores
        # raw uint8 CHW tensors below the image-transform/model pipeline so
        # callers can still update transforms or train the observation encoder.
        self._camera_frame_cache = camera_frame_cache

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
        *,
        abs_idx: int | None = None,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        # A preloaded cache avoids repeatedly seeking/decoding PyAV frames.
        # Indexes in ``query_indices`` are absolute dataset indexes, matching
        # the keys populated by :meth:`preload_camera_frame_cache`.
        if self._camera_frame_cache is not None:
            if abs_idx is None:
                raise ValueError("abs_idx is required when using a camera frame cache")
            cached: dict[str, torch.Tensor] = {}
            for vid_key in self._meta.video_keys:
                indices = (
                    query_indices.get(vid_key, [abs_idx])
                    if query_indices is not None
                    else [abs_idx]
                )
                frames: list[torch.Tensor] = []
                for frame_idx in indices:
                    try:
                        frame = self._camera_frame_cache[frame_idx][vid_key]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"camera frame cache is missing index={frame_idx} key={vid_key!r}"
                        ) from exc
                    # Keep the cache itself raw uint8, but honor the reader's
                    # public ``return_uint8`` contract when serving samples.
                    if self._return_uint8:
                        frames.append(frame)
                    else:
                        frames.append(frame.to(dtype=torch.float32).div_(255.0))
                cached[vid_key] = torch.stack(frames)
            return cached

        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
            )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera: decode in parallel (video decoding releases the GIL)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(
                query_timestamps,
                ep_idx,
                abs_idx=abs_idx,
                query_indices=query_indices,
            )
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # add subtask information if available
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

        return item

    def preload_camera_frame_cache(self) -> dict[str, int | float]:
        """Decode all selected camera videos once and keep raw RGB frames in RAM.

        The cache is keyed by absolute dataset frame index and stores CHW
        ``torch.uint8`` tensors.  Decoding is performed sequentially per video
        (rather than seeking once per frame), which removes the dominant PyAV
        overhead during image-heavy IL training.  The cache is deliberately
        below image transforms and policy encoders, so normalization/crops and
        observation-encoder parameters remain trainable and configurable.
        """

        if not self._meta.video_keys:
            self._camera_frame_cache = {}
            return {"camera_cache_frames": 0, "camera_cache_bytes": 0.0}
        if self._camera_frame_cache is not None and self._camera_frame_cache:
            total_bytes = sum(
                value.numel() * value.element_size()
                for frame in self._camera_frame_cache.values()
                for value in frame.values()
            )
            return {
                "camera_cache_frames": len(self._camera_frame_cache),
                "camera_cache_bytes": float(total_bytes),
            }
        if self.hf_dataset is None:
            self.load_and_activate()

        # ``av`` is imported lazily because datasets without videos should not
        # require the optional PyAV dependency at import time.
        import av
        import numpy as np

        selected_episodes = (
            tuple(self.episodes)
            if self.episodes is not None
            else tuple(range(self._meta.total_episodes))
        )
        cache: dict[int, dict[str, torch.Tensor]] = {}
        total_bytes = 0

        for ep_idx in selected_episodes:
            episode = self._meta.episodes[ep_idx]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            length = stop - start
            for vid_key in self._meta.video_keys:
                video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                frames: list[torch.Tensor] = []
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    for frame in container.decode(stream):
                        # Convert to CHW uint8, matching decode_video_frames.
                        arr = frame.to_ndarray(format="rgb24")
                        if not isinstance(arr, np.ndarray):
                            arr = np.asarray(arr)
                        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                        frames.append(tensor)
                if len(frames) < length:
                    raise RuntimeError(
                        f"camera video has fewer frames than metadata: key={vid_key!r}, "
                        f"episode={ep_idx}, decoded={len(frames)}, expected={length}, path={video_path}"
                    )
                if len(frames) > length:
                    # Encoders may leave a trailing frame; metadata remains
                    # authoritative for the frame-to-index mapping.
                    frames = frames[:length]
                for offset, frame in enumerate(frames):
                    frame_index = start + offset
                    cache.setdefault(frame_index, {})[vid_key] = frame
                    total_bytes += frame.numel() * frame.element_size()

        self._camera_frame_cache = cache
        return {
            "camera_cache_frames": len(cache),
            "camera_cache_bytes": float(total_bytes),
        }

    def preload_camera_frame_cache_disk(self) -> dict[str, int | float]:
        """Decode camera videos once into a disk-backed memmap cache.

        This is equivalent to :meth:`preload_camera_frame_cache` but avoids
        retaining tens of gigabytes of RGB tensors in process RAM on large
        real-robot datasets.  The cache is keyed by absolute dataset index and
        is reusable across subsequent IL/offline-RL launches.
        """
        if not self._meta.video_keys:
            self._camera_frame_cache = {}
            return {"camera_cache_frames": 0, "camera_cache_bytes": 0.0}
        if self.hf_dataset is None:
            self.load_and_activate()

        cache_root = os.environ.get("LEROBOT_CAMERA_CACHE_DIR")
        cache_dir = Path(cache_root).expanduser() if cache_root else self.root / ".camera_frame_cache"
        # Keep datasets immutable/read-only when a shared external cache root
        # is configured; namespace by dataset path to avoid collisions.
        if cache_root:
            import hashlib

            namespace = hashlib.sha1(str(self.root.resolve()).encode("utf-8")).hexdigest()[:16]
            cache_dir = cache_dir / namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"

        class _DiskCache:
            def __init__(self, root: Path, manifest: Mapping[str, object]):
                self.root = root
                self.keys = tuple(str(k) for k in manifest["keys"])
                self.total_frames = int(manifest["total_frames"])
                self.shapes = {str(k): tuple(int(x) for x in v) for k, v in manifest["shapes"].items()}  # type: ignore[union-attr]
                self.arrays = {
                    key: np.memmap(
                        root / f"{key.replace('/', '_')}.mmap",
                        mode="r",
                        dtype=np.uint8,
                        shape=(self.total_frames, *self.shapes[key]),
                    )
                    for key in self.keys
                }

            def __bool__(self) -> bool:
                return True

            def __getitem__(self, frame_idx: int) -> dict[str, torch.Tensor]:
                index = int(frame_idx)
                return {
                    key: torch.from_numpy(self.arrays[key][index])
                    for key in self.keys
                }

            def summary(self) -> dict[str, int | float]:
                bytes_total = sum(int(np.prod(shape)) * self.total_frames for shape in self.shapes.values())
                return {"camera_cache_frames": self.total_frames, "camera_cache_bytes": float(bytes_total)}

        import av
        import numpy as np

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("total_frames") == self._meta.total_frames
                    and tuple(manifest.get("keys", ())) == tuple(self._meta.video_keys)
                    and all((cache_dir / f"{str(k).replace('/', '_')}.mmap").is_file() for k in manifest["keys"])
                ):
                    cache_obj = _DiskCache(cache_dir, manifest)
                    self._camera_frame_cache = cache_obj
                    return cache_obj.summary()
            except (OSError, ValueError, KeyError, TypeError):
                pass

        selected_episodes = (
            tuple(self.episodes)
            if self.episodes is not None
            else tuple(range(self._meta.total_episodes))
        )
        total_frames = int(self._meta.total_frames)
        arrays: dict[str, np.memmap] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for ep_idx in selected_episodes:
            episode = self._meta.episodes[ep_idx]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            length = stop - start
            for vid_key in self._meta.video_keys:
                video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    for offset, frame in enumerate(container.decode(stream)):
                        if offset >= length:
                            break
                        arr = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8).transpose(2, 0, 1)
                        if vid_key not in arrays:
                            shapes[vid_key] = tuple(arr.shape)
                            arrays[vid_key] = np.memmap(
                                cache_dir / f"{vid_key.replace('/', '_')}.mmap",
                                mode="w+",
                                dtype=np.uint8,
                                shape=(total_frames, *arr.shape),
                            )
                        arrays[vid_key][start + offset] = arr
                    arrays[vid_key].flush()
        manifest = {"total_frames": total_frames, "keys": list(self._meta.video_keys), "shapes": {k: list(v) for k, v in shapes.items()}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        cache_obj = _DiskCache(cache_dir, manifest)
        self._camera_frame_cache = cache_obj
        return cache_obj.summary()
