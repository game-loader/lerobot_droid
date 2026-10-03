"""Read existing lossless FR3 RGB caches as chunk-TD demonstrations; never decode videos."""

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import chunk_window


class CachedDemoChunkDataset(Dataset):
    """Use every episode; match the existing demo +1 terminal reward convention."""

    def __init__(self, cache, *, config, stride=1):
        """Open numeric arrays and frame offsets, leaving the existing RGB cache read-only."""
        self.root = Path(cache)
        self.config = config
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["status"] != "complete" or self.manifest["codec"] != "zlib1_native_rgb_uint8":
            raise ValueError("Expected an already completed lossless FR3 RGB cache")
        self.data = {
            key: np.load(self.root / f"{key}.npy", mmap_mode="r")
            for key in ("episode_index", "frame_index", "task_index", "action")
        }
        self.tasks = self.manifest["audit"]["tasks"]
        self.lookup, self.ends, self.rows = {}, {}, []
        for stream in self.manifest["streams"]:
            if stream["camera"] not in config.camera_keys:
                continue
            with np.load(self.root / stream["index"]) as index:
                rows = index["rows"]
                self.lookup[(stream["episode"], stream["camera"])] = (
                    stream["path"],
                    index["offsets"].copy(),
                    index["lengths"].copy(),
                )
                if stream["episode"] not in self.ends:
                    self.ends[stream["episode"]] = int(rows[-1]) + 1
                    self.rows.extend(rows[::stride].tolist())
        self.rows = np.sort(np.asarray(self.rows, dtype=np.int64))
        self._files = {}
        self._pid = None

    def __len__(self):
        """Return the number of eligible overlapping chunk starts."""
        return len(self.rows)

    def __getstate__(self):
        """Keep open file handles out of spawned workers."""
        return {**self.__dict__, "_files": {}, "_pid": None}

    def images_at(self, row):
        """Decompress selected cached RGB records only, in model camera order."""
        if self._pid != os.getpid():
            self._files = {}
            self._pid = os.getpid()
        episode = int(self.data["episode_index"][row])
        frame = int(self.data["frame_index"][row])
        images = []
        for key in self.config.camera_keys:
            name, offsets, lengths = self.lookup[(episode, key)]
            if name not in self._files:
                if len(self._files) >= 24:
                    _, old = self._files.popitem()
                    old.close()
                self._files[name] = (self.root / name).open("rb", buffering=0)
            raw = zlib.decompress(
                os.pread(self._files[name].fileno(), int(lengths[frame]), int(offsets[frame]))
            )
            shape = self.manifest["info"]["features"][key]["shape"]
            images.append(
                torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(shape).transpose(2, 0, 1).copy())
            )
        return torch.stack(images)

    def __getitem__(self, index):
        """Return episode-bounded current/next chunks with sparse demonstration rewards."""
        start = int(self.rows[index])
        end = self.ends[int(self.data["episode_index"][start])]
        h = self.config.chunk_size
        stop = min(start + 2 * h, end)
        actions = torch.from_numpy(np.array(self.data["action"][start:stop], dtype=np.float32))
        rewards = torch.zeros(stop - start)
        dones = torch.zeros(stop - start, dtype=torch.bool)
        if stop == end:
            rewards[-1] = 1.0
            dones[-1] = True
        sample = chunk_window(actions, rewards, dones, h)
        images = self.images_at(start)
        sample.update(
            images=images,
            next_images=images if sample["done"] else self.images_at(start + h),
            tasks=self.tasks[int(self.data["task_index"][start])],
        )
        return sample
