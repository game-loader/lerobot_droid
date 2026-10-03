"""Lazy bounded file-backed lossless frame accessor; no video decoder in training."""

import os
import zlib
from pathlib import Path

import numpy as np

from scripts.lpwm_ab.data import LPWMCachedDataset


class PackedImages:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.shards = manifest["image_shards"]
        self.ends = np.array([s["end"] for s in self.shards])
        self.indices = []
        for shard in self.shards:
            with np.load(self.root / shard["index"]) as index:
                self.indices.append({key: np.array(index[key], copy=True) for key in index.files})
        self.pid = None
        self.files = {}
        self.blocks = {}
        self.shape = (manifest["num_frames"], 2, 3, 128, 128)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["files"] = {}
        d["blocks"] = {}
        d["pid"] = None
        return d

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            return np.stack([self[i] for i in range(start, stop, step)])
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.files = {}
            self.blocks = {}
        if key < 0 or key >= self.shape[0]:
            raise IndexError(key)
        shard = int(np.searchsorted(self.ends, key, side="right"))
        local = key - self.shards[shard]["start"]
        index = self.indices[shard]
        if shard not in self.files:
            if len(self.files) >= 24:
                _, old = self.files.popitem()
                old.close()
            self.files[shard] = (self.root / self.shards[shard]["path"]).open("rb", buffering=0)
        block_key = (shard, int(index["offsets"][local]))
        if block_key not in self.blocks:
            compressed = os.pread(self.files[shard].fileno(), int(index["lengths"][local]), block_key[1])
            raw = zlib.decompress(compressed)
            frames = int(index["block_frames"][local])
            if len(raw) != frames * 2 * 3 * 128 * 128:
                raise ValueError("Corrupt frame block size")
            pixels = np.bitwise_xor.accumulate(
                np.frombuffer(raw, dtype=np.uint8).reshape(frames, 2, 3, 128, 128), axis=0
            )
            if len(self.blocks) >= 8:
                self.blocks.pop(next(iter(self.blocks)))
            self.blocks[block_key] = pixels
        return self.blocks[block_key][int(index["frame_in_block"][local])]


class FullCachedDataset(LPWMCachedDataset):
    def __init__(
        self,
        root,
        episode_ids,
        state_stats,
        history=2,
        action_horizon=16,
        world_horizon=1,
        include_world=True,
    ):
        import json

        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.images = PackedImages(root, self.manifest)
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.task_index = np.load(self.root / "task_index.npy", mmap_mode="r")
        self.language = np.load(self.root / "language_embeddings.npy", mmap_mode="r")
        self.language_masks = np.load(self.root / "language_masks.npy", mmap_mode="r")
        self.language_lookup = {task: i for i, task in enumerate(self.manifest["language"]["task_ids"])}
        self.mean = np.asarray(state_stats["mean"], dtype=np.float32)
        self.std = np.asarray(state_stats["std"], dtype=np.float32)
        self.history = history
        self.action_horizon = action_horizon
        self.world_horizon = world_horizon
        self.include_world = include_world
        allowed = set(episode_ids)
        forward = max(action_horizon, world_horizon + 1)
        windows = [
            np.arange(ep["start"] + history - 1, ep["end"] - forward + 1, dtype=np.int64)
            for ep in self.manifest["episodes"]
            if ep["episode_index"] in allowed
        ]
        self.anchors = np.concatenate(windows) if windows else np.empty(0, dtype=np.int64)
        if not len(self.anchors):
            raise ValueError("No complete episode-safe windows")
