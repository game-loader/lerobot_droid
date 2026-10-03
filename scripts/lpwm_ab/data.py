"""Episode-safe LIBERO windows shared by paired single-GPU LPWM A/B jobs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_split(manifest: dict, seed: int, validation_fraction: float = 0.1) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    tasks = sorted({episode["task_index"] for episode in manifest["episodes"]})
    for task in tasks:
        episodes = sorted(
            episode["episode_index"] for episode in manifest["episodes"] if episode["task_index"] == task
        )
        if len(episodes) < 2:
            raise ValueError(f"Task {task} requires at least two episodes for held-out validation")
        episodes = np.asarray(episodes)[rng.permutation(len(episodes))]
        n_val = max(1, min(len(episodes) - 1, round(len(episodes) * validation_fraction)))
        validation.extend(episodes[:n_val].tolist())
        train.extend(episodes[n_val:].tolist())
    result = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_episode_ids": sorted(train),
        "validation_episode_ids": sorted(validation),
    }
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def training_state_statistics(states: np.ndarray, episodes: list[dict], train_ids: list[int]) -> dict:
    selected = set(train_ids)
    chunks = [
        np.asarray(states[episode["start"] : episode["end"]], dtype=np.float64)
        for episode in episodes
        if episode["episode_index"] in selected
    ]
    values = np.concatenate(chunks, axis=0)
    return {
        "mean": values.mean(0).astype(np.float32).tolist(),
        "std": np.maximum(values.std(0), 1e-6).astype(np.float32).tolist(),
        "source": "train episodes only",
        "num_frames": len(values),
    }


class LPWMCachedDataset(Dataset):
    """Preserve action/next-frame alignment and never sample across episode boundaries."""

    def __init__(
        self,
        root: str | Path,
        episode_ids: list[int],
        state_stats: dict,
        history: int = 2,
        action_horizon: int = 16,
        world_horizon: int = 1,
        include_world: bool = True,
    ):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.images = np.load(self.root / "images.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.task_index = np.load(self.root / "task_index.npy", mmap_mode="r")
        self.language = np.load(self.root / "language_embeddings.npy", mmap_mode="r")
        self.language_masks = np.load(self.root / "language_masks.npy", mmap_mode="r")
        self.language_lookup = {
            task: index for index, task in enumerate(self.manifest["language"]["task_ids"])
        }
        self.mean = np.asarray(state_stats["mean"], dtype=np.float32)
        self.std = np.asarray(state_stats["std"], dtype=np.float32)
        self.history, self.action_horizon, self.world_horizon = history, action_horizon, world_horizon
        self.include_world = include_world
        allowed = set(episode_ids)
        forward_required = max(action_horizon, world_horizon + 1)
        windows = [
            np.arange(episode["start"] + history - 1, episode["end"] - forward_required + 1, dtype=np.int64)
            for episode in self.manifest["episodes"]
            if episode["episode_index"] in allowed
        ]
        self.anchors = np.concatenate(windows) if windows else np.empty(0, dtype=np.int64)
        if not len(self.anchors):
            raise ValueError("No complete training windows in selected episodes")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        t = int(self.anchors[index])
        start = t - self.history + 1
        observation_images = (
            torch.from_numpy(np.array(self.images[start : t + 1], copy=True)).float().div_(255)
        )
        state = (np.array(self.states[start : t + 1], copy=True) - self.mean) / self.std
        language_row = self.language_lookup[int(self.task_index[t])]
        sample = {
            "observation.state": torch.from_numpy(state),
            "observation.language.embedding": torch.from_numpy(
                np.array(self.language[language_row], copy=True)
            ),
            "observation.language.attention_mask": torch.from_numpy(
                np.array(self.language_masks[language_row], copy=True)
            ),
            "action": torch.from_numpy(np.array(self.actions[t : t + self.action_horizon], copy=True)),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
        }
        sample.update(
            {key: observation_images[:, camera] for camera, key in enumerate(self.manifest["cameras"])}
        )
        if self.include_world:
            end = t + self.world_horizon + 1
            sample["world.images"] = (
                torch.from_numpy(np.array(self.images[start:end], copy=True)).float().div_(255)
            )
            sample["world.actions"] = torch.from_numpy(np.array(self.actions[start : end - 1], copy=True))
            sample["world.states"] = torch.from_numpy(
                (np.array(self.states[start:end], copy=True) - self.mean) / self.std
            )
        return sample


def balanced_validation_order(dataset: LPWMCachedDataset, seed: int) -> list[int]:
    """Interleave held-out tasks so a bounded validation budget covers every task."""
    rng = np.random.default_rng(seed)
    task_ids = np.asarray(dataset.task_index[dataset.anchors])
    groups = [
        rng.permutation(np.flatnonzero(task_ids == task)).tolist() for task in sorted(np.unique(task_ids))
    ]
    return [
        group[offset] for offset in range(max(map(len, groups))) for group in groups if offset < len(group)
    ]
