"""Episode-safe, task-specific LeRobot decision windows without altering source data."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


def verified_manifest_labels(manifest, episodes):
    """Explicit opt-in only: caller has confirmed manifest_reward means task success."""
    if manifest.get("schema") != "franka_duo_tele_data.mcap_to_lerobot.rgb20d.v1":
        raise ValueError("Unsupported manifest schema; provide explicit episode labels instead")
    entries = manifest.get("episodes", [])
    if len(entries) != len(episodes):
        raise ValueError("Manifest and dataset episode counts differ")
    labels = {}
    # Source episode numbers may have gaps. Validate the converter's ordered mapping by length.
    for row, entry in zip(episodes, entries, strict=True):
        if int(row["length"]) != entry.get("stats", {}).get("frames_written"):
            raise ValueError("Manifest episode order/length does not match the dataset")
        provenance = entry.get("provenance", {})
        reward = provenance.get("manifest_reward")
        if (
            provenance.get("manifest_status") != "complete"
            or type(reward) not in (int, float)
            or reward not in (0, 1)
        ):
            raise ValueError("Missing complete binary task reward in manifest")
        labels[int(row["episode_index"])] = bool(reward)
    return labels


class SmolVLADecisionDataset(Dataset):
    def __init__(
        self,
        root,
        repo_id,
        *,
        chunk_size=32,
        gamma=0.99,
        labels_path=None,
        confirmed_manifest_rewards=False,
        task="pick cup and bowl",
        stride=None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if type(chunk_size) is not int or chunk_size < 1 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Invalid chunk size or discount")
        if bool(labels_path) == bool(confirmed_manifest_rewards):
            raise ValueError("Choose exactly one: --labels or --confirmed-manifest-rewards")
        self.source = LeRobotDataset(repo_id, root=Path(root), video_backend="pyav", return_uint8=True)
        self.chunk_size, self.gamma, self.task = chunk_size, gamma, task
        self.episodes = [dict(row) for row in self.source.meta.episodes]
        if labels_path:
            records = json.loads(Path(labels_path).read_text())["episodes"]
            self.labels = {}
            for record in records:
                index, success = record["episode_index"], record["success"]
                if type(index) is not int or type(success) is not bool or index in self.labels:
                    raise ValueError("Episode labels require unique integer indices and boolean success")
                self.labels[index] = success
            if set(self.labels) != {int(row["episode_index"]) for row in self.episodes}:
                raise ValueError("Labels must cover exactly all dataset episodes")
        else:
            manifest = json.loads((Path(root) / "franka_duo_extras/derived_manifest.json").read_text())
            self.labels = verified_manifest_labels(manifest, self.episodes)
        tasks = {int(i) for i in self.source.hf_dataset["task_index"]}
        task_table = self.source.meta.tasks
        actual_tasks = {str(task_table.index[task_table["task_index"] == i][0]) for i in tasks}
        if actual_tasks != {task}:
            raise ValueError(
                f"This runner is single-task; dataset tasks are {actual_tasks}, requested {task!r}"
            )
        stride = chunk_size if stride is None else stride
        if type(stride) is not int or not 1 <= stride <= chunk_size:
            raise ValueError("Decision stride must be in [1, execution_steps]")
        self.locations = []
        for row in self.episodes:
            start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            if end <= start:
                raise ValueError("Empty episode")
            for index in range(start, end, stride):
                self.locations.append((int(row["episode_index"]), index, end))

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        episode, start, end = self.locations[index]
        stop = min(start + self.chunk_size, end)
        count, terminal = stop - start, stop == end
        current = self.source[start]
        following = self.source[min(stop, end - 1)]
        keys = ["observation.state", *self.source.meta.camera_keys]
        rows = self.source.hf_dataset[start:stop]["action"]
        action = (
            rows.float()
            if isinstance(rows, torch.Tensor)
            else torch.stack([torch.as_tensor(row, dtype=torch.float32) for row in rows])
        )
        if action.ndim != 2 or not torch.isfinite(action).all():
            raise ValueError("Invalid dataset action values")
        padded = torch.zeros(self.chunk_size, action.shape[-1])
        padded[:count] = action
        return {
            "observation": {k: current[k] for k in keys},
            "next_observation": {k: following[k] for k in keys},
            "action": padded,
            "valid": torch.arange(self.chunk_size) < count,
            "reward": torch.tensor([float(terminal and self.labels[episode]) * self.gamma ** (count - 1)]),
            "done": torch.tensor([terminal]),
            "discount": torch.tensor([self.gamma**count]),
        }

    def split(self, seed=0, validation_fraction=0.2):
        episodes = sorted(self.labels)
        if len(episodes) < 2 or not 0 < validation_fraction < 1:
            raise ValueError("At least two episodes and a validation fraction in (0,1) are required")
        order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed)).tolist()
        nval = max(1, min(len(episodes) - 1, round(len(episodes) * validation_fraction)))
        held_out = {episodes[i] for i in order[:nval]}
        train = [i for i, (episode, _, _) in enumerate(self.locations) if episode not in held_out]
        val = [i for i, (episode, _, _) in enumerate(self.locations) if episode in held_out]
        return train, val, sorted(held_out)
