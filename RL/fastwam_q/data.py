"""Episode-safe chunk replay over LeRobot demonstrations and executed rollout actions."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, default_collate

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import NormalizerProcessorStep
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def as_tensor(values, dtype=torch.float32):
    """Convert dataset columns into a stacked tensor without changing row order."""
    if isinstance(values, torch.Tensor):
        return values.to(dtype)
    return torch.stack([torch.as_tensor(v) for v in values]).to(dtype)


def chunk_window(actions, rewards, dones, h):
    """Keep the terminal reward, repeat-pad actions, and mask padded steps (never cross episodes)."""
    count = min(h, len(actions))
    terminal = torch.where(dones[:count])[0]
    if len(terminal):
        count = int(terminal[0]) + 1
    done = bool(len(terminal)) or count < h
    first = actions[:count]
    action = torch.cat([first, first[-1:].expand(h - count, -1)])
    valid = torch.arange(h) < count
    reward = torch.zeros(h)
    reward[:count] = rewards[:count]
    following = actions[h : 2 * h] if not done else actions[:0]
    next_valid = torch.arange(h) < len(following)
    next_action = torch.zeros_like(action)
    if len(following):
        next_action[: len(following)] = following
        next_action[len(following) :] = following[-1]
    return {
        "action_chunk": action,
        "next_action_chunk": next_action,
        "rewards": reward,
        "valid": valid,
        "next_valid": next_valid,
        "done": torch.tensor(done),
    }


class LeRobotChunkDataset(Dataset):
    """Start offsets may overlap; each sample still evaluates one contiguous H-step action chunk.

    Demonstrations may use a +1 final reward. Rollouts need next.reward/done/truncated
    or a JSON mapping {"episode_index": success}; they are never assumed successful.
    """

    def __init__(self, repo_id, root=None, *, config, demonstrations=False, labels=None, stride=1):
        """Construct the component from its configuration and supplied dependencies."""
        require_package("datasets", extra="fastwam-q")
        self.source = LeRobotDataset(
            repo_id, root=Path(root) if root else None, video_backend="pyav", return_uint8=True
        )
        self.config, self.stride = config, stride
        self.demonstrations = demonstrations
        self.labels = json.loads(Path(labels).read_text()) if labels else None
        self.episodes = list(self.source.meta.episodes)
        self.ends = []
        total = 0
        for episode in self.episodes:
            length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
            total += (length + stride - 1) // stride
            self.ends.append(total)
        columns = self.source.hf_dataset.column_names
        self.has_rewards = "next.reward" in columns
        self.done_columns = [key for key in ("next.done", "next.truncated") if key in columns]
        if not demonstrations and not self.has_rewards and self.labels is None:
            raise ValueError("Online rollout data needs next.reward or episode success labels")

    def __len__(self):
        """Return the number of available replay samples."""
        return self.ends[-1]

    def __getitem__(self, index):
        """Read one episode-bounded training sample."""
        episode_index = bisect.bisect_right(self.ends, index)
        episode = self.episodes[episode_index]
        offset = index - (self.ends[episode_index - 1] if episode_index else 0)
        start = int(episode["dataset_from_index"]) + offset * self.stride
        end = int(episode["dataset_to_index"])
        h = self.config.chunk_size
        stop = min(start + 2 * h, end)
        rows = self.source.hf_dataset[start:stop]
        actions = as_tensor(rows["action"])
        rewards = (
            as_tensor(rows["next.reward"]).reshape(-1) if self.has_rewards else torch.zeros(stop - start)
        )
        dones = torch.zeros(stop - start, dtype=torch.bool)
        for key in self.done_columns:
            dones |= as_tensor(rows[key], torch.bool).reshape(-1)
        if stop == end:
            dones[-1] = True  # finite-horizon episode boundary, including timeouts
            if not self.has_rewards:
                success = (
                    self.demonstrations if self.labels is None else self.labels[str(episode["episode_index"])]
                )
                rewards[-1] = float(success)
        sample = chunk_window(actions, rewards, dones, h)
        current = self.source[start]
        following = current if sample["done"] else self.source[start + h]
        sample.update(
            images=torch.stack([current[k] for k in self.config.camera_keys]),
            next_images=torch.stack([following[k] for k in self.config.camera_keys]),
            tasks=current["task"],
        )
        return sample


class FastWAMQReplay(Dataset):
    """CPU episode replay. Store actual executed primitive actions, not discarded planned suffixes."""

    def __init__(self, config, capacity_episodes=1000, stride=1):
        """Construct the component from its configuration and supplied dependencies."""
        self.config, self.capacity_episodes, self.stride = config, capacity_episodes, stride
        self.episodes, self.locations = [], []

    def add_episode(self, episode):
        # images [T,V,C,H,W], actions [T,A], rewards/done [T], task string.
        """Append a CPU copy of an executed episode, retaining its terminal transition."""
        episode = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in episode.items()
        }
        episode["done"] = episode.get("done", torch.zeros(len(episode["actions"]), dtype=torch.bool)).bool()
        episode["done"][-1] = True
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.capacity_episodes :]
        self.locations = [
            (i, start)
            for i, ep in enumerate(self.episodes)
            for start in range(0, len(ep["actions"]), self.stride)
        ]

    def __len__(self):
        """Return the number of available replay samples."""
        return len(self.locations)

    def __getitem__(self, index):
        """Read one episode-bounded training sample."""
        i, start = self.locations[index]
        episode = self.episodes[i]
        h = self.config.chunk_size
        sample = chunk_window(
            episode["actions"][start : start + 2 * h],
            episode["rewards"][start : start + 2 * h],
            episode["done"][start : start + 2 * h],
            h,
        )
        sample.update(
            images=episode["images"][start],
            next_images=episode["images"][start if sample["done"] else start + h],
            tasks=episode["task"],
        )
        return sample


def sample_mixed(demos, online, batch_size, generator=None):
    """Exactly half online after rollouts exist; sampling with replacement supports small buffers."""
    n_online = batch_size // 2 if online is not None and len(online) else 0
    samples = []
    for dataset, count in ((demos, batch_size - n_online), (online, n_online)):
        if count:
            indices = torch.randint(len(dataset), (count,), generator=generator).tolist()
            samples.extend(dataset[i] for i in indices)
    return default_collate(samples)


class ActionNormalizer:
    """Reuse the FastWAM checkpoint's exact LeRobot action normalization, not rollout statistics."""

    def __init__(self, step):
        """Construct the component from its configuration and supplied dependencies."""
        self.step = step

    @classmethod
    def from_processor(cls, preprocessor):
        """Select the action normalizer already saved with the BC processor."""
        return cls(next(s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)))

    def __call__(self, actions):
        """Apply the saved BC action transform without normalizing images."""
        self.step.to(device=actions.device, dtype=torch.float32)
        return self.step({TransitionKey.ACTION: actions})[TransitionKey.ACTION]

    def state_dict(self):
        """Serialize the action transform configuration and statistics."""
        return {"config": self.step.get_config(), "state": self.step.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild the saved action transform without estimating new statistics."""
        if state.get("kind") == "fastwam_bundle_global_zscore_v1":
            from .bundle import BundleActionNormalizer

            return BundleActionNormalizer(state["mean"], state["std"])
        step = NormalizerProcessorStep(**state["config"])
        step.load_state_dict(state["state"])
        return cls(step)
