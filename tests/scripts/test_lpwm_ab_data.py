"""Check row/transition and split invariants without robot or external data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "lpwm_ab_data", Path(__file__).parents[2] / "scripts/lpwm_ab/data.py"
)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def make_cache(root):
    n = 48
    episodes = [
        {"episode_index": i, "start": i * 12, "end": (i + 1) * 12, "task_index": i // 2} for i in range(4)
    ]
    manifest = {
        "episodes": episodes,
        "cameras": ["observation.images.image", "observation.images.image2"],
        "language": {"task_ids": [0, 1]},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = {
        "images": np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None, None, None], (n, 2, 3, 4, 4)),
        "states": np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 8)),
        "actions": np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 7)),
        "task_index": np.repeat([0, 1], n // 2),
        "language_embeddings": np.ones((2, 3, 6), dtype=np.float32),
        "language_masks": np.ones((2, 3), dtype=bool),
    }
    for key, values in arrays.items():
        np.save(root / f"{key}.npy", values)
    return manifest, arrays


def test_split_is_episode_disjoint_reproducible_and_task_stratified(tmp_path):
    manifest, _ = make_cache(tmp_path)
    split = data.build_split(manifest, seed=42, validation_fraction=0.5)
    assert split == data.build_split(manifest, seed=42, validation_fraction=0.5)
    assert not set(split["train_episode_ids"]) & set(split["validation_episode_ids"])
    assert len(split["train_episode_ids"]) == len(split["validation_episode_ids"]) == 2
    assert {i // 2 for i in split["train_episode_ids"]} == {0, 1}


def test_train_only_statistics_ignore_validation(tmp_path):
    manifest, arrays = make_cache(tmp_path)
    stats = data.training_state_statistics(arrays["states"], manifest["episodes"], [0])
    np.testing.assert_allclose(stats["mean"], np.full(8, 5.5))
    assert stats["num_frames"] == 12


def test_action_and_next_frame_windows_never_cross_episodes(tmp_path):
    _, _ = make_cache(tmp_path)
    ds = data.LPWMCachedDataset(
        tmp_path, [0, 2], {"mean": [0.0] * 8, "std": [1.0] * 8}, history=2, action_horizon=4, world_horizon=2
    )
    assert np.array_equal(ds.anchors, np.concatenate((np.arange(1, 9), np.arange(25, 33))))
    for i, anchor in enumerate(ds.anchors):
        sample = ds[i]
        torch.testing.assert_close(sample["action"][:, 0], torch.arange(anchor, anchor + 4).float())
        torch.testing.assert_close(
            sample["world.actions"][:, 0], torch.arange(anchor - 1, anchor + 2).float()
        )
        torch.testing.assert_close(
            sample["world.images"][:, 0, 0, 0, 0] * 255, torch.arange(anchor - 1, anchor + 3).float()
        )
        assert sample["observation.state"].shape == (2, 8)
        assert not sample["action_is_pad"].any()


def test_a_variant_omits_future_images_and_actions(tmp_path):
    make_cache(tmp_path)
    ds = data.LPWMCachedDataset(
        tmp_path, [0], {"mean": [0.0] * 8, "std": [1.0] * 8}, history=2, action_horizon=4, include_world=False
    )
    assert not any(key.startswith("world.") for key in ds[0])


def test_bounded_validation_is_task_balanced(tmp_path):
    make_cache(tmp_path)
    ds = data.LPWMCachedDataset(
        tmp_path,
        [0, 2],
        {"mean": [0.0] * 8, "std": [1.0] * 8},
        history=2,
        action_horizon=4,
        include_world=False,
    )
    order = data.balanced_validation_order(ds, seed=42)
    assert sorted(order) == list(range(len(ds)))
    assert order == data.balanced_validation_order(ds, seed=42)
    assert set(ds.task_index[ds.anchors[order[:2]]]) == {0, 1}
