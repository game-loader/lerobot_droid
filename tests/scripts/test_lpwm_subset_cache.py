import json
import os

import numpy as np
import pytest

from scripts.lpwm_full.evaluate import SUITES
from scripts.lpwm_full.subset_cache import digest, subset_cache, write_json


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "full130"
    root.mkdir()
    tasks, episodes, shards, states, actions = [], [], [], [], []
    for suite, count in SUITES.items():
        for local in range(count):
            tid = len(tasks)
            folder = root / f"task_{tid:03d}"
            folder.mkdir()
            task = {
                "global_task_id": tid,
                "suite": suite,
                "task_id": local,
                "language": f"task{tid}",
                "name": f"task{tid}",
                "bytes": 100,
                "sha256": f"{tid:064x}",
            }
            tasks.append(task)
            state = np.full((4, 8), tid, dtype=np.float32)
            action = np.full((4, 7), -tid, dtype=np.float32)
            np.save(folder / "states.npy", state)
            np.save(folder / "actions.npy", action)
            (folder / "images.bin").write_bytes(b"fixture-image-" + str(tid).encode())
            np.savez(folder / "index.npz", offsets=np.zeros(4, dtype=np.int64))
            local_eps = [
                {"start": i * 2, "end": i * 2 + 2, "task_index": tid, "source_demo": f"demo_{i}"}
                for i in range(2)
            ]
            hashes = {
                name: digest(folder / name)
                for name in ["images.bin", "index.npz", "states.npy", "actions.npy"]
            }
            write_json(
                folder / "complete.json",
                {
                    "task": task,
                    "source_sha256": task["sha256"],
                    "frames": 4,
                    "episodes": local_eps,
                    "all_frame_lossless_roundtrip": True,
                    "files_sha256": hashes,
                },
            )
            for ep in local_eps:
                episodes.append(
                    dict(
                        ep,
                        start=tid * 4 + ep["start"],
                        end=tid * 4 + ep["end"],
                        episode_index=len(episodes),
                        suite=suite,
                        suite_task_id=local,
                    )
                )
            shards.append(
                {
                    "start": tid * 4,
                    "end": tid * 4 + 4,
                    "path": f"{folder.name}/images.bin",
                    "index": f"{folder.name}/index.npz",
                    "files_sha256": hashes,
                }
            )
            states.append(state)
            actions.append(action)
    lang = {
        "task_ids": list(reversed(range(130))),
        "hidden_dim": 3,
        "max_length": 2,
        "task_count": 130,
        "unique_texts": 130,
    }
    arrays = {
        "states": np.concatenate(states),
        "actions": np.concatenate(actions),
        "task_index": np.repeat(np.arange(130), 4),
        "language_embeddings": np.array([np.full((2, 3), i, np.float32) for i in lang["task_ids"]]),
        "language_masks": np.ones((130, 2), dtype=bool),
    }
    for name, arr in arrays.items():
        np.save(root / f"{name}.npy", arr)
    manifest = {
        "schema": "lpwm_libero_full_zlib_v1",
        "image_codec": "xor16_zlib_rgb128_v1",
        "num_frames": 520,
        "cameras": ["a", "b"],
        "image_size": 128,
        "task_catalog": {"tasks": tasks, "suites": list(SUITES)},
        "tasks": {str(t["global_task_id"]): t["language"] for t in tasks},
        "episodes": episodes,
        "image_shards": shards,
        "language": lang,
        "scalar_sha256": {f"{name}.npy": digest(root / f"{name}.npy") for name in arrays},
    }
    write_json(root / "manifest.json", manifest)
    return root


def test_subset_hardlinks_and_remaps_every_contract(source, tmp_path):
    hashes = {p.relative_to(source): digest(p) for p in source.rglob("*") if p.is_file()}
    result = subset_cache(source, tmp_path / "common40")
    out = tmp_path / "common40"
    assert result["num_frames"] == 160 and len(result["episodes"]) == 80
    assert list(result["tasks"]) == [str(i) for i in range(40)]
    assert "libero_90" not in result["task_catalog"]["suites"]
    assert result["task_catalog"]["tasks"][30]["suite"] == "libero_10"
    assert result["task_catalog"]["tasks"][30]["source_global_task_id"] == 120
    assert np.load(out / "states.npy")[120, 0] == 120
    assert np.load(out / "actions.npy")[120, 0] == -120
    assert np.load(out / "task_index.npy")[120] == 30
    assert np.load(out / "language_embeddings.npy")[30, 0, 0] == 120
    assert result["episodes"][60]["source_full_episode_index"] == 240
    assert result["episodes"][60]["start"] == 120
    assert result["episodes"][60]["task_index"] == 30
    assert os.path.samefile(source / "task_120/images.bin", out / "task_030/images.bin")
    assert not os.path.samefile(source / "task_120/complete.json", out / "task_030/complete.json")
    assert json.loads((out / "task_030/complete.json").read_text())["task"]["global_task_id"] == 30
    report = json.loads((out / "reuse_verification.json").read_text())
    assert (
        report["hardlinked_files"] == 160
        and report["image_bytes_copied"] == 0
        and report["network_downloads"] == 0
    )
    assert all(digest(source / path) == value for path, value in hashes.items())
    assert not (out / "task_040").exists()


@pytest.mark.parametrize("where", ["same", "child", "exists", "symlink_parent"])
def test_refuse_output_overwrite_or_nested_source(source, tmp_path, where):
    output = {
        "same": source,
        "child": source / "child",
        "exists": tmp_path / "exists",
        "symlink_parent": tmp_path / "alias" / "child",
    }[where]
    if where == "exists":
        output.mkdir()
    if where == "symlink_parent":
        (tmp_path / "alias").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="new output outside"):
        subset_cache(source, output)


@pytest.mark.parametrize("damage", ["image", "scalar", "episode", "identity", "language"])
def test_fail_closed_on_source_corruption(source, tmp_path, damage):
    if damage == "image":
        (source / "task_120/images.bin").write_bytes(b"corrupt")
    elif damage == "scalar":
        (source / "states.npy").write_bytes(b"corrupt")
    elif damage == "episode":
        p = source / "task_120/complete.json"
        d = json.loads(p.read_text())
        d["episodes"][1]["start"] = 1
        write_json(p, d)
    elif damage == "identity":
        p = source / "task_120/complete.json"
        d = json.loads(p.read_text())
        d["task"]["task_id"] = 9
        write_json(p, d)
    else:
        p = source / "manifest.json"
        d = json.loads(p.read_text())
        d["language"]["task_ids"][0] = d["language"]["task_ids"][1]
        write_json(p, d)
    with pytest.raises(ValueError):
        subset_cache(source, tmp_path / "common40")
    assert not (tmp_path / "common40/manifest.json").exists()
