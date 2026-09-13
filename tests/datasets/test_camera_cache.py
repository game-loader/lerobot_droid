"""RGB cache equivalence across shared video files, episode views, and worker processes."""

import json
import multiprocessing
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av", reason="Camera cache video tests require lerobot[dataset]")
datasets = pytest.importorskip("datasets", reason="Camera cache tests require lerobot[dataset]")
pd = pytest.importorskip("pandas", reason="Camera cache metadata tests require lerobot[dataset]")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.datasets.camera_cache import DiskCameraFrameCache  # noqa: E402
from lerobot.datasets.dataset_reader import DatasetReader  # noqa: E402
from lerobot.datasets.io_utils import hf_transform_to_torch  # noqa: E402

KEY = "observation.images.wrist"


def _video_path(episode, key):
    return Path("shared.mp4")


def _worker_read(cache, connection):
    connection.send(int(cache[3][KEY].sum()))
    connection.close()


@pytest.fixture
def reader_factory(tmp_path):
    with av.open(str(tmp_path / "shared.mp4"), mode="w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(6):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 35, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    def make(*, episodes=None, history=False, uint8=True, transform=None):
        meta = SimpleNamespace(
            total_episodes=2,
            total_frames=6,
            fps=10,
            video_keys=[KEY],
            camera_keys=[KEY],
            depth_keys=[],
            image_keys=[],
            features={KEY: {"dtype": "video", "shape": (16, 16, 3)}},
            episodes=[
                {"dataset_from_index": 0, "dataset_to_index": 3, f"videos/{KEY}/from_timestamp": 0.0},
                {"dataset_from_index": 3, "dataset_to_index": 6, f"videos/{KEY}/from_timestamp": 0.3},
            ],
            get_video_file_path=_video_path,
            tasks=pd.DataFrame({"task_index": [0]}, index=["test"]),
        )
        reader = DatasetReader(
            meta,
            tmp_path,
            episodes,
            1e-4,
            "pyav",
            {KEY: [-0.1, 0.0, 0.1]} if history else None,
            transform,
            return_uint8=uint8,
        )
        rows = [i for i in range(6) if episodes is None or i // 3 in episodes]
        reader.hf_dataset = datasets.Dataset.from_dict(
            {
                "index": rows,
                "episode_index": [i // 3 for i in rows],
                "frame_index": [i % 3 for i in rows],
                "timestamp": np.array([i % 3 / 10 for i in rows], np.float32),
                "task_index": [0] * len(rows),
            }
        )
        reader.hf_dataset.set_transform(hf_transform_to_torch)
        reader._build_index_mapping()
        return reader

    return make


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("episodes", [None, [1]])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("uint8", [False, True])
def test_cache_matches_native_shared_video(reader_factory, disk, episodes, history, uint8):
    reader = reader_factory(episodes=episodes, history=history, uint8=uint8)
    expected = [reader.get_item(i) for i in range(reader.num_frames)]
    summary = reader.preload_camera_frame_cache_disk() if disk else reader.preload_camera_frame_cache()
    assert summary["camera_cache_frames"] == len(expected)
    assert summary["camera_cache_bytes"] == len(expected) * 3 * 16 * 16
    if episodes == [1]:
        with pytest.raises(KeyError):
            reader._camera_frame_cache[0]
    for i, item in enumerate(expected):
        actual = reader.get_item(i)
        torch.testing.assert_close(actual[KEY], item[KEY], rtol=0, atol=0)
        if history:
            assert torch.equal(actual[f"{KEY}_is_pad"], item[f"{KEY}_is_pad"])


@pytest.mark.parametrize("disk", [False, True])
def test_transforms_cannot_mutate_raw_cache(reader_factory, disk):
    reader = reader_factory()
    original = reader.get_item(3)[KEY].clone()
    if disk:
        reader.preload_camera_frame_cache_disk()
    else:
        reader.preload_camera_frame_cache()
    reader.set_image_transforms(lambda frame: frame.zero_())
    assert reader.get_item(3)[KEY].sum() == 0
    reader.clear_image_transforms()
    torch.testing.assert_close(reader.get_item(3)[KEY], original, rtol=0, atol=0)


def test_disk_reuse_corruption_and_spawn(reader_factory, monkeypatch):
    reader = reader_factory(episodes=[1])
    reader.preload_camera_frame_cache_disk()
    cache = reader._camera_frame_cache
    assert isinstance(cache, DiskCameraFrameCache)
    expected = int(cache[3][KEY].sum())
    assert cache._arrays
    assert not pickle.loads(pickle.dumps(cache))._arrays
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker_read, args=(cache, send))
    process.start()
    send.close()
    try:
        assert receive.poll(30), "Spawned cache reader did not return"
        assert receive.recv() == expected
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receive.close()
    with monkeypatch.context() as patch:
        patch.setattr(
            "lerobot.datasets.camera_cache.decode_video_frames",
            lambda *args, **kwargs: pytest.fail("cache not reused"),
        )
        reader_factory(episodes=[1]).preload_camera_frame_cache_disk()
    path = cache.root / cache.manifest["arrays"][KEY]["file"]
    path.write_bytes(b"corrupt")
    repaired = reader_factory(episodes=[1])
    repaired.preload_camera_frame_cache_disk()
    assert int(repaired._camera_frame_cache[3][KEY].sum()) == expected
    assert list(cache.root.parent.glob("*.invalid-*"))


@pytest.mark.parametrize(
    "change", ["offset", "timestamp", "video_stat", "episodes", "manifest_indices", "manifest_keys"]
)
def test_disk_fingerprint_and_manifest_validation(reader_factory, change):
    reader = reader_factory()
    reader.preload_camera_frame_cache_disk()
    original = reader._camera_frame_cache.root
    other = reader_factory(episodes=[1] if change == "episodes" else None)
    if change == "offset":
        other._meta.episodes[1][f"videos/{KEY}/from_timestamp"] = 0.2
    elif change == "timestamp":
        other.hf_dataset = other.hf_dataset.map(
            lambda row: {"timestamp": max(0.0, row["timestamp"].item() - 0.1)}
        )
    elif change == "video_stat":
        (other.root / "shared.mp4").touch()
    elif change.startswith("manifest"):
        path = original / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["indices" if change == "manifest_indices" else "arrays"] = (
            [] if change == "manifest_indices" else {}
        )
        path.write_text(json.dumps(manifest))
    other.preload_camera_frame_cache_disk()
    if change.startswith("manifest"):
        assert other._camera_frame_cache.root == original
        assert other._camera_frame_cache.manifest["indices"] == list(range(6))
        assert set(other._camera_frame_cache.manifest["arrays"]) == {KEY}
    else:
        assert other._camera_frame_cache.root != original


def test_failed_build_is_not_published(reader_factory, monkeypatch):
    reader = reader_factory()

    def fail(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr("lerobot.datasets.camera_cache.decode_video_frames", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        reader.preload_camera_frame_cache_disk()
    assert reader._camera_frame_cache is None
    assert not list((reader.root / ".camera_frame_cache").glob("*/manifest.json"))
    assert not list((reader.root / ".camera_frame_cache").glob(".building-*"))


def test_depth_bypasses_rgb_cache(reader_factory, monkeypatch):
    reader = reader_factory()
    reader._meta.depth_keys = [KEY]
    reader._depth_encoder_configs[KEY] = SimpleNamespace(depth_min=0, depth_max=1, shift=0, use_log=False)
    assert reader.preload_camera_frame_cache()["camera_cache_frames"] == 0
    calls = []

    def decode(*args, **kwargs):
        calls.append(kwargs)
        return torch.ones(1, 1, 16, 16)

    monkeypatch.setattr("lerobot.datasets.dataset_reader.decode_video_frames", decode)
    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.dequantize_depth", lambda frames, **kwargs: frames * 2
    )
    assert reader.get_item(0)[KEY].shape == (1, 16, 16)
    assert calls == [{"return_uint8": True, "is_depth": True}]


def test_cache_config_rejects_streaming():
    with pytest.raises(ValueError, match="streaming"):
        DatasetConfig(repo_id="test/data", streaming=True, camera_cache="ram")
    with pytest.raises(ValueError, match="camera_cache"):
        DatasetConfig(repo_id="test/data", camera_cache="invalid")


@pytest.mark.parametrize("mode", ["ram", "disk"])
def test_factory_preloads_actual_split_datasets(monkeypatch, mode):
    from lerobot.datasets import factory

    full = SimpleNamespace(
        episodes=None,
        num_episodes=4,
        meta=SimpleNamespace(episodes={"tasks": [["test"]] * 4}, camera_keys=[]),
    )
    calls = []

    def make_full(cfg, *, preload_camera_cache):
        assert not preload_camera_cache
        return full

    def make_view(repo_id, *, episodes, **kwargs):
        view = SimpleNamespace(
            episodes=episodes,
            meta=SimpleNamespace(camera_keys=[]),
            preload_camera_frame_cache=lambda: calls.append(("ram", episodes)),
            preload_camera_frame_cache_disk=lambda: calls.append(("disk", episodes)),
        )
        return view

    monkeypatch.setattr(factory, "make_dataset", make_full)
    monkeypatch.setattr(factory, "LeRobotDataset", make_view)
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args: None)
    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="test/data", camera_cache=mode, eval_split=0.25),
        trainable_config=None,
        rename_map={},
        tolerance_s=1e-4,
    )
    train, evaluation = factory.make_train_eval_datasets(cfg)
    assert train.episodes == [0, 1, 2]
    assert evaluation.episodes == [3]
    assert calls == [(mode, [0, 1, 2]), (mode, [3])]


def test_factory_rejects_other_readers(monkeypatch):
    from lerobot.datasets import factory

    monkeypatch.setattr(
        factory,
        "load_dataset_metadata",
        lambda *args, **kwargs: SimpleNamespace(storage_format="lance", total_episodes=1),
    )
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args: None)
    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="test/data", camera_cache="disk"),
        trainable_config=None,
        rename_map={},
    )
    with pytest.raises(ValueError, match="Parquet"):
        factory.make_dataset(cfg)


def test_facade_preload_and_injected_cache(reader_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reader = reader_factory(episodes=[1])
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.reader = reader
    dataset.writer = None
    dataset._is_finalized = True
    assert dataset.preload_camera_frame_cache()["camera_cache_frames"] == 3
    assert dataset._camera_frame_cache is reader._camera_frame_cache
    other = reader_factory(episodes=[1])
    other._camera_frame_cache = dataset._camera_frame_cache
    torch.testing.assert_close(other.get_item(0)[KEY], dataset[0][KEY], rtol=0, atol=0)


def test_disk_cache_through_spawn_dataloader(reader_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.reader = reader_factory(episodes=[1], history=True)
    dataset.writer = None
    dataset._is_finalized = True
    expected = dataset[0][KEY]
    dataset.preload_camera_frame_cache_disk()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn", timeout=30
    )
    batches = list(loader)
    torch.testing.assert_close(batches[0][KEY][0], expected, rtol=0, atol=0)
    assert torch.cat([batch["index"] for batch in batches]).tolist() == [3, 4, 5]


def test_disk_cache_external_root(reader_factory, tmp_path, monkeypatch):
    reader = reader_factory()
    external = tmp_path / "external-cache"
    monkeypatch.setenv("LEROBOT_CAMERA_CACHE_DIR", str(external))
    reader.preload_camera_frame_cache_disk()
    assert reader._camera_frame_cache.root.parent == external
    assert not (reader.root / ".camera_frame_cache").exists()
