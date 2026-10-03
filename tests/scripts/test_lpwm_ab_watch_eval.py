"""Checkpoint capture and success-selection tests, without a simulator or network."""

import importlib.util
import json
import os
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lpwm_watch_eval", Path(__file__).parents[2] / "scripts/lpwm_ab/watch_eval.py"
)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def source_checkpoint(root, step=2000):
    root.mkdir(parents=True)
    for name in watcher.REQUIRED:
        (root / name).write_text(name)
    (root / "experiment.json").write_text(json.dumps({"step": step, "variant": "A"}))
    timestamp = time.time() - 10
    for name in watcher.REQUIRED:
        os.utime(root / name, (timestamp, timestamp))
    os.utime(root / "experiment.json", (timestamp + 1, timestamp + 1))
    return root


def test_freeze_is_immutable_when_latest_changes(tmp_path):
    source = source_checkpoint(tmp_path / "latest")
    snapshot = watcher.freeze_checkpoint(source, tmp_path / "snapshots")
    assert snapshot.name == "step_002000"
    assert (snapshot / "model.safetensors").read_text() == "model.safetensors"
    (source / "model.safetensors").write_text("changed")
    assert (snapshot / "model.safetensors").read_text() == "model.safetensors"
    assert watcher.freeze_checkpoint(source, tmp_path / "snapshots") is None


def test_uncommitted_new_weights_with_old_step_are_rejected(tmp_path):
    source = source_checkpoint(tmp_path / "latest")
    timestamp = time.time() - 5
    os.utime(source / "model.safetensors", (timestamp, timestamp))
    assert watcher.freeze_checkpoint(source, tmp_path / "snapshots") is None


def test_partial_export_is_rejected(tmp_path):
    source = source_checkpoint(tmp_path / "latest")
    (source / "language_embeddings.npy").unlink()
    assert watcher.freeze_checkpoint(source, tmp_path / "snapshots") is None


def test_copy_detects_concurrent_write(tmp_path, monkeypatch):
    source = source_checkpoint(tmp_path / "latest")
    original = watcher.shutil.copy2

    def racing_copy(src, dst):
        result = original(src, dst)
        if src.name == "model.safetensors":
            src.write_text("concurrent overwrite")
        return result

    monkeypatch.setattr(watcher.shutil, "copy2", racing_copy)
    assert watcher.freeze_checkpoint(source, tmp_path / "snapshots") is None
    assert not list((tmp_path / "snapshots").iterdir())


def test_success_checkpoint_separate_from_loss_best(tmp_path):
    source = source_checkpoint(tmp_path / "run/checkpoints/latest")
    (source.parent / "best").mkdir()
    snapshot = watcher.freeze_checkpoint(source, tmp_path / "snapshots")
    result = {
        "success_rate": 0.3,
        "pc_success": 30.0,
        "successes": 3,
        "num_episodes": 10,
        "protocol_sha256": "fixed",
    }
    assert watcher.publish_best(tmp_path / "run", snapshot, result, tmp_path / "result.json")
    assert (source.parent / "best").is_dir()
    assert (source.parent / "best_success").resolve() == snapshot.resolve()
    assert not watcher.publish_best(tmp_path / "run", snapshot, result, tmp_path / "result.json")
