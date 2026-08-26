from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.franka_duo_real_recorder.action_spec import (
    ACTION_DIM,
    FrankaDuoActionSpec,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from examples.franka_duo_real_recorder.eval_franka_duo import build_policy_observation
from examples.franka_duo_real_recorder.export_rl100_eval_bundle import main as export_bundle_main
from examples.franka_duo_real_recorder.franka_duo_eval_io import (
    EvalCameraConfig,
    EvalObservationCache,
    PointCloudConfig,
    SynchronizedObservationReader,
    make_point_cloud,
)
from examples.franka_duo_real_recorder.rl100_eval_policy import (
    _prepare_native_input,
    load_policy_bundle,
)


def _stamp(stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )


def _image(value: int, stamp_ns: int, width: int, height: int) -> SimpleNamespace:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    return SimpleNamespace(
        height=height,
        width=width,
        step=width * 3,
        encoding="rgb8",
        is_bigendian=False,
        data=pixels.tobytes(),
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def _depth(value: float, stamp_ns: int, width: int, height: int) -> SimpleNamespace:
    pixels = np.full((height, width), value, dtype=np.float32)
    return SimpleNamespace(
        height=height,
        width=width,
        step=width * 4,
        encoding="32FC1",
        is_bigendian=False,
        data=pixels.tobytes(),
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def _camera_info(width: int, height: int, stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        k=[100.0, 0.0, width / 2, 0.0, 100.0, height / 2, 0.0, 0.0, 1.0],
        header=SimpleNamespace(stamp=_stamp(stamp_ns)),
    )


def test_rot6d_contract_matches_rl100_rows():
    matrix = np.eye(3, dtype=np.float32)
    six = matrix_to_rot6d(matrix)
    assert six.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(rot6d_to_matrix(six), matrix, atol=1e-6)


def test_action_spec_rejects_bad_rotation_and_workspace():
    spec = FrankaDuoActionSpec(workspace_min=(-1.0, -1.0, 0.0), workspace_max=(1.0, 1.0, 1.0))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[3:9] = [1, 0, 0, 0, 1, 0]
    action[12:18] = [1, 0, 0, 0, 1, 0]
    action[0] = 2.0
    with pytest.raises(ValueError, match="outside workspace"):
        spec.validate(action)
    action[0] = 0.0
    action[3:9] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        spec.validate(action)


def test_pointcloud_supports_xyz_and_xyzrgb_and_fps():
    width, height = 8, 6
    depth = np.ones((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)
    info = _camera_info(width, height, 1)
    for channels in (3, 6):
        points = make_point_cloud(
            depth,
            rgb,
            info,
            PointCloudConfig(num_points=12, channels=channels, sampling="fps", min_depth=0.1),
        )
        assert points.shape == (12, channels)
        assert np.isfinite(points).all()
    with pytest.raises(ValueError, match="shapes differ"):
        make_point_cloud(depth, rgb[:-1], info, PointCloudConfig(num_points=4))


def test_ros_reader_anchors_head_stamp_and_matches_depth_and_wrists():
    width, height = 8, 6
    cameras = {
        "head": EvalCameraConfig("head", "/head", width, height, depth_topic="/depth", camera_info_topic="/info"),
        "wrist_left": EvalCameraConfig("wrist_left", "/left", width, height),
        "wrist_right": EvalCameraConfig("wrist_right", "/right", width, height),
    }
    cache = EvalObservationCache(history_size=4)
    reader = SynchronizedObservationReader(
        cache,
        cameras,
        PointCloudConfig(num_points=8),
        rgb_tolerance_ms=2.0,
        depth_tolerance_ms=2.0,
        sync_wait_timeout_ms=2.0,
    )
    stamp = 1_000_000_000
    cache.store_camera_info(_camera_info(width, height, stamp))
    cache.store_depth(_depth(1.0, stamp + 100_000, width, height))
    cache.store_image("wrist_left", _image(2, stamp - 100_000, width, height))
    cache.store_image("wrist_right", _image(3, stamp + 100_000, width, height))
    cache.store_image("head", _image(1, stamp, width, height))
    observation = reader.next(timeout_s=0.1)
    assert observation.stamp_ns == stamp
    assert observation.point_cloud.shape == (8, 3)
    assert observation.source_stamps_ns["depth"] == stamp + 100_000
    # A new frame with a stale depth must be rejected instead of reusing the old pair.
    cache.store_image("head", _image(4, stamp + 33_333_333, width, height))
    with pytest.raises(TimeoutError, match="skew"):
        reader.next(timeout_s=0.01)


def _native_manifest() -> dict:
    return {
        "manifest_version": 1,
        "backend": "rl100_native",
        "action_dim": 20,
        "action_spec": {
            "dimension": 20,
            "ee_dimension": 9,
            "ee_rotation": "rot6d_rows",
            "gripper_range": [0.0, 1.0],
        },
        "pointcloud": {"num_points": 8, "channels": 3, "sampling": "random"},
        "inputs": {
            "point_cloud_key": "point_cloud",
            "state_key": None,
            "image_keys": {
                "wrist_left": "wrist_left",
                "wrist_right": "wrist_right",
            },
        },
        "native": {"factory": "factory_mod:make", "python_root": "."},
    }


def test_native_bundle_requires_manifest_factory_and_preserves_contract(tmp_path):
    (tmp_path / "factory_mod.py").write_text(
        """
class Model:
    def predict(self, batch):
        assert batch['point_cloud'].shape == (1, 8, 3)
        assert batch['wrist_left'].shape[1] == 3
        assert batch['wrist_right'].shape[1] == 3
        return [0.0] * 20
def make(bundle_dir, device):
    return Model()
""",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(_native_manifest()), encoding="utf-8")
    bundle = load_policy_bundle(tmp_path, device="cpu")
    assert bundle.action_spec.dimension == ACTION_DIM
    assert bundle.required_observation_keys == ("point_cloud", "wrist_left", "wrist_right")
    observation = {
        "point_cloud": np.zeros((8, 3), dtype=np.float32),
        "wrist_left": np.zeros((6, 16, 3), dtype=np.uint8),
        "wrist_right": np.zeros((6, 16, 3), dtype=np.uint8),
    }
    np.testing.assert_equal(bundle.predict(observation), np.zeros(20, dtype=np.float32))


def test_native_input_uses_manifest_image_keys_for_custom_names():
    prepared = _prepare_native_input(
        {"wrist_left": np.zeros((6, 8, 3), dtype=np.uint8)},
        "cpu",
        image_keys=("wrist_left",),
    )
    assert tuple(prepared["wrist_left"].shape) == (1, 3, 6, 8)
    assert prepared["wrist_left"].dtype == torch.float32


def test_native_bundle_rejects_missing_required_observation(tmp_path):
    (tmp_path / "factory_mod.py").write_text(
        """
class Model:
    def predict(self, batch):
        return [0.0] * 20
def make(bundle_dir, device):
    return Model()
""",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(_native_manifest()), encoding="utf-8")
    bundle = load_policy_bundle(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="missing required"):
        bundle.predict({"point_cloud": np.zeros((8, 3), dtype=np.float32)})


def test_existing_checkpoint_without_bundle_manifest_is_rejected(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"not a self describing export")
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_policy_bundle(tmp_path, device="cpu")


def test_export_bundle_copies_native_weights_and_writes_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    (checkpoint / "encoder.pt").write_bytes(b"encoder")
    output = tmp_path / "bundle"
    assert export_bundle_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--factory",
            "factory_mod:make",
            "--workspace-min=-1,-1,0",
            "--workspace-max=1,1,1",
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "rl100_native"
    assert manifest["action_spec"]["dimension"] == ACTION_DIM
    assert (output / "checkpoint" / "encoder.pt").is_file()


def test_policy_observation_can_pack_both_wrist_images():
    observation = SimpleNamespace(
        head_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        wrist_left_rgb=np.ones((2, 3, 3), dtype=np.uint8),
        wrist_right_rgb=np.full((2, 3, 3), 2, dtype=np.uint8),
        point_cloud=np.zeros((4, 3), dtype=np.float32),
        state=None,
    )
    values = build_policy_observation(
        observation,
        {
            "inputs": {
                "point_cloud_key": "point_cloud",
                "image_keys": {"image": ["wrist_left", "wrist_right"]},
            }
        },
    )
    assert values["image"].shape == (2, 6, 3)
    assert values["image"][0, 0, 0] == 1 and values["image"][0, -1, 0] == 2
