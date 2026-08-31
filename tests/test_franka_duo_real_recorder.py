from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.franka_duo_real_recorder.build_pointcloud_dataset import (
    PointCloudBuilder,
    main as build_pointcloud_dataset_main,
)
from examples.franka_duo_real_recorder.record_franka_duo import (
    ACTION_DIM,
    ACTION_NAMES,
    STATE_DIM,
    STATE_NAMES,
    CameraSpec,
    DepthSidecarWriter,
    KeyboardCommands,
    RecorderConfig,
    TimedMessage,
    _encoder_dropped_frames,
    _episode_integrity_errors,
    _nearest_depth,
    _ready_snapshot,
    _select_synchronized_messages,
    _stamp_ns,
    _validate_resume_contract,
    _write_or_validate_recording_manifest,
    build_action,
    build_features,
    build_state,
    camera_info_to_dict,
    depth_msg_to_meters,
    existing_dataset_versions,
    gripper_open_fraction,
    image_msg_to_rgb,
    load_config,
    validate_config,
)
from examples.franka_duo_real_recorder.validate_dataset import validate_dataset
from lerobot.utils.constants import OBS_POINT_CLOUD


def _camera(
    key: str,
    width: int = 8,
    height: int = 8,
    *,
    record_depth: bool = False,
) -> CameraSpec:
    return CameraSpec(
        key=key,
        image_topic=f"/{key}/image",
        width=width,
        height=height,
        fps=30,
        depth_topic=f"/{key}/depth" if record_depth else None,
        camera_info_topic=f"/{key}/camera_info",
        record_depth=record_depth,
    )


def _timed(message: object, stamp_ns: int) -> TimedMessage:
    return TimedMessage(message=message, arrival_ns=time.monotonic_ns(), stamp_ns=stamp_ns)


def _camera_info(camera: CameraSpec, stamp_ns: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        width=camera.width,
        height=camera.height,
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=[100.0, 0.0, camera.width / 2, 0.0, 100.0, camera.height / 2, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[100.0, 0.0, camera.width / 2, 0.0, 0.0, 100.0, camera.height / 2, 0.0, 0.0, 0.0, 1.0, 0.0],
        header=SimpleNamespace(
            frame_id=f"{camera.key}_optical",
            stamp=SimpleNamespace(sec=0, nanosec=stamp_ns),
        ),
    )


def test_real_contract_excludes_base_and_world_pose():
    measured = {
        **{f"left_fr3v2_joint{i}": float(i) for i in range(1, 8)},
        **{f"right_fr3v2_joint{i}": float(i) for i in range(1, 8)},
        "left_right_finger_joint": 0.4,
        "right_right_finger_joint": 0.8,
        "franka_spine_vertical_joint": 0.2,
    }
    state = build_state(measured)
    action = build_action(measured, measured)
    assert state.shape == (STATE_DIM,) == (17,)
    assert action.shape == (ACTION_DIM,) == (17,)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.isclose(state[-3], 0.5)
    assert np.isclose(state[-2], 0.0)
    assert state[-1] == 0.2


def test_real_feature_names_match_benchmark_joint_contract():
    assert ACTION_NAMES[:2] == ("left_fr3v2_joint1.target", "left_fr3v2_joint2.target")
    assert ACTION_NAMES[7] == "right_fr3v2_joint1.target"
    assert STATE_NAMES[:2] == ("left_fr3v2_joint1.pos", "left_fr3v2_joint2.pos")
    assert STATE_NAMES[7] == "right_fr3v2_joint1.pos"


def test_gripper_calibration_supports_real_franka_finger_positions():
    assert gripper_open_fraction(0.0, closed_rad=0.0, open_rad=0.04) == 0.0
    assert gripper_open_fraction(0.04, closed_rad=0.0, open_rad=0.04) == 1.0
    assert gripper_open_fraction(0.02, closed_rad=0.0, open_rad=0.04) == 0.5
    assert gripper_open_fraction(0.4, closed_rad=0.8, open_rad=0.0) == 0.5


def test_image_decoding_handles_stride_and_bgr():
    # Two RGB pixels per row plus four bytes of padding.
    payload = bytes((1, 2, 3, 4, 5, 6, 99, 99))
    message = SimpleNamespace(height=1, width=2, step=8, encoding="bgr8", data=payload)
    image = image_msg_to_rgb(message, expected_shape=(1, 2, 3))
    np.testing.assert_array_equal(image, [[[3, 2, 1], [6, 5, 4]]])


@pytest.mark.parametrize(
    ("encoding", "payload"),
    (("yuv422_yuy2", bytes((16, 128, 235, 128))), ("yuv422", bytes((128, 16, 128, 235)))),
)
def test_image_decoding_supports_d405_yuv422(encoding: str, payload: bytes):
    message = SimpleNamespace(height=1, width=2, step=4, encoding=encoding, data=payload)

    image = image_msg_to_rgb(message, expected_shape=(1, 2, 3))

    np.testing.assert_allclose(image[0, 0], [0, 0, 0], atol=1)
    np.testing.assert_allclose(image[0, 1], [255, 255, 255], atol=1)


def test_depth_decoding_16uc1_to_meters():
    values = np.asarray([1000, 2500], dtype=np.uint16)
    message = SimpleNamespace(
        height=1,
        width=2,
        step=4,
        encoding="16UC1",
        data=values.tobytes(),
    )
    np.testing.assert_allclose(depth_msg_to_meters(message), [[1.0, 2.5]], atol=1e-3)


def test_depth_decoding_respects_big_endian_ros_image():
    values = np.asarray([1000, 2500], dtype=">u2")
    message = SimpleNamespace(
        height=1,
        width=2,
        step=4,
        encoding="16UC1",
        is_bigendian=True,
        data=values.tobytes(),
    )

    np.testing.assert_allclose(depth_msg_to_meters(message), [[1.0, 2.5]], atol=1e-3)


def test_zero_ros_stamp_is_rejected_as_uninitialized():
    zero = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))
    assert _stamp_ns(zero) is None

    positive = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=3)))
    assert _stamp_ns(positive) == 2_000_000_003


def test_depth_matching_uses_header_stamp_not_arrival_order():
    first = SimpleNamespace(stamp_ns=100, message="first")
    second = SimpleNamespace(stamp_ns=200, message="second")
    assert _nearest_depth((first, second), 190, 20).message == "second"
    assert _nearest_depth((first, second), 190, 5) is None


def test_depth_sidecar_preserves_frame_and_stamp_alignment(tmp_path: Path):
    writer = DepthSidecarWriter(tmp_path, queue_size=2)
    writer.start_episode(3)
    writer.record_sample_timestamps(
        7,
        state_stamp_ns=98,
        action_stamp_ns=99,
        rgb_stamps_ns={"head": 100, "wrist_left": 101, "wrist_right": 102},
    )
    assert writer.enqueue(7, 100, 101, np.ones((2, 3), dtype=np.float32))
    metadata = writer.finish_episode(save=True)
    assert metadata["frames_written"] == 1
    assert metadata["missing_rgb_frame_indices"] == []
    episode = tmp_path / "episode_000003"
    assert np.fromfile(episode / "depth_head_frame_index.i8", dtype="<i8").tolist() == [7]
    assert np.fromfile(episode / "depth_head_stamp_ns.i8", dtype="<i8").tolist() == [100, 101]
    saved_metadata = json.loads((episode / "metadata.json").read_text())
    assert saved_metadata["unit"] == "meter"
    assert saved_metadata["sample_timestamp_layout"] == [
        "frame_index",
        "state_stamp_ns",
        "action_stamp_ns",
        "rgb_head_stamp_ns",
        "rgb_wrist_left_stamp_ns",
        "rgb_wrist_right_stamp_ns",
    ]
    assert np.fromfile(episode / "sample_timestamps_ns.i8", dtype="<i8").tolist() == [
        7,
        98,
        99,
        100,
        101,
        102,
    ]


def test_features_are_video_rgb_only_for_three_cameras():
    cameras = {
        "head": CameraSpec(
            key="head",
            image_topic="/head",
            width=1280,
            height=720,
            fps=30,
            depth_topic="/head/depth",
            camera_info_topic="/head/camera_info",
            record_depth=True,
        ),
        "wrist_left": _camera("wrist_left", width=480, height=270),
        "wrist_right": _camera("wrist_right", width=480, height=270),
    }
    features = build_features(cameras)
    assert features["observation.images.head"]["dtype"] == "video"
    assert {key for key in features if key.startswith("observation.images.")} == {
        "observation.images.head",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    }
    assert features["observation.images.wrist_left"]["shape"] == (270, 480, 3)


def test_synchronization_uses_nearest_new_rgb_and_requires_camera_info():
    cameras = {
        "head": _camera("head"),
        "wrist_left": _camera("wrist_left"),
        "wrist_right": _camera("wrist_right"),
    }
    snapshot = {
        "joint_states": (_timed("state", 99), _timed("state-new", 101)),
        "applied_commands": (_timed("action", 100),),
        "images": {
            "head": (_timed("head", 100),),
            "wrist_left": (_timed("left", 98), _timed("left-next", 132)),
            "wrist_right": (_timed("right", 102),),
        },
        "camera_info": {key: _timed(_camera_info(camera), 1) for key, camera in cameras.items()},
    }

    selected = _select_synchronized_messages(
        snapshot,
        cameras,
        100,
        rgb_tolerance_ns=5,
        control_tolerance_ns=5,
        max_age_ns=1_000_000_000,
    )
    assert selected["images"]["wrist_left"].message == "left"
    assert selected["joint_states"].message in {"state", "state-new"}

    with pytest.raises(ValueError, match="timestamp tolerances"):
        _select_synchronized_messages(
            snapshot,
            cameras,
            101,
            rgb_tolerance_ns=5,
            control_tolerance_ns=5,
            max_age_ns=1_000_000_000,
            rgb_after_stamps={"head": 100, "wrist_left": 98, "wrist_right": 102},
        )

    snapshot["camera_info"].pop("head")
    assert not _ready_snapshot(snapshot, cameras, max_age_ns=1_000_000_000)


def test_numeric_dataset_version_ordering(tmp_path: Path):
    for name in ("demo_v1", "demo_v9", "demo_v10", "demo_vx", "other_v99"):
        (tmp_path / name).mkdir()

    versions = existing_dataset_versions(tmp_path, "demo")

    assert [(version, path.name) for version, path in versions] == [
        (1, "demo_v1"),
        (9, "demo_v9"),
        (10, "demo_v10"),
    ]


def test_default_config_uses_requested_d405_profile_and_requires_calibration(tmp_path: Path):
    config_path = Path("examples/franka_duo_real_recorder/config.yaml")
    config = load_config(config_path)

    assert (config.cameras["wrist_left"].width, config.cameras["wrist_left"].height) == (480, 270)
    assert config.cameras["wrist_left"].fps == 30
    assert config.cameras["wrist_right"].record_depth is False
    assert config.cameras["head"].record_depth is True
    assert config.depth_every == 1
    assert config.sync_history_size == 30
    assert config.reject_episode_on_missing_depth is True
    with pytest.raises(ValueError, match="real-hardware calibration"):
        validate_config(config)

    config.gripper_closed_rad = 0.0
    config.gripper_open_rad = 0.04
    validate_config(config)

    config.encoder_queue_maxsize = 0
    with pytest.raises(ValueError, match="encoder_queue_maxsize"):
        validate_config(config)
    config.encoder_queue_maxsize = 90

    _write_or_validate_recording_manifest(
        tmp_path,
        config,
        config.cameras,
        resume=False,
    )
    _write_or_validate_recording_manifest(
        tmp_path,
        config,
        config.cameras,
        resume=True,
    )
    config.rgb_match_tolerance_ms += 1
    with pytest.raises(ValueError, match="recording_manifest"):
        _write_or_validate_recording_manifest(
            tmp_path,
            config,
            config.cameras,
            resume=True,
        )


def test_keyboard_restores_terminal_settings(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    class FakeStdin:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 7

    class FakeThread:
        def start(self) -> None:
            calls.append(("thread_start",))

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float) -> None:
            calls.append(("join", timeout))

    monkeypatch.setattr("examples.franka_duo_real_recorder.record_franka_duo.sys.stdin", FakeStdin())
    monkeypatch.setattr(
        "examples.franka_duo_real_recorder.record_franka_duo.termios.tcgetattr",
        lambda fd: ["saved", fd],
    )
    monkeypatch.setattr(
        "examples.franka_duo_real_recorder.record_franka_duo.tty.setcbreak",
        lambda fd: calls.append(("cbreak", fd)),
    )
    monkeypatch.setattr(
        "examples.franka_duo_real_recorder.record_franka_duo.termios.tcsetattr",
        lambda fd, when, settings: calls.append(("restore", fd, when, settings)),
    )
    keyboard = KeyboardCommands()
    keyboard._thread = FakeThread()

    keyboard.start()
    keyboard.close()

    assert ("cbreak", 7) in calls
    assert any(call[0] == "restore" and call[1] == 7 for call in calls)


def test_resume_contract_and_encoder_drop_guard():
    cameras = {"head": _camera("head")}
    features = build_features(cameras)
    features["observation.images.head"]["info"] = {
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
    }
    meta = SimpleNamespace(
        fps=30,
        robot_type="franka_duo_real",
        features=features,
        video_keys=["observation.images.head"],
    )
    encoder = SimpleNamespace(vcodec="h264_nvenc", pix_fmt="yuv420p")
    streaming_encoder = SimpleNamespace(_dropped_frames={})
    dataset = SimpleNamespace(
        meta=meta,
        writer=SimpleNamespace(
            _camera_encoder=encoder,
            _streaming_encoder=streaming_encoder,
        ),
    )
    config = RecorderConfig(
        cameras=cameras,
        gripper_closed_rad=0.0,
        gripper_open_rad=0.04,
    )
    _validate_resume_contract(dataset, config, cameras)

    features["observation.images.head"]["shape"] = (7, 8, 3)
    with pytest.raises(ValueError, match="shape"):
        _validate_resume_contract(dataset, config, cameras)
    features["observation.images.head"]["shape"] = (8, 8, 3)

    streaming_encoder._dropped_frames = {"observation.images.head": 1}
    depth_writer = DepthSidecarWriter(Path("unused"))
    assert _encoder_dropped_frames(dataset) == {"observation.images.head": 1}
    assert "streaming video encoder dropped frames" in " ".join(
        _episode_integrity_errors(dataset, depth_writer, frames=2)
    )
    depth_writer.mark_missing(0)
    assert "depth synchronization missed" in " ".join(
        _episode_integrity_errors(dataset, depth_writer, frames=2)
    )


def test_pointcloud_builder_reads_depth_sidecar_and_zero_pads_sparse_points(tmp_path: Path):
    episode = tmp_path / "episode_000000"
    episode.mkdir()
    depth = np.ones((1, 2, 2), dtype="<f2")
    depth.tofile(episode / "depth_head.f16")
    np.asarray([0], dtype="<i8").tofile(episode / "depth_head_frame_index.i8")
    metadata = {
        "shape": [2, 2],
        "frames_written": 1,
        "frame_index_file": "depth_head_frame_index.i8",
        "depth_file": "depth_head.f16",
        "camera_info": {
            "head": {
                "height": 2,
                "width": 2,
                "k": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }
        },
        "head_to_robot_base_transform": np.eye(4).reshape(-1).tolist(),
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    builder = PointCloudBuilder(
        tmp_path,
        num_points=6,
        extrinsics=None,
        workspace_min=None,
        workspace_max=None,
        min_depth=0.05,
        max_depth=5.0,
        max_frame_gap=0,
        seed=3,
        require_extrinsics=True,
    )

    points = builder({}, episode_index=0, frame_in_episode=0)

    assert points.shape == (6, 3)
    assert np.unique(points[:4], axis=0).shape[0] == 4
    np.testing.assert_array_equal(points[4:], np.zeros((2, 3), dtype=np.float32))
    assert builder.source_frame_indices == [(0, 0)]


def test_lerobot_v3_video_to_pointcloud_dataset_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("av", reason="PyAV is required for the recorder video smoke test")
    hf_datasets = pytest.importorskip("datasets", reason="datasets is required for LeRobot v3")
    import pandas as pd

    hf_cache = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    monkeypatch.setattr(hf_datasets.config, "HF_DATASETS_CACHE", str(hf_cache / "datasets"))

    from lerobot.configs.video import VideoEncoderConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.pyav_utils import get_codec

    if get_codec("libsvtav1") is None:
        pytest.skip("libsvtav1 encoder is unavailable")

    source_root = tmp_path / "source_v1"
    output_root = tmp_path / "pointcloud_v1"
    head = _camera("head", width=48, height=32, record_depth=True)
    dataset = LeRobotDataset.create(
        repo_id="local/source_v1",
        root=source_root,
        fps=30,
        features=build_features({"head": head}),
        robot_type="franka_duo_real",
        use_videos=True,
        camera_encoder=VideoEncoderConfig(vcodec="libsvtav1", preset=13),
    )
    for frame_index in range(2):
        dataset.add_frame(
            {
                "action": np.full(ACTION_DIM, frame_index, dtype=np.float32),
                "observation.state": np.full(STATE_DIM, frame_index, dtype=np.float32),
                "observation.images.head": np.full(
                    (head.height, head.width, 3), 30 + frame_index, dtype=np.uint8
                ),
                "task": "point-cloud derivation smoke test",
            }
        )
    dataset.save_episode()
    dataset.finalize()

    transform = tuple(np.eye(4).reshape(-1).tolist())
    recorder_config = RecorderConfig(
        cameras={"head": head},
        gripper_closed_rad=0.0,
        gripper_open_rad=0.04,
        depth_every=1,
        rgb_vcodec="libsvtav1",
        head_to_robot_base_transform=transform,
    )
    _write_or_validate_recording_manifest(
        source_root,
        recorder_config,
        {"head": head},
        resume=False,
    )
    sidecar_writer = DepthSidecarWriter(
        source_root / "franka_duo_extras",
        head_to_robot_base_transform=transform,
    )
    sidecar_writer.start_episode(0)
    sidecar_writer.set_camera_info({"head": camera_info_to_dict(_camera_info(head))})
    for frame_index in range(2):
        stamp_ns = 1_000_000_000 + frame_index * 33_333_333
        sidecar_writer.record_sample_timestamps(
            frame_index,
            state_stamp_ns=stamp_ns - 1_000_000,
            action_stamp_ns=stamp_ns - 2_000_000,
            rgb_stamps_ns={"head": stamp_ns},
        )
        assert sidecar_writer.enqueue(
            frame_index,
            stamp_ns,
            stamp_ns,
            np.ones((head.height, head.width), dtype=np.float32),
        )
    sidecar_writer.finish_episode(save=True)

    validation_report = validate_dataset(source_root, repo_id="local/source_v1")
    assert validation_report["valid"], validation_report["errors"]

    assert (
        build_pointcloud_dataset_main(
            [
                "--dataset-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--num-points",
                "16",
                "--max-frame-gap",
                "0",
            ]
        )
        == 0
    )

    derived = LeRobotDataset(
        repo_id="local/pointcloud_v1",
        root=output_root,
        download_videos=False,
    )
    assert derived.meta.video_keys == []
    assert tuple(derived.meta.features[OBS_POINT_CLOUD]["shape"]) == (16, 3)
    assert OBS_POINT_CLOUD in derived.meta.stats
    assert np.asarray(derived.get_raw_item(0)[OBS_POINT_CLOUD]).shape == (16, 3)
    episode_columns = pd.read_parquet(next((output_root / "meta" / "episodes").rglob("*.parquet"))).columns
    assert not any(str(column).startswith("videos/") for column in episode_columns)
    assert (output_root / "pointcloud_extras" / "calibrations.json").is_file()
    assert np.load(output_root / "pointcloud_extras" / "source_depth_episode_and_frame.npy").tolist() == [
        [0, 0],
        [0, 1],
    ]
