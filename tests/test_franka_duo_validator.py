from __future__ import annotations

from pathlib import Path

from examples.franka_duo_real_recorder.validate_dataset import _probe_video_file, validate_dataset


def test_probe_video_file_reports_missing_reference(tmp_path: Path) -> None:
    duration, errors = _probe_video_file(
        tmp_path / "missing.mp4",
        feature_key="observation.images.head",
        expected_shape=(720, 1280, 3),
        expected_fps=30,
        fps_tolerance_ratio=0.1,
    )

    assert duration is None
    assert errors == [
        f"observation.images.head: referenced video file is missing: {tmp_path / 'missing.mp4'}"
    ]


def test_probe_video_file_reports_unreadable_reference(tmp_path: Path) -> None:
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not an MP4")

    duration, errors = _probe_video_file(
        video_path,
        feature_key="observation.images.head",
        expected_shape=(720, 1280, 3),
        expected_fps=30,
        fps_tolerance_ratio=0.1,
    )

    assert duration is None
    assert any("cannot read referenced video" in error for error in errors)


def test_validate_dataset_reports_malformed_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "franka_duo_extras" / "recording_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{", encoding="utf-8")

    report = validate_dataset(tmp_path)

    assert not report["valid"]
    assert "Cannot read recording manifest" in report["errors"][0]


def test_validate_dataset_reports_missing_manifest_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "franka_duo_extras" / "recording_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")

    report = validate_dataset(tmp_path)

    assert not report["valid"]
    assert "missing required fields" in report["errors"][0]
