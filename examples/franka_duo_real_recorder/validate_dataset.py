#!/usr/bin/env python3
"""Validate a real Franka Duo LeRobot v3 dataset and its synchronized sidecars."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _shape(feature: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in feature.get("shape", ()))


def _episode_rows(metadata: Any) -> list[dict[str, Any]]:
    episodes = metadata.episodes
    if episodes is None:
        return []
    return [episodes[index] for index in range(len(episodes))]


def _append(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _probe_video_file(
    path: Path,
    *,
    feature_key: str,
    expected_shape: tuple[int, int, int],
    expected_fps: int,
    fps_tolerance_ratio: float,
) -> tuple[float | None, list[str]]:
    """Check one v3 MP4 reference and return its physical duration in seconds.

    Episode metadata stores references to chunked MP4 files rather than one
    file per episode. A valid episode row can therefore still point at a
    deleted or truncated video. Keep this probe lazy and cache it at the
    caller so each physical file is opened once even when several episodes
    share the same chunk.
    """

    errors: list[str] = []
    if not path.is_file():
        return None, [f"{feature_key}: referenced video file is missing: {path}"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, [f"{feature_key}: cannot stat referenced video file {path}: {exc}"]
    if size <= 0:
        return None, [f"{feature_key}: referenced video file is empty: {path}"]

    try:
        import av
    except ImportError as exc:
        return None, [f"{feature_key}: PyAV is required to probe {path}: {exc}"]

    duration_s: float | None = None
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if not streams:
                return None, [f"{feature_key}: referenced file has no video stream: {path}"]
            stream = streams[0]
            actual_shape = (int(stream.height), int(stream.width), 3)
            if actual_shape != expected_shape:
                errors.append(
                    f"{feature_key}: video stream shape {actual_shape} != expected {expected_shape} ({path})"
                )
            rate = stream.average_rate or stream.base_rate
            actual_fps = float(rate) if rate else 0.0
            if actual_fps <= 0.0:
                errors.append(f"{feature_key}: video stream has no positive FPS ({path})")
            elif abs(actual_fps - expected_fps) / expected_fps > fps_tolerance_ratio:
                errors.append(
                    f"{feature_key}: video stream FPS {actual_fps:.3f} != expected {expected_fps} "
                    f"within {fps_tolerance_ratio:.1%} ({path})"
                )

            # Opening a container only validates its header. Decode one frame
            # as well so a truncated/corrupt MP4 cannot pass the preflight.
            first_frame = next(container.decode(video=stream.index), None)
            if first_frame is None:
                errors.append(f"{feature_key}: referenced video contains no decodable frames ({path})")

            if container.duration is not None:
                duration_s = float(container.duration / av.time_base)
            elif stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            if duration_s is None or not np.isfinite(duration_s) or duration_s <= 0:
                errors.append(f"{feature_key}: referenced video has no positive duration ({path})")
                duration_s = None
    except Exception as exc:  # PyAV raises codec/container-specific exceptions.
        errors.append(f"{feature_key}: cannot read referenced video {path}: {exc}")
        duration_s = None
    return duration_s, errors


def validate_dataset(
    dataset_root: Path,
    *,
    repo_id: str | None = None,
    max_depth_missing_ratio: float = 0.0,
    fps_tolerance_ratio: float = 0.1,
    max_frame_gap_factor: float = 1.8,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    errors: list[str] = []
    manifest_path = root / "franka_duo_extras" / "recording_manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": [f"Missing recording manifest: {manifest_path}"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError, UnicodeError) as exc:
        return {"valid": False, "errors": [f"Cannot read recording manifest {manifest_path}: {exc}"]}
    if not isinstance(manifest, Mapping):
        return {"valid": False, "errors": [f"Recording manifest {manifest_path} must be a JSON object"]}
    try:
        fps = int(manifest["fps"])
        cameras = manifest["cameras"]
        sync = manifest["sync"]
        robot_type = manifest["robot_type"]
        expected_numeric = {
            "action": manifest["action"],
            "observation.state": manifest["observation.state"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"valid": False, "errors": [f"Recording manifest is missing required fields: {exc}"]}
    if fps <= 0:
        return {"valid": False, "errors": [f"Recording manifest FPS must be positive, got {fps}"]}
    if not isinstance(cameras, Mapping) or not cameras:
        return {"valid": False, "errors": ["Recording manifest cameras must be a non-empty object"]}
    if not isinstance(sync, Mapping):
        return {"valid": False, "errors": ["Recording manifest sync must be an object"]}
    required_sync = {"rgb_match_tolerance_ms", "control_match_tolerance_ms", "depth_every"}
    missing_sync = required_sync - set(sync)
    if missing_sync:
        return {
            "valid": False,
            "errors": [f"Recording manifest sync is missing {sorted(missing_sync)}"],
        }

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    try:
        metadata = LeRobotDatasetMetadata(repo_id or f"local/{root.name}", root=root)
    except Exception as exc:
        return {"valid": False, "errors": [f"Cannot load LeRobot metadata from {root}: {exc}"]}
    _append(errors, str(metadata.info.codebase_version).startswith("v3."), "Dataset is not LeRobot v3")
    _append(errors, metadata.fps == fps, f"Metadata FPS {metadata.fps} != manifest FPS {fps}")
    _append(
        errors,
        metadata.robot_type == robot_type,
        f"robot_type {metadata.robot_type!r} != manifest {robot_type!r}",
    )

    for key, expected in expected_numeric.items():
        if not isinstance(expected, Mapping) or "shape" not in expected or "names" not in expected:
            return {
                "valid": False,
                "errors": [f"Recording manifest feature {key!r} is missing shape/names"],
            }
        feature = metadata.features.get(key)
        _append(errors, feature is not None, f"Missing feature {key}")
        if feature is not None:
            _append(errors, feature.get("dtype") == "float32", f"{key} is not float32")
            _append(errors, _shape(feature) == tuple(expected["shape"]), f"{key} shape mismatch")
            _append(
                errors,
                list(feature.get("names") or []) == list(expected["names"]),
                f"{key} names mismatch",
            )

    expected_video_keys = set()
    expected_video_shapes: dict[str, tuple[int, int, int]] = {}
    for key, camera in cameras.items():
        if not isinstance(camera, Mapping):
            return {
                "valid": False,
                "errors": [f"Recording manifest camera {key!r} must be an object"],
            }
        if "record_depth" not in camera:
            return {
                "valid": False,
                "errors": [f"Recording manifest camera {key!r} is missing record_depth"],
            }
        feature_key = f"observation.images.{key}"
        expected_video_keys.add(feature_key)
        try:
            expected_video_shapes[feature_key] = (
                int(camera["height"]),
                int(camera["width"]),
                3,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "valid": False,
                "errors": [f"Recording manifest camera {key!r} has invalid dimensions: {exc}"],
            }
        feature = metadata.features.get(feature_key)
        _append(errors, feature is not None, f"Missing RGB feature {feature_key}")
        if feature is not None:
            expected_shape = expected_video_shapes[feature_key]
            _append(errors, feature.get("dtype") == "video", f"{feature_key} is not video")
            _append(errors, _shape(feature) == expected_shape, f"{feature_key} shape mismatch")
    _append(
        errors,
        set(metadata.video_keys) == expected_video_keys,
        "Dataset video feature set differs from recording manifest",
    )

    parquet_rows = 0
    for parquet_path in sorted((root / "data").rglob("*.parquet")):
        frame_table = pd.read_parquet(parquet_path, columns=list(expected_numeric))
        parquet_rows += len(frame_table)
        for key, expected in expected_numeric.items():
            try:
                values = np.stack(frame_table[key].to_numpy())
            except (KeyError, ValueError) as exc:
                errors.append(f"{parquet_path}: cannot stack {key}: {exc}")
                continue
            _append(
                errors,
                values.shape == (len(frame_table), *tuple(expected["shape"])),
                f"{parquet_path}: {key} row shape mismatch",
            )
            _append(errors, np.isfinite(values).all(), f"{parquet_path}: {key} contains NaN/Inf")
    _append(errors, parquet_rows == metadata.total_frames, "Parquet row count != metadata total_frames")

    max_rgb_skew_ms = 0.0
    max_control_skew_ms = 0.0
    max_depth_skew_ms = 0.0
    effective_fps: list[float] = []
    depth_written = 0
    depth_missing = 0
    episode_rows = _episode_rows(metadata)
    _append(errors, len(episode_rows) == metadata.total_episodes, "Episode metadata count mismatch")
    video_probe_cache: dict[tuple[str, Path], tuple[float | None, list[str]]] = {}

    for episode_index, episode_row in enumerate(episode_rows):
        length = int(episode_row["length"])
        for video_key in sorted(expected_video_keys):
            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            if from_key not in episode_row or to_key not in episode_row:
                errors.append(f"Episode {episode_index}: missing video timestamps for {video_key}")
                continue
            duration = float(episode_row[to_key]) - float(episode_row[from_key])
            expected_duration = length / fps
            _append(
                errors,
                abs(duration - expected_duration) <= 0.51 / fps,
                f"Episode {episode_index}: {video_key} duration implies a frame-count mismatch",
            )
            try:
                relative_video_path = metadata.get_video_file_path(episode_index, video_key)
                video_path = (root / relative_video_path).resolve()
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                errors.append(
                    f"Episode {episode_index}: cannot resolve video reference for {video_key}: {exc}"
                )
                continue
            if not video_path.is_relative_to(root):
                errors.append(
                    f"Episode {episode_index}: video reference for {video_key} escapes dataset root: "
                    f"{video_path}"
                )
                continue
            cache_key = (video_key, video_path)
            if cache_key not in video_probe_cache:
                probe_result = _probe_video_file(
                    video_path,
                    feature_key=video_key,
                    expected_shape=expected_video_shapes[video_key],
                    expected_fps=fps,
                    fps_tolerance_ratio=fps_tolerance_ratio,
                )
                video_probe_cache[cache_key] = probe_result
                errors.extend(probe_result[1])
            physical_duration = video_probe_cache[cache_key][0]
            if physical_duration is not None:
                edge_tolerance = 0.51 / fps
                _append(
                    errors,
                    float(episode_row[from_key]) >= -edge_tolerance,
                    f"Episode {episode_index}: {video_key} starts before the referenced file",
                )
                _append(
                    errors,
                    float(episode_row[to_key]) <= physical_duration + edge_tolerance,
                    f"Episode {episode_index}: {video_key} ends after the referenced file",
                )

        episode_dir = root / "franka_duo_extras" / f"episode_{episode_index:06d}"
        sidecar_path = episode_dir / "metadata.json"
        if not sidecar_path.is_file():
            errors.append(f"Episode {episode_index}: missing sidecar metadata")
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError, UnicodeError) as exc:
            errors.append(f"Episode {episode_index}: cannot read sidecar metadata: {exc}")
            continue
        if not isinstance(sidecar, Mapping):
            errors.append(f"Episode {episode_index}: sidecar metadata must be a JSON object")
            continue
        layout = list(sidecar.get("sample_timestamp_layout", ()))
        timestamp_file = episode_dir / str(sidecar.get("sample_timestamp_file", ""))
        _append(
            errors,
            int(sidecar.get("samples_recorded", -1)) == length,
            f"Episode {episode_index}: sample count mismatch",
        )
        if not layout or not timestamp_file.is_file():
            errors.append(f"Episode {episode_index}: missing sample timestamp table")
            continue
        raw_timestamps = np.fromfile(timestamp_file, dtype="<i8")
        if raw_timestamps.size != length * len(layout):
            errors.append(f"Episode {episode_index}: sample timestamp file size mismatch")
            continue
        timestamps = raw_timestamps.reshape(length, len(layout))
        columns = {name: timestamps[:, offset] for offset, name in enumerate(layout)}
        required_timestamp_columns = {
            "frame_index",
            "state_stamp_ns",
            "action_stamp_ns",
            *(f"rgb_{key}_stamp_ns" for key in cameras),
        }
        missing_timestamp_columns = required_timestamp_columns - set(columns)
        if missing_timestamp_columns:
            errors.append(
                f"Episode {episode_index}: timestamp table is missing {sorted(missing_timestamp_columns)}"
            )
            continue
        _append(
            errors,
            np.array_equal(columns["frame_index"], np.arange(length)),
            f"Episode {episode_index}: frame indices are not contiguous",
        )
        reference_key = "head" if "head" in cameras else sorted(cameras)[0]
        reference = columns[f"rgb_{reference_key}_stamp_ns"]
        _append(errors, np.all(reference > 0), f"Episode {episode_index}: invalid RGB stamps")
        if length > 1:
            deltas = np.diff(reference)
            _append(errors, np.all(deltas > 0), f"Episode {episode_index}: repeated/non-monotonic head RGB")
            if np.all(deltas > 0):
                measured_fps = 1e9 / float(np.median(deltas))
                effective_fps.append(measured_fps)
                _append(
                    errors,
                    abs(measured_fps - fps) / fps <= fps_tolerance_ratio,
                    f"Episode {episode_index}: measured head FPS {measured_fps:.3f} is out of tolerance",
                )
                _append(
                    errors,
                    float(np.max(deltas)) <= max_frame_gap_factor * 1e9 / fps,
                    f"Episode {episode_index}: head RGB contains a timestamp gap",
                )

        for key in cameras:
            rgb = columns.get(f"rgb_{key}_stamp_ns")
            if rgb is None:
                errors.append(f"Episode {episode_index}: missing RGB stamps for {key}")
                continue
            _append(errors, np.all(np.diff(rgb) > 0), f"Episode {episode_index}: {key} RGB stamp reused")
            skew_ms = float(np.max(np.abs(rgb - reference))) / 1e6 if length else 0.0
            max_rgb_skew_ms = max(max_rgb_skew_ms, skew_ms)
            _append(
                errors,
                skew_ms <= float(sync["rgb_match_tolerance_ms"]),
                f"Episode {episode_index}: {key} RGB skew exceeds manifest tolerance",
            )
        for key in ("state_stamp_ns", "action_stamp_ns"):
            skew_ms = float(np.max(np.abs(columns[key] - reference))) / 1e6 if length else 0.0
            max_control_skew_ms = max(max_control_skew_ms, skew_ms)
            _append(
                errors,
                skew_ms <= float(sync["control_match_tolerance_ms"]),
                f"Episode {episode_index}: {key} skew exceeds manifest tolerance",
            )

        camera_info = sidecar.get("camera_info", {})
        for key, camera in cameras.items():
            info = camera_info.get(key)
            if info is None:
                errors.append(f"Episode {episode_index}: missing CameraInfo for {key}")
                continue
            expected_info_shape = (int(camera["height"]), int(camera["width"]))
            actual_info_shape = (int(info.get("height", 0)), int(info.get("width", 0)))
            _append(
                errors,
                actual_info_shape == expected_info_shape and len(info.get("k", ())) == 9,
                f"Episode {episode_index}: invalid CameraInfo for {key}",
            )

        depth_keys = [key for key, camera in cameras.items() if camera["record_depth"]]
        if not depth_keys:
            continue
        depth_key = depth_keys[0]
        count = int(sidecar.get("frames_written", -1))
        missing = np.asarray(sidecar.get("missing_rgb_frame_indices", ()), dtype=np.int64)
        depth_written += max(0, count)
        depth_missing += missing.size
        _append(
            errors,
            int(sidecar.get("frames_dropped", -1)) == 0,
            f"Episode {episode_index}: depth writer dropped frames",
        )
        required_depth_fields = {"frame_index_file", "stamp_file", "depth_file"}
        missing_depth_fields = required_depth_fields - set(sidecar)
        if missing_depth_fields:
            errors.append(
                f"Episode {episode_index}: depth metadata is missing {sorted(missing_depth_fields)}"
            )
            continue
        frame_indices = np.fromfile(episode_dir / sidecar["frame_index_file"], dtype="<i8")
        stamp_pairs = np.fromfile(episode_dir / sidecar["stamp_file"], dtype="<i8")
        shape = tuple(int(value) for value in sidecar.get("shape", ()))
        expected_depth_shape = (int(cameras[depth_key]["height"]), int(cameras[depth_key]["width"]))
        _append(errors, shape == expected_depth_shape, f"Episode {episode_index}: depth shape mismatch")
        expected_bytes = max(0, count) * int(np.prod(shape, dtype=np.int64)) * 2
        depth_path = episode_dir / sidecar["depth_file"]
        _append(
            errors,
            depth_path.is_file() and depth_path.stat().st_size == expected_bytes,
            f"Episode {episode_index}: depth file size mismatch",
        )
        _append(errors, frame_indices.size == count, f"Episode {episode_index}: depth index count mismatch")
        _append(errors, stamp_pairs.size == count * 2, f"Episode {episode_index}: depth stamp count mismatch")
        expected_indices = np.arange(0, length, int(sync["depth_every"]), dtype=np.int64)
        observed_indices = np.sort(np.concatenate((frame_indices, missing)))
        _append(
            errors,
            np.array_equal(observed_indices, expected_indices),
            f"Episode {episode_index}: depth cadence does not cover each requested RGB frame",
        )
        if count > 0 and stamp_pairs.size == count * 2:
            stamp_pairs = stamp_pairs.reshape(count, 2)
            _append(
                errors, np.all(np.diff(stamp_pairs[:, 1]) > 0), f"Episode {episode_index}: depth stamp reused"
            )
            skew_ms = float(np.max(np.abs(stamp_pairs[:, 1] - stamp_pairs[:, 0]))) / 1e6
            max_depth_skew_ms = max(max_depth_skew_ms, skew_ms)
            _append(
                errors,
                skew_ms <= float(sync["depth_match_tolerance_ms"]),
                f"Episode {episode_index}: depth/RGB skew exceeds manifest tolerance",
            )

    attempted_depth = depth_written + depth_missing
    missing_ratio = depth_missing / attempted_depth if attempted_depth else 0.0
    _append(
        errors,
        missing_ratio <= max_depth_missing_ratio,
        f"Depth missing ratio {missing_ratio:.4%} exceeds {max_depth_missing_ratio:.4%}",
    )
    return {
        "valid": not errors,
        "errors": errors,
        "dataset_root": str(root),
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "effective_head_fps_median": float(np.median(effective_fps)) if effective_fps else None,
        "max_rgb_skew_ms": max_rgb_skew_ms,
        "max_control_skew_ms": max_control_skew_ms,
        "max_depth_skew_ms": max_depth_skew_ms,
        "depth_frames_written": depth_written,
        "depth_frames_missing": depth_missing,
        "depth_missing_ratio": missing_ratio,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--max-depth-missing-ratio", type=float, default=0.0)
    parser.add_argument("--fps-tolerance-ratio", type=float, default=0.1)
    parser.add_argument("--max-frame-gap-factor", type=float, default=1.8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("max_depth_missing_ratio", "fps_tolerance_ratio"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.max_frame_gap_factor <= 1.0:
        raise ValueError("--max-frame-gap-factor must be greater than 1")
    report = validate_dataset(
        args.dataset_root,
        repo_id=args.repo_id,
        max_depth_missing_ratio=args.max_depth_missing_ratio,
        fps_tolerance_ratio=args.fps_tolerance_ratio,
        max_frame_gap_factor=args.max_frame_gap_factor,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
