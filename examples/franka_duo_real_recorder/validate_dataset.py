#!/usr/bin/env python3
"""Validate a real Franka Duo LeRobot v3 dataset and its synchronized sidecars."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fps = int(manifest["fps"])
    cameras = manifest["cameras"]
    sync = manifest["sync"]

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id or f"local/{root.name}", root=root)
    _append(errors, str(metadata.info.codebase_version).startswith("v3."), "Dataset is not LeRobot v3")
    _append(errors, metadata.fps == fps, f"Metadata FPS {metadata.fps} != manifest FPS {fps}")
    _append(
        errors,
        metadata.robot_type == manifest["robot_type"],
        f"robot_type {metadata.robot_type!r} != manifest {manifest['robot_type']!r}",
    )

    expected_numeric = {
        "action": manifest["action"],
        "observation.state": manifest["observation.state"],
    }
    for key, expected in expected_numeric.items():
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
    for key, camera in cameras.items():
        feature_key = f"observation.images.{key}"
        expected_video_keys.add(feature_key)
        feature = metadata.features.get(feature_key)
        _append(errors, feature is not None, f"Missing RGB feature {feature_key}")
        if feature is not None:
            expected_shape = (int(camera["height"]), int(camera["width"]), 3)
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

    for episode_index, episode_row in enumerate(episode_rows):
        length = int(episode_row["length"])
        for video_key in expected_video_keys:
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

        episode_dir = root / "franka_duo_extras" / f"episode_{episode_index:06d}"
        sidecar_path = episode_dir / "metadata.json"
        if not sidecar_path.is_file():
            errors.append(f"Episode {episode_index}: missing sidecar metadata")
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
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
