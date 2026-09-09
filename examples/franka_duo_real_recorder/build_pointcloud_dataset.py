#!/usr/bin/env python3
"""Build a DP3-ready LeRobot v3 dataset from matched ZED depth sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.policies.dp3.pointcloud import depth_to_point_cloud
from lerobot.utils.constants import OBS_POINT_CLOUD


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


class PointCloudBuilder:
    """Callable used by ``modify_features`` while it streams parquet files."""

    def __init__(
        self,
        extras_root: Path,
        *,
        num_points: int,
        extrinsics: np.ndarray | None,
        workspace_min: Sequence[float] | None,
        workspace_max: Sequence[float] | None,
        min_depth: float,
        max_depth: float,
        max_frame_gap: int,
        seed: int,
        require_extrinsics: bool,
    ):
        self.extras_root = Path(extras_root)
        self.num_points = int(num_points)
        self.override_extrinsics = extrinsics
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_frame_gap = int(max_frame_gap)
        self.seed = int(seed)
        self.require_extrinsics = bool(require_extrinsics)
        self._episode_index: int | None = None
        self._depth: np.memmap | None = None
        self._depth_frame_indices = np.empty(0, dtype=np.int64)
        self._camera_matrix: np.ndarray | None = None
        self._extrinsics: np.ndarray | None = None
        self._reference_camera_matrix: np.ndarray | None = None
        self._reference_extrinsics: np.ndarray | None = None
        self.source_frame_indices: list[tuple[int, int]] = []
        self.calibrations: dict[int, dict[str, Any]] = {}

    def _load_episode(self, episode_index: int) -> None:
        episode_dir = self.extras_root / f"episode_{episode_index:06d}"
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Depth sidecar metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape = tuple(int(value) for value in metadata["shape"])
        count = int(metadata["frames_written"])
        if len(shape) != 2 or min(shape, default=0) <= 0 or count <= 0:
            raise ValueError(f"Episode {episode_index} has no usable head depth frames")
        frame_indices = np.fromfile(episode_dir / metadata["frame_index_file"], dtype="<i8")
        if frame_indices.shape != (count,):
            raise ValueError(
                f"Episode {episode_index} depth frame index count {frame_indices.size} != {count}"
            )
        depth_path = episode_dir / metadata["depth_file"]
        expected_bytes = count * shape[0] * shape[1] * np.dtype("<f2").itemsize
        if depth_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Episode {episode_index} depth bytes {depth_path.stat().st_size} != {expected_bytes}"
            )

        camera_info = metadata.get("camera_info", {}).get("head")
        if not camera_info or len(camera_info.get("k", ())) != 9:
            raise ValueError(f"Episode {episode_index} is missing head CameraInfo.k")
        if (int(camera_info.get("height", 0)), int(camera_info.get("width", 0))) != shape:
            raise ValueError(
                f"Episode {episode_index} head CameraInfo shape "
                f"{(camera_info.get('height'), camera_info.get('width'))} != depth shape {shape}"
            )
        matrix = np.asarray(camera_info["k"], dtype=np.float32).reshape(3, 3)

        extrinsics = self.override_extrinsics
        if extrinsics is None and metadata.get("head_to_robot_base_transform") is not None:
            extrinsics = np.asarray(metadata["head_to_robot_base_transform"], dtype=np.float32).reshape(4, 4)
        if extrinsics is not None and (extrinsics.shape != (4, 4) or not np.isfinite(extrinsics).all()):
            raise ValueError(f"Episode {episode_index} has invalid head-to-base extrinsics")
        if self.require_extrinsics and extrinsics is None:
            raise ValueError(f"Episode {episode_index} has no head-to-base extrinsics")
        if self._reference_camera_matrix is None:
            self._reference_camera_matrix = matrix.copy()
            self._reference_extrinsics = None if extrinsics is None else extrinsics.copy()
        else:
            if not np.allclose(matrix, self._reference_camera_matrix, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Episode {episode_index} head intrinsics differ from earlier episodes")
            if (extrinsics is None) != (self._reference_extrinsics is None) or (
                extrinsics is not None
                and self._reference_extrinsics is not None
                and not np.allclose(extrinsics, self._reference_extrinsics, rtol=1e-6, atol=1e-6)
            ):
                raise ValueError(f"Episode {episode_index} head extrinsics differ from earlier episodes")

        self.calibrations[episode_index] = {
            "camera_info": camera_info,
            "effective_camera_to_training_frame": extrinsics.tolist() if extrinsics is not None else None,
        }

        self._episode_index = episode_index
        self._depth = np.memmap(depth_path, dtype="<f2", mode="r", shape=(count, *shape))
        self._depth_frame_indices = frame_indices
        self._camera_matrix = matrix
        self._extrinsics = extrinsics

    def __call__(self, row: dict[str, Any], episode_index: int, frame_in_episode: int) -> np.ndarray:
        del row
        episode_index = int(episode_index)
        frame_in_episode = int(frame_in_episode)
        if episode_index != self._episode_index:
            self._load_episode(episode_index)
        distance = np.abs(self._depth_frame_indices - frame_in_episode)
        source_offset = int(np.argmin(distance))
        gap = int(distance[source_offset])
        if gap > self.max_frame_gap:
            raise ValueError(
                f"Episode {episode_index} frame {frame_in_episode} has no depth within "
                f"{self.max_frame_gap} RGB frame(s); nearest gap is {gap}"
            )
        assert self._depth is not None
        assert self._camera_matrix is not None
        source_frame = int(self._depth_frame_indices[source_offset])
        self.source_frame_indices.append((episode_index, source_frame))
        return depth_to_point_cloud(
            np.asarray(self._depth[source_offset], dtype=np.float32),
            self._camera_matrix,
            extrinsics=self._extrinsics,
            workspace_min=self.workspace_min,
            workspace_max=self.workspace_max,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            num_points=self.num_points,
            seed=self.seed + len(self.source_frame_indices) - 1,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument(
        "--extrinsics", type=_floats, default=None, help="Row-major camera-to-training-frame 4x4"
    )
    parser.add_argument("--allow-camera-frame", action="store_true")
    parser.add_argument("--workspace-min", type=_floats, default=None, help="x,y,z lower bound")
    parser.add_argument("--workspace-max", type=_floats, default=None, help="x,y,z upper bound")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Copy RGB videos into the derived dataset (normally wasteful for DP3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if args.max_frame_gap < 0:
        raise ValueError("--max-frame-gap must be non-negative")
    if args.extrinsics is not None and len(args.extrinsics) != 16:
        raise ValueError("--extrinsics must contain 16 row-major values")
    for name, bounds in (("--workspace-min", args.workspace_min), ("--workspace-max", args.workspace_max)):
        if bounds is not None and len(bounds) != 3:
            raise ValueError(f"{name} must contain x,y,z")

    root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    repo_id = args.repo_id or f"local/{root.name}"
    output_repo_id = args.output_repo_id or f"local/{output_root.name}"

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import modify_features, recompute_stats

    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    extrinsics = (
        np.asarray(args.extrinsics, dtype=np.float32).reshape(4, 4) if args.extrinsics is not None else None
    )
    builder = PointCloudBuilder(
        root / "franka_duo_extras",
        num_points=args.num_points,
        extrinsics=extrinsics,
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frame_gap=args.max_frame_gap,
        seed=args.seed,
        require_extrinsics=not args.allow_camera_frame,
    )

    # Fail before creating the output when neither CLI nor sidecar supplies a
    # stable transform, unless the caller explicitly accepts camera-frame XYZ.
    first_meta = root / "franka_duo_extras" / "episode_000000" / "metadata.json"
    first = json.loads(first_meta.read_text(encoding="utf-8"))
    recorded_extrinsics = first.get("head_to_robot_base_transform") is not None
    if extrinsics is None and not args.allow_camera_frame and not recorded_extrinsics:
        raise ValueError(
            "No head-to-base extrinsics were recorded. Pass --extrinsics or explicitly "
            "use --allow-camera-frame."
        )

    remove_features = [] if args.keep_videos else list(dataset.meta.video_keys)
    derived_dataset = modify_features(
        dataset,
        add_features={
            OBS_POINT_CLOUD: (
                builder,
                {
                    "dtype": "float32",
                    "shape": (args.num_points, 3),
                    "names": ["point", "xyz"],
                },
            )
        },
        remove_features=remove_features or None,
        output_dir=output_root,
        repo_id=output_repo_id,
    )
    recompute_stats(derived_dataset, skip_image_video=True)

    extras = output_root / "pointcloud_extras"
    extras.mkdir(parents=True, exist_ok=True)
    np.save(
        extras / "source_depth_episode_and_frame.npy",
        np.asarray(builder.source_frame_indices, dtype=np.int64),
    )
    calibrations_json = json.dumps(builder.calibrations, indent=2, sort_keys=True)
    (extras / "calibrations.json").write_text(calibrations_json, encoding="utf-8")
    provenance = {
        "format": "franka_duo_pointcloud_v1",
        "source_dataset": str(root),
        "source_repo_id": dataset.repo_id,
        "source_codebase_version": dataset.meta.info.codebase_version,
        "source_total_episodes": dataset.meta.total_episodes,
        "source_total_frames": dataset.meta.total_frames,
        "feature": OBS_POINT_CLOUD,
        "shape": [args.num_points, 3],
        "coordinate_frame": (
            "camera"
            if args.allow_camera_frame and extrinsics is None and not recorded_extrinsics
            else "configured_stable_frame"
        ),
        "workspace_min": list(args.workspace_min) if args.workspace_min is not None else None,
        "workspace_max": list(args.workspace_max) if args.workspace_max is not None else None,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "max_frame_gap": args.max_frame_gap,
        "seed": args.seed,
        "videos_kept": args.keep_videos,
        "calibrations_file": "calibrations.json",
        "calibrations_sha256": hashlib.sha256(calibrations_json.encode("utf-8")).hexdigest(),
    }
    (extras / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Created DP3 dataset at {output_root} with {len(builder.source_frame_indices)} point clouds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
