#!/usr/bin/env python
"""Normalize real W01 LeRobot video datasets to a square resolution and merge them.

This script is intentionally kept in the repository so the real-data conversion
used for SmolVLA training is reproducible.

It creates per-source normalized copies by:
  * copying metadata/parquet files unchanged except video feature metadata;
  * transcoding every video file to the requested square resolution;
  * updating meta/info.json video feature shapes and stream info;
  * merging the normalized sources with LeRobot's aggregate_datasets.

Example:
  uv run python scripts/real_dataset/normalize_and_merge_real_square.py \
    --src /data/lerobot-imf-attnres-exp/datasets/move_pink_bowl \
    --src /data/lerobot-imf-attnres-exp/datasets/move_blue_cup \
    --src /data/lerobot-imf-attnres-exp/datasets/cloth_fold \
    --work-root /data/lerobot-imf-attnres-exp/datasets/real_512_work \
    --merged-root /data/lerobot-imf-attnres-exp/datasets/real_bowl_cup_cloth_512 \
    --merged-repo-id local/real_bowl_cup_cloth_512
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.utils import unflatten_dict

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_chunk_file(path: Path) -> tuple[int, int]:
    """Parse chunk/file indices from LeRobot paths like chunk-000/file-009.mp4."""
    chunk = int(path.parent.name.split("-")[-1])
    file = int(path.stem.split("-")[-1])
    return chunk, file


def to_numpy_tree(value):
    if isinstance(value, dict):
        return {k: to_numpy_tree(v) for k, v in value.items()}
    arr = np.asarray(value)
    # PyArrow/Pandas can produce object arrays for nested list-valued stats.
    # LeRobot's stat aggregation expects numeric ndarrays so force a numeric
    # dtype when possible.
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    return arr


def row_stats_from_episode_row(row: pd.Series) -> dict:
    stats_flat = {k[len("stats/") :]: row[k] for k in row.index if k.startswith("stats/")}
    stats = to_numpy_tree(unflatten_dict(stats_flat))
    # Pandas/PyArrow often round-trips image stats as shape (3,), while the
    # LeRobot validator/aggregator expects per-channel image stats as (3,1,1).
    # Recreate that canonical shape when repairing stats from episode rows.
    for feature_key, feature_stats in stats.items():
        if "image" not in feature_key:
            continue
        for stat_key, stat_value in list(feature_stats.items()):
            if stat_key != "count" and getattr(stat_value, "shape", None) == (3,):
                feature_stats[stat_key] = stat_value.reshape(3, 1, 1)
    return stats


def repair_copied_metadata(dst_root: Path) -> dict[str, set[tuple[int, int]]]:
    """Drop unreadable trailing parquet shards after copying a dataset.

    Some manually-recorded datasets can contain an unclosed final parquet file
    if recording was interrupted.  Such files have a valid ``PAR1`` header but
    no footer and cannot be used for training.  When this happens we keep all
    readable episodes, update ``info.json``/``stats.json``, and return the video
    files referenced by the remaining episodes so transcoding can skip unusable
    videos.  If dropping a shard would leave non-contiguous episode indices, we
    stop instead of silently producing a surprising dataset.
    """
    episode_paths = sorted((dst_root / "meta" / "episodes").glob("*/*.parquet"))
    data_paths = sorted((dst_root / "data").glob("*/*.parquet"))
    if not episode_paths or not data_paths:
        return {}

    good_episode_dfs: list[pd.DataFrame] = []
    bad_episode_paths: list[Path] = []
    for path in episode_paths:
        try:
            good_episode_dfs.append(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001 - keep processing other shards.
            print(f"WARNING: dropping unreadable episode metadata shard {path}: {exc}", flush=True)
            bad_episode_paths.append(path)

    if not good_episode_dfs:
        raise RuntimeError(f"No readable episode metadata shards under {dst_root / 'meta' / 'episodes'}")

    episodes = pd.concat(good_episode_dfs, ignore_index=True).sort_values("episode_index")
    episode_indices = [int(v) for v in episodes["episode_index"].tolist()]
    expected = list(range(len(episode_indices)))
    if episode_indices != expected:
        raise RuntimeError(
            "Refusing to repair non-contiguous episode indices after dropping unreadable shards: "
            f"got {episode_indices[:5]}...{episode_indices[-5:]}, expected 0..{len(episode_indices) - 1}"
        )

    referenced_data = {
        (int(c), int(f))
        for c, f in zip(episodes["data/chunk_index"], episodes["data/file_index"], strict=False)
    }
    for path in data_paths:
        chunk_file = parse_chunk_file(path)
        if chunk_file not in referenced_data:
            print(f"WARNING: removing unreferenced data shard {path}", flush=True)
            path.unlink()
            continue
        try:
            pd.read_parquet(path, columns=["episode_index"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Referenced data shard is unreadable and cannot be repaired: {path}") from exc

    for path in bad_episode_paths:
        path.unlink()

    video_refs: dict[str, set[tuple[int, int]]] = {}
    for col in episodes.columns:
        if not col.startswith("videos/") or not col.endswith("/chunk_index"):
            continue
        video_key = col[len("videos/") : -len("/chunk_index")]
        file_col = f"videos/{video_key}/file_index"
        video_refs[video_key] = {
            (int(c), int(f)) for c, f in zip(episodes[col], episodes[file_col], strict=False)
        }

    changed = bool(bad_episode_paths)
    info_path = dst_root / "meta" / "info.json"
    info = read_json(info_path)
    total_episodes = len(episodes)
    total_frames = int(episodes["length"].sum())
    if info.get("total_episodes") != total_episodes or info.get("total_frames") != total_frames:
        changed = True
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{total_episodes}"}
        write_json(info_path, info)

    if changed and any(k.startswith("stats/") for k in episodes.columns):
        print(
            f"WARNING: repaired {dst_root.name}: keeping episodes={total_episodes}, frames={total_frames}",
            flush=True,
        )
        write_stats(
            aggregate_stats([row_stats_from_episode_row(row) for _, row in episodes.iterrows()]), dst_root
        )

    return video_refs


def video_keys_from_info(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def choose_encoder(requested: str, crf: int, cq: int) -> tuple[str, list[str], str]:
    if requested == "libx264":
        return "libx264", ["-preset", "veryfast", "-crf", str(crf)], "yuv420p"
    if requested == "h264_nvenc":
        return "h264_nvenc", ["-preset", "p4", "-cq", str(cq)], "yuv420p"
    if requested != "auto":
        raise ValueError(f"Unsupported encoder: {requested}")
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "h264_nvenc" in probe.stdout:
        return "h264_nvenc", ["-preset", "p4", "-cq", str(cq)], "yuv420p"
    return "libx264", ["-preset", "veryfast", "-crf", str(crf)], "yuv420p"


def transcode_video(
    src: Path, dst: Path, size: int, encoder: str, encoder_args: list[str], pix_fmt: str
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep the final suffix last so ffmpeg can infer the muxer.  A path like
    # file.mp4.tmp is ambiguous to ffmpeg, while file.tmp.mp4 is recognized as mp4.
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-an",
        "-vf",
        f"scale={size}:{size}:flags=bicubic",
        "-c:v",
        encoder,
        *encoder_args,
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    run(cmd)
    tmp.replace(dst)


def copy_non_video_tree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for child in src_root.iterdir():
        if child.name == "videos":
            continue
        target = dst_root / child.name
        if child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target)


def normalize_dataset(
    src_root: Path,
    dst_root: Path,
    size: int,
    encoder: str,
    encoder_args: list[str],
    pix_fmt: str,
    workers: int,
) -> None:
    print(f"\n=== Normalize {src_root} -> {dst_root} ({size}x{size}) ===", flush=True)
    copy_non_video_tree(src_root, dst_root)
    video_refs = repair_copied_metadata(dst_root)

    src_videos = [
        p for p in (src_root / "videos").rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if video_refs:
        src_videos = [
            p
            for p in src_videos
            if p.relative_to(src_root / "videos").parts[0] in video_refs
            and parse_chunk_file(p) in video_refs[p.relative_to(src_root / "videos").parts[0]]
        ]
    if not src_videos:
        raise FileNotFoundError(f"No videos found under {src_root / 'videos'}")
    print(f"Found {len(src_videos)} videos", flush=True)

    def one(src_video: Path) -> None:
        rel = src_video.relative_to(src_root)
        transcode_video(src_video, dst_root / rel, size, encoder, encoder_args, pix_fmt)

    if workers <= 1:
        for p in src_videos:
            one(p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(one, p): p for p in src_videos}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed transcoding {p}") from exc

    info_path = dst_root / "meta" / "info.json"
    info = read_json(info_path)
    for key in video_keys_from_info(info):
        info["features"][key]["shape"] = [size, size, 3]
        # Store both legacy video_info and current info for compatibility with
        # this repo's loaders/aggregate code and older dataset metadata.
        first_video = next((dst_root / "videos" / key).rglob("*.mp4"))
        stream_info = get_video_info(first_video)
        info["features"][key]["info"] = stream_info
        info["features"][key]["video_info"] = {
            "video.fps": stream_info.get("video.fps", info.get("fps")),
            "video.codec": stream_info.get("video.codec"),
            "video.pix_fmt": stream_info.get("video.pix_fmt"),
            "video.is_depth_map": stream_info.get("video.is_depth_map", False),
            "has_audio": stream_info.get("has_audio", False),
        }
    write_json(info_path, info)

    # Validate metadata can load and sees the new shapes.
    meta = LeRobotDatasetMetadata(f"local/{dst_root.name}", root=dst_root)
    print(
        f"Validated {dst_root.name}: episodes={meta.total_episodes}, frames={meta.total_frames}, "
        f"video_shapes={[meta.features[k]['shape'] for k in meta.video_keys]}",
        flush=True,
    )


def merge_normalized(norm_roots: list[Path], merged_root: Path, merged_repo_id: str) -> None:
    print(f"\n=== Merge -> {merged_root} ({merged_repo_id}) ===", flush=True)
    if merged_root.exists():
        shutil.rmtree(merged_root)
    repo_ids = [f"local/{p.name}" for p in norm_roots]
    aggregate_datasets(
        repo_ids=repo_ids,
        roots=norm_roots,
        aggr_repo_id=merged_repo_id,
        aggr_root=merged_root,
    )
    meta = LeRobotDatasetMetadata(merged_repo_id, root=merged_root)
    print(
        f"Merged validated: episodes={meta.total_episodes}, frames={meta.total_frames}, "
        f"tasks={meta.info.total_tasks}, video_shapes={[meta.features[k]['shape'] for k in meta.video_keys]}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", action="append", required=True, help="Source LeRobot dataset root; pass multiple times"
    )
    p.add_argument("--work-root", type=Path, required=True, help="Directory for normalized per-source copies")
    p.add_argument("--merged-root", type=Path, required=True, help="Output merged dataset root")
    p.add_argument("--merged-repo-id", default="local/real_bowl_cup_cloth_512")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--encoder", choices=["auto", "h264_nvenc", "libx264"], default="auto")
    p.add_argument("--crf", type=int, default=23, help="libx264 constant-rate-factor quality")
    p.add_argument("--cq", type=int, default=23, help="h264_nvenc constant-quality value")
    p.add_argument("--workers", type=int, default=1, help="Parallel ffmpeg workers; use 1 for NVENC")
    p.add_argument("--skip-normalize", action="store_true")
    p.add_argument("--skip-merge", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_roots = [Path(s).resolve() for s in args.src]
    for src in src_roots:
        if not (src / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Not a LeRobot dataset root: {src}")
    encoder, encoder_args, pix_fmt = choose_encoder(args.encoder, args.crf, args.cq)
    print(f"Using encoder={encoder}, args={encoder_args}, pix_fmt={pix_fmt}", flush=True)

    args.work_root.mkdir(parents=True, exist_ok=True)
    norm_roots = [args.work_root / f"{src.name}_{args.size}" for src in src_roots]
    if not args.skip_normalize:
        for src, dst in zip(src_roots, norm_roots, strict=True):
            normalize_dataset(src, dst, args.size, encoder, encoder_args, pix_fmt, args.workers)

    if not args.skip_merge:
        merge_normalized(norm_roots, args.merged_root, args.merged_repo_id)


if __name__ == "__main__":
    main()
