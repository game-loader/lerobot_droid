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
  uv run python scripts/real_dataset/normalize_and_merge_real_256.py \
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

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import get_video_info

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


def video_keys_from_info(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def choose_encoder(requested: str) -> tuple[str, list[str], str]:
    if requested == "libx264":
        return "libx264", ["-preset", "veryfast", "-crf", "23"], "yuv420p"
    if requested == "h264_nvenc":
        return "h264_nvenc", ["-preset", "p4", "-cq", "23"], "yuv420p"
    if requested != "auto":
        raise ValueError(f"Unsupported encoder: {requested}")
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "h264_nvenc" in probe.stdout:
        return "h264_nvenc", ["-preset", "p4", "-cq", "23"], "yuv420p"
    return "libx264", ["-preset", "veryfast", "-crf", "23"], "yuv420p"


def transcode_video(
    src: Path, dst: Path, size: int, encoder: str, encoder_args: list[str], pix_fmt: str
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
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

    src_videos = [
        p for p in (src_root / "videos").rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
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
    encoder, encoder_args, pix_fmt = choose_encoder(args.encoder)
    print(f"Using encoder={encoder}, args={encoder_args}, pix_fmt={pix_fmt}", flush=True)

    args.work_root.mkdir(parents=True, exist_ok=True)
    norm_roots = [args.work_root / f"{src.name}_256" for src in src_roots]
    if not args.skip_normalize:
        for src, dst in zip(src_roots, norm_roots, strict=True):
            normalize_dataset(src, dst, args.size, encoder, encoder_args, pix_fmt, args.workers)

    if not args.skip_merge:
        merge_normalized(norm_roots, args.merged_root, args.merged_repo_id)


if __name__ == "__main__":
    main()
