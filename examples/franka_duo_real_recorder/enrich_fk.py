#!/usr/bin/env python3
"""Add offline Franka end-effector poses computed from a calibrated URDF.

The recorder stores a 17D raw state.  This command reads the two 7-joint
segments and writes a sidecar array with 14 values per frame:
``left_xyzqxqyqzqw + right_xyzqxqyqzqw``.  The result is expressed in the
configured arm-base frames (or the supplied static mount transforms), never
claimed to be a world pose.

Keeping FK offline avoids adding variable CPU work and URDF dependencies to a
time-critical RGB capture loop.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from franka_kinematics import FrankaUrdfKinematics, compute_dual_fk


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default=None, help="Local repo id; defaults to local/<dataset directory>"
    )
    parser.add_argument(
        "--urdf", type=Path, required=True, help="Calibrated dual-arm or single shared Franka URDF"
    )
    parser.add_argument(
        "--left-joints", type=_csv, default=tuple(f"left_fr3v2_joint{i}" for i in range(1, 8))
    )
    parser.add_argument(
        "--right-joints", type=_csv, default=tuple(f"right_fr3v2_joint{i}" for i in range(1, 8))
    )
    parser.add_argument("--left-frame", default="left_fr3v2_link8")
    parser.add_argument("--right-frame", default="right_fr3v2_link8")
    parser.add_argument("--left-mount", type=_floats, default=None, help="xyz or xyz,qx,qy,qz,qw")
    parser.add_argument("--right-mount", type=_floats, default=None, help="xyz or xyz,qx,qy,qz,qw")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.dataset_root.resolve()
    repo_id = args.repo_id or f"local/{root.name}"
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    left_fk = FrankaUrdfKinematics.from_urdf(args.urdf, args.left_joints, args.left_frame, args.left_mount)
    right_fk = FrankaUrdfKinematics.from_urdf(
        args.urdf, args.right_joints, args.right_frame, args.right_mount
    )

    poses = np.empty((len(dataset), 14), dtype=np.float32)
    frame_indices = np.arange(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        state = np.asarray(dataset.get_raw_item(index)["observation.state"], dtype=np.float64)
        if state.shape != (17,):
            raise ValueError(f"Expected raw 17D observation.state, got {state.shape} at frame {index}")
        poses[index] = compute_dual_fk(left_fk, right_fk, state[:7], state[7:14])

    extras = root / "franka_duo_extras"
    extras.mkdir(parents=True, exist_ok=True)
    np.save(extras / "fk_ee_pose_xyzw.npy", poses)
    np.save(extras / "fk_frame_index.npy", frame_indices)
    metadata = {
        "format": "franka_duo_fk_sidecar_v1",
        "source_state": "observation.state[0:14]",
        "output": "fk_ee_pose_xyzw.npy",
        "shape": list(poses.shape),
        "order": ["left_xyzqxqyqzqw", "right_xyzqxqyqzqw"],
        "frame": "arm-base frames plus configured static mount transforms",
        "urdf": str(args.urdf),
        "left_frame": args.left_frame,
        "right_frame": args.right_frame,
        "left_joints": list(args.left_joints),
        "right_joints": list(args.right_joints),
        "left_mount": list(args.left_mount) if args.left_mount is not None else None,
        "right_mount": list(args.right_mount) if args.right_mount is not None else None,
    }
    (extras / "fk_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {poses.shape[0]} FK poses to {extras / 'fk_ee_pose_xyzw.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
