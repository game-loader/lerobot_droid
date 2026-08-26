#!/usr/bin/env python3
"""Package an RL-100 checkpoint with the explicit Franka eval contract.

This command deliberately does not instantiate RL-100.  The source checkpoint
must be accompanied by a small Python factory that knows the exact Hydra policy
configuration and normalizer used during training.  The resulting manifest is
what makes an otherwise opaque ``model.pt``/``encoder.pt`` pair safe to load.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from .action_spec import FrankaDuoActionSpec, action_spec_manifest
from .franka_duo_eval_io import PointCloudConfig


def _floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RL-100 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="new self-contained bundle directory")
    parser.add_argument("--factory", required=True, help="python.module:function used by the native loader")
    parser.add_argument("--python-root", default=None, help="optional path inside the bundle for the factory")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--channels", type=int, choices=(3, 6), default=3)
    parser.add_argument("--sampling", choices=("random", "fps"), default="fps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-min", type=_floats, default=None)
    parser.add_argument("--workspace-max", type=_floats, default=None)
    parser.add_argument("--extrinsics", type=_floats, default=None, help="camera-to-training-frame 4x4")
    parser.add_argument("--state-key", default=None, help="native state input key, e.g. agent_pos")
    parser.add_argument(
        "--image-keys",
        default='{"image":["wrist_left","wrist_right"]}',
        help="JSON object mapping model image keys to wrist/head sources",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output bundle")
    return parser


def build_manifest(args: argparse.Namespace) -> dict:
    image_keys = json.loads(args.image_keys)
    if not isinstance(image_keys, dict):
        raise ValueError("--image-keys must be a JSON object")
    pointcloud = PointCloudConfig(
        num_points=args.num_points,
        channels=args.channels,
        sampling=args.sampling,
        seed=args.seed,
        workspace_min=tuple(args.workspace_min) if args.workspace_min is not None else None,
        workspace_max=tuple(args.workspace_max) if args.workspace_max is not None else None,
        extrinsics=tuple(args.extrinsics) if args.extrinsics is not None else None,
    )
    if args.workspace_min is not None and len(args.workspace_min) != 3:
        raise ValueError("--workspace-min must contain three values")
    if args.workspace_max is not None and len(args.workspace_max) != 3:
        raise ValueError("--workspace-max must contain three values")
    if args.extrinsics is not None and len(args.extrinsics) != 16:
        raise ValueError("--extrinsics must contain sixteen values")
    manifest = {
        "manifest_version": 1,
        "backend": "rl100_native",
        "action_dim": 20,
        "action_spec": action_spec_manifest(
            # Publish limits are intentionally absent unless the operator supplies them.
            # The eval CLI refuses --publish without explicit bounds.
            FrankaDuoActionSpec(
                workspace_min=tuple(args.workspace_min) if args.workspace_min is not None else None,
                workspace_max=tuple(args.workspace_max) if args.workspace_max is not None else None,
            )
        ),
        "pointcloud": {
            "num_points": pointcloud.num_points,
            "channels": pointcloud.channels,
            "sampling": pointcloud.sampling,
            "seed": pointcloud.seed,
            "workspace_min": list(pointcloud.workspace_min) if pointcloud.workspace_min else None,
            "workspace_max": list(pointcloud.workspace_max) if pointcloud.workspace_max else None,
            "extrinsics": list(pointcloud.extrinsics) if pointcloud.extrinsics else None,
        },
        "inputs": {
            "point_cloud_key": "point_cloud",
            "state_key": args.state_key,
            "image_keys": image_keys,
        },
        "native": {
            "factory": args.factory,
            "python_root": args.python_root,
            "checkpoint_dir": "checkpoint",
        },
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    for name in ("model.pt", "encoder.pt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"RL-100 checkpoint is missing {name}: {checkpoint}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists (use --force to replace): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(checkpoint, output / "checkpoint")
    manifest = build_manifest(args)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RL-100 eval bundle: {output}")
    print("The factory must load checkpoint/ with the training Hydra config and normalizer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
