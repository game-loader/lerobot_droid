# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Checkpoint-bound provenance for resumable policy evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVAL_PROVENANCE_FILENAME = "eval_provenance.json"
EVAL_PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: Path | str) -> str:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"cannot hash a non-file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path | str) -> str:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"cannot hash a non-directory: {root}")
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot fingerprint an empty directory: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_eval_provenance(
    checkpoint: Path | str,
    *,
    episodes: int,
    batch_size: int,
    inference_steps: int,
    policy_device: str,
    env_device: str,
    env_type: str,
    seed: int,
) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (episodes, batch_size, inference_steps)
    ):
        raise ValueError("evaluation episodes, batch_size, and inference_steps must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evaluation seed must be an integer")
    if not all(isinstance(value, str) and value for value in (policy_device, env_device, env_type)):
        raise ValueError("evaluation devices and env_type must be nonempty strings")
    resolved_checkpoint = Path(checkpoint).resolve(strict=True)
    return {
        "schema_version": EVAL_PROVENANCE_SCHEMA_VERSION,
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": sha256_tree(resolved_checkpoint),
        "episodes": episodes,
        "batch_size": batch_size,
        "inference_steps": inference_steps,
        "policy_device": policy_device,
        "env_device": env_device,
        "env_type": env_type,
        "seed": seed,
    }


def write_eval_provenance(eval_dir: Path | str, payload: Mapping[str, Any]) -> Path:
    output_dir = Path(eval_dir)
    if not output_dir.is_dir():
        raise ValueError(f"evaluation output directory does not exist: {output_dir}")
    destination = output_dir / EVAL_PROVENANCE_FILENAME
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination.resolve(strict=True)
