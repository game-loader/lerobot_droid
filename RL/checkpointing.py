# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic, LeRobot-compatible checkpoints for the RL migration.

The standard policy bundle remains the only source of current-policy weights.
The auxiliary RL state is deliberately a plain ``torch.save`` mapping so it can
be loaded with ``weights_only=True`` and inspected without executing objects.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from RL.adapters.checkpoint import CheckpointAdapter, _processor_artifact_fingerprint
from RL.config import RLConfig

_FORMAT_VERSION = 1
_KIND = "rl100_diffusion_rl"
_RL100_ALGORITHM_COMMIT = "7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26"
_RL100_AUDITED_CHECKOUT = "3d52f73be4a1f7c27ed3bb32280adb36428f57e5"


def _cpu_state(value: Any) -> Any:
    """Recursively detach tensors while retaining optimizer integer keys."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("checkpoint state contains a non-finite scalar")
        return value
    raise ValueError(f"checkpoint state contains unsupported value {type(value).__name__}")


def _validate_plain(value: Any, *, path: str = "state") -> None:
    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError(f"{path} contains non-finite tensor values")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"{path} contains a non-finite scalar")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ValueError(f"{path} has an unsupported key type {type(key).__name__}")
            _validate_plain(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_plain(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _safe_relative_state_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"processor state_file must be a nonempty string, got {name!r}")
    pure = PurePosixPath(name)
    raw_parts = name.replace("\\", "/").split("/")
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"processor state_file must be a safe relative path, got {name!r}")
    unresolved = root / Path(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"processor state_file must not be a symlink: {name!r}")
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"processor state_file must remain inside the checkpoint: {name!r}")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"processor state file must be a regular non-symlink file: {name!r}")
    return candidate


def _processor_files(root: Path) -> list[tuple[Path, Path]]:
    input_root = Path(root)
    if input_root.is_symlink():
        raise ValueError(f"checkpoint source must not be a symlink: {input_root}")
    root = input_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint source must be a real directory: {root}")
    files: dict[str, Path] = {}
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config_path = root / config_name
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError(f"checkpoint is missing regular processor config {config_name!r}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid processor config {config_path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
            raise ValueError(f"processor config {config_name!r} must contain a steps list")
        files[config_name] = config_path
        for step in payload["steps"]:
            if isinstance(step, Mapping) and "state_file" in step:
                state_path = _safe_relative_state_file(root, step["state_file"])
                files[state_path.relative_to(root).as_posix()] = state_path
    return [(path, Path(relative)) for relative, path in sorted(files.items())]


def _copy_processors(source: Path, destination: Path) -> None:
    for source_path, relative_path in _processor_files(source):
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)


def _module_state(module: nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    return _cpu_state(dict(module.state_dict()))


@dataclass(frozen=True)
class RLProvenance:
    stage: str
    root_base_path: str
    root_base_hash: str
    input_checkpoint: str
    input_checkpoint_hash: str
    processor_fingerprint: str
    policy_type: str
    policy_config_hash: str
    dataset_root: str
    dataset_repo_id: str
    dataset_summary_path: str
    dataset_summary_hash: str
    feature_keys: tuple[str, ...]
    state_key: str
    state_dim: int
    chunk_size: int
    action_dim: int
    active_action_mask: tuple[bool, ...]
    lerobot_commit: str
    rl100_url: str
    rl100_algorithm_commit: str = _RL100_ALGORITHM_COMMIT
    rl100_audited_checkout_commit: str = _RL100_AUDITED_CHECKOUT
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.stage not in {"offline", "online"}:
            raise ValueError(f"stage must be 'offline' or 'online', got {self.stage!r}")
        if (
            self.rl100_algorithm_commit != _RL100_ALGORITHM_COMMIT
            or self.rl100_audited_checkout_commit != _RL100_AUDITED_CHECKOUT
        ):
            raise ValueError("RL-100 provenance commits are not the audited commits")
        feature_keys = tuple(self.feature_keys)
        active_action_mask = tuple(self.active_action_mask)
        if not feature_keys or any(not isinstance(key, str) or not key for key in feature_keys):
            raise ValueError("feature_keys must contain nonempty strings")
        if len(active_action_mask) != self.action_dim or not all(
            isinstance(value, bool) for value in active_action_mask
        ):
            raise ValueError("active_action_mask must contain one bool per action dimension")
        if self.state_dim <= 0 or self.action_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("state/action dimensions and chunk_size must be positive")
        object.__setattr__(self, "feature_keys", feature_keys)
        object.__setattr__(self, "active_action_mask", active_action_mask)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "feature_keys": list(self.feature_keys),
            "active_action_mask": list(self.active_action_mask),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RLProvenance:
        values = dict(payload)
        values["feature_keys"] = tuple(values.get("feature_keys", ()))
        values["active_action_mask"] = tuple(values.get("active_action_mask", ()))
        return cls(**values)


@dataclass(frozen=True)
class RLCounters:
    global_updates: int = 0
    iql_updates: int = 0
    actor_updates: int = 0
    dynamics_updates: int = 0
    environment_steps: int = 0
    decisions_seen: int = 0
    samples_seen: int = 0
    old_policy_syncs: int = 0
    promotion_attempts: int = 0
    promotions: int = 0
    last_old_policy_sync_actor_update: int = 0
    last_promotion_actor_update: int = 0
    metrics_rows: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field.name} must be a nonnegative integer")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RLCounters:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("checkpoint counters fields disagree")
        return cls(**{field.name: payload[field.name] for field in fields(cls)})


@dataclass(frozen=True)
class LoadedRLState:
    format_version: int
    stage: str
    old_policy_state: dict[str, Any] | None
    iql_state: dict[str, Any] | None
    dynamics_state: dict[str, Any] | None
    actor_optimizer_state: dict[str, Any] | None
    optimizer_states: dict[str, dict[str, Any]]
    scheduler_states: dict[str, dict[str, Any]]
    amp_scaler_states: dict[str, dict[str, Any]]
    rng_state: dict[str, Any]
    sampler_state: dict[str, Any] | None
    counters: RLCounters
    trainer_state: dict[str, Any]


@dataclass(frozen=True)
class LoadedRLCheckpoint:
    path: Path
    current: CheckpointAdapter
    config: RLConfig
    provenance: RLProvenance
    state: LoadedRLState
    metrics_path: Path
    manifest: dict[str, Any]


def _capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    result: dict[str, Any] = {
        "python": _cpu_state(random.getstate()),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(numpy_state[1].astype(np.uint32), dtype=torch.int64),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda_all"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    else:
        result["torch_cuda_all"] = []
    return result


def _restore_rng(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(tuple(state["python"]))
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        entries.append(
            {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {"format_version": _FORMAT_VERSION, "complete": True, "files": entries}


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in sorted([root, *[item for item in root.rglob("*") if item.is_dir()]], key=lambda item: len(item.parts), reverse=True):
        flags = getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, os.O_RDONLY | flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a directory without clobbering a concurrently-created target."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        raise OSError(
            errno.ENOTSUP, "atomic renameat2(RENAME_NOREPLACE) is required"
        ) from None
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"checkpoint destination already exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("checkpoint manifest must be a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("checkpoint manifest is incomplete")
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise ValueError("checkpoint manifest files must be a list")
    declared: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("checkpoint manifest contains an invalid file entry")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("checkpoint manifest contains an unsafe path")
        relative_name = relative.as_posix()
        if relative_name in declared:
            raise ValueError("checkpoint manifest contains duplicate paths")
        declared.add(relative_name)
        path = root / Path(*relative.parts)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != entry.get("sha256") or path.stat().st_size != entry.get("size"):
            raise ValueError(f"checkpoint manifest mismatch for {entry['path']!r}")
    all_paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("checkpoint contains an unexpected symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in all_paths
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != declared:
        raise ValueError("checkpoint manifest file set does not match the directory")
    return payload


def _validate_module_state(module: nn.Module | None, state: Mapping[str, Any] | None, *, name: str) -> None:
    if module is None:
        if state is not None:
            raise ValueError(f"{name} state is present but no module was supplied")
        return
    if not isinstance(state, Mapping):
        raise ValueError(f"{name} state is missing")
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError(f"{name} state keys disagree")
    for key, value in expected.items():
        saved = state[key]
        if (
            not isinstance(saved, Tensor)
            or saved.shape != value.shape
            or saved.dtype != value.dtype
            or saved.layout != value.layout
        ):
            raise ValueError(f"{name} state tensor mismatch for {key!r}")
        _validate_plain(saved, path=f"{name}.{key}")


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer | None,
    state: Mapping[str, Any] | None,
    *,
    name: str,
) -> None:
    if optimizer is None:
        if state is not None:
            raise ValueError(f"{name} state is present but no optimizer was supplied")
        return
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError(f"{name} optimizer state is malformed")
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError(f"{name} optimizer group count disagrees")
    saved_state = state["state"]
    if not isinstance(saved_state, Mapping):
        raise ValueError(f"{name} optimizer state entries are malformed")
    saved_ids: set[int] = set()
    for saved_group, group in zip(state["param_groups"], optimizer.param_groups, strict=True):
        if not isinstance(saved_group, Mapping) or not isinstance(saved_group.get("params"), list):
            raise ValueError(f"{name} optimizer parameter group is malformed")
        if len(saved_group["params"]) != len(group["params"]):
            raise ValueError(f"{name} optimizer parameter count disagrees")
        for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int) or saved_id in saved_ids:
                raise ValueError(f"{name} optimizer parameter ids are malformed")
            saved_ids.add(saved_id)
            entry = saved_state.get(saved_id)
            if entry is None:
                continue
            if not isinstance(entry, Mapping):
                raise ValueError(f"{name} optimizer state entry is malformed")
            for state_name, value in entry.items():
                if isinstance(value, Tensor):
                    if value.numel() != 1 and value.shape != parameter.shape:
                        raise ValueError(
                            f"{name} optimizer state {state_name!r} shape disagrees"
                        )
                    _validate_plain(value, path=f"{name}.{state_name}")
        saved_hyperparameters = set(saved_group) - {"params", "param_names"}
        current_hyperparameters = set(group) - {"params", "param_names"}
        if saved_hyperparameters != current_hyperparameters:
            raise ValueError(f"{name} optimizer hyperparameter keys disagree")
    if any(key not in saved_ids for key in saved_state):
        raise ValueError(f"{name} optimizer contains an unknown parameter state")
    _validate_plain(state, path=name)


def save_rl_checkpoint(
    destination: str | Path,
    *,
    current_policy: CheckpointAdapter,
    old_policy: nn.Module | None = None,
    iql: nn.Module | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    counters: RLCounters,
    provenance: RLProvenance,
    rl_config: RLConfig,
    metrics_path: str | Path | None = None,
    dynamics: nn.Module | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    schedulers: Mapping[str, Any] | None = None,
    amp_scalers: Mapping[str, Any] | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> Path:
    if not isinstance(current_policy, CheckpointAdapter):
        raise ValueError("current_policy must be a CheckpointAdapter")
    if not isinstance(counters, RLCounters) or not isinstance(provenance, RLProvenance):
        raise ValueError("counters and provenance must use the RL checkpoint dataclasses")
    if not isinstance(rl_config, RLConfig):
        raise ValueError("rl_config must be an RLConfig")
    if provenance.processor_fingerprint != current_policy.processor_fingerprint():
        raise ValueError("provenance processor fingerprint does not match the current policy")
    policy_config = current_policy.policy.config
    action_feature = policy_config.action_feature
    state_feature = policy_config.robot_state_feature
    if (
        action_feature is None
        or state_feature is None
        or provenance.state_dim != state_feature.shape[0]
        or provenance.action_dim != action_feature.shape[0]
        or provenance.chunk_size != policy_config.n_action_steps
        or tuple(provenance.active_action_mask)
        != tuple(current_policy.active_action_mask.tolist())
    ):
        raise ValueError("provenance dimensions or active action mask disagree with the policy")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("checkpoint destination parent must not be a symlink")
    source = current_policy.source_path
    _processor_files(source)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        pretrained = temp / "pretrained_model"
        pretrained.mkdir()
        current_policy.policy.save_pretrained(pretrained)
        _copy_processors(source, pretrained)
        if _processor_artifact_fingerprint(pretrained) != current_policy.processor_fingerprint():
            raise ValueError("copied processor fingerprint does not match source")

        metrics_payload = b""
        metric_rows = 0
        if metrics_path is not None:
            metrics_payload = Path(metrics_path).read_bytes()
            if metrics_payload and not metrics_payload.endswith(b"\n"):
                raise ValueError("metrics.jsonl must end with a newline")
            for line in metrics_payload.splitlines():
                row = json.loads(line)
                _validate_plain(row, path="metrics")
                metric_rows += 1
        if metric_rows != counters.metrics_rows:
            raise ValueError(
                f"metrics_rows={counters.metrics_rows} does not match snapshot rows={metric_rows}"
            )
        _write_bytes(temp / "metrics.jsonl", metrics_payload)
        _write_json(temp / "rl_config.json", json.loads(rl_config.to_json()))
        _write_json(temp / "provenance.json", provenance.to_dict())

        optimizer_states = {
            name: _cpu_state(optimizer.state_dict())
            for name, optimizer in (optimizers or {}).items()
        }
        payload = {
            "format_version": _FORMAT_VERSION,
            "kind": _KIND,
            "stage": provenance.stage,
            "current_policy": {
                "storage": "pretrained_model/model.safetensors",
                "sha256": _sha256_file(pretrained / "model.safetensors"),
            },
            "modules": {
                "old_policy": _module_state(old_policy),
                "iql": _module_state(iql),
                "dynamics": _module_state(dynamics),
            },
            "optimizers": {
                "actor": _cpu_state(actor_optimizer.state_dict()) if actor_optimizer else None,
                "named": optimizer_states,
            },
            "schedulers": {name: _cpu_state(item.state_dict()) for name, item in (schedulers or {}).items()},
            "amp_scalers": {name: _cpu_state(item.state_dict()) for name, item in (amp_scalers or {}).items()},
            "rng": _cpu_state(dict(rng_state) if rng_state is not None else _capture_rng()),
            "sampler": _cpu_state(dict(sampler_state)) if sampler_state is not None else None,
            "counters": asdict(counters),
            "trainer": dict(trainer_state or {}),
        }
        payload = _cpu_state(payload)
        _validate_plain(payload)
        torch.save(payload, temp / "rl_state.pt")
        _fsync_tree(temp)
        _write_json(temp / "manifest.json", _manifest(temp))
        _fsync_tree(temp)
        if destination.exists():
            raise FileExistsError(f"checkpoint destination already exists: {destination}")
        _rename_noreplace(temp, destination)
        fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return destination
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_rl_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_config: RLConfig | None = None,
    expected_provenance: RLProvenance | None = None,
    action_range_tolerance: float = 1e-8,
) -> LoadedRLCheckpoint:
    input_root = Path(checkpoint)
    if input_root.is_symlink():
        raise ValueError("checkpoint root must not be a symlink")
    root = input_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint must be a directory: {checkpoint!r}")
    manifest = _load_manifest(root)
    config = RLConfig.load_json(root / "rl_config.json")
    provenance = RLProvenance.from_dict(json.loads((root / "provenance.json").read_text(encoding="utf-8")))
    if expected_config is not None and config != expected_config:
        raise ValueError("checkpoint RL config does not match expected config")
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("checkpoint provenance does not match expected provenance")
    payload = torch.load(root / "rl_state.pt", map_location="cpu", weights_only=True)
    _validate_plain(payload)
    if not isinstance(payload, Mapping) or payload.get("kind") != _KIND or payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported RL checkpoint state format")
    current_meta = payload.get("current_policy")
    current_storage = root / "pretrained_model" / "model.safetensors"
    if (
        not isinstance(current_meta, Mapping)
        or current_meta.get("storage") != "pretrained_model/model.safetensors"
        or not current_storage.is_file()
        or _sha256_file(current_storage) != current_meta.get("sha256")
    ):
        raise ValueError("current policy storage hash is invalid")
    adapter = CheckpointAdapter.load(root / "pretrained_model", device=device, action_range_tolerance=action_range_tolerance)
    if adapter.processor_fingerprint() != provenance.processor_fingerprint:
        raise ValueError("checkpoint processor fingerprint disagrees with provenance")
    metric_rows = 0
    metrics_path = root / "metrics.jsonl"
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("metrics.jsonl contains an empty line")
        _validate_plain(json.loads(line), path="metrics")
        metric_rows += 1
    state_counters = int(payload.get("counters", {}).get("metrics_rows", -1))
    if metric_rows != state_counters:
        raise ValueError(
            f"metrics_rows={state_counters} does not match checkpoint snapshot rows={metric_rows}"
        )
    modules = payload.get("modules", {})
    optimizers = payload.get("optimizers", {})
    state = LoadedRLState(
        format_version=int(payload["format_version"]),
        stage=str(payload["stage"]),
        old_policy_state=modules.get("old_policy"),
        iql_state=modules.get("iql"),
        dynamics_state=modules.get("dynamics"),
        actor_optimizer_state=optimizers.get("actor"),
        optimizer_states=dict(optimizers.get("named", {})),
        scheduler_states=dict(payload.get("schedulers", {})),
        amp_scaler_states=dict(payload.get("amp_scalers", {})),
        rng_state=dict(payload.get("rng", {})),
        sampler_state=payload.get("sampler"),
        counters=RLCounters.from_dict(payload["counters"]),
        trainer_state=dict(payload.get("trainer", {})),
    )
    return LoadedRLCheckpoint(root, adapter, config, provenance, state, metrics_path, manifest)


def restore_rl_state(
    loaded: LoadedRLCheckpoint,
    *,
    old_policy: nn.Module | None = None,
    iql: nn.Module | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    dynamics: nn.Module | None = None,
    schedulers: Mapping[str, Any] | None = None,
    amp_scalers: Mapping[str, Any] | None = None,
    restore_rng: bool = True,
) -> RLCounters:
    if not isinstance(loaded, LoadedRLCheckpoint):
        raise ValueError("loaded must be a LoadedRLCheckpoint")
    state = loaded.state
    _validate_module_state(old_policy, state.old_policy_state, name="old_policy")
    _validate_module_state(iql, state.iql_state, name="iql")
    _validate_module_state(dynamics, state.dynamics_state, name="dynamics")
    _validate_optimizer_state(actor_optimizer, state.actor_optimizer_state, name="actor")
    for name, optimizer in (optimizers or {}).items():
        _validate_optimizer_state(optimizer, state.optimizer_states.get(name), name=name)
    if (optimizers or {}).keys() != state.optimizer_states.keys():
        raise ValueError("named optimizer keys disagree")
    if set(schedulers or {}) != set(state.scheduler_states):
        raise ValueError("scheduler keys disagree")
    if set(amp_scalers or {}) != set(state.amp_scaler_states):
        raise ValueError("AMP scaler keys disagree")
    for name, _scheduler in (schedulers or {}).items():
        if not isinstance(state.scheduler_states[name], Mapping):
            raise ValueError(f"scheduler state {name!r} is malformed")
        _validate_plain(state.scheduler_states[name], path=f"scheduler.{name}")
    for name, _scaler in (amp_scalers or {}).items():
        if not isinstance(state.amp_scaler_states[name], Mapping):
            raise ValueError(f"AMP scaler state {name!r} is malformed")
        _validate_plain(state.amp_scaler_states[name], path=f"amp_scaler.{name}")
    if old_policy is not None and state.old_policy_state is not None:
        old_policy.load_state_dict(state.old_policy_state, strict=True)
    if iql is not None and state.iql_state is not None:
        iql.load_state_dict(state.iql_state, strict=True)
    if dynamics is not None and state.dynamics_state is not None:
        dynamics.load_state_dict(state.dynamics_state, strict=True)
    if actor_optimizer is not None and state.actor_optimizer_state is not None:
        actor_optimizer.load_state_dict(state.actor_optimizer_state)
    for name, optimizer in (optimizers or {}).items():
        optimizer.load_state_dict(state.optimizer_states[name])
    for name, scheduler in (schedulers or {}).items():
        scheduler.load_state_dict(state.scheduler_states[name])
    for name, scaler in (amp_scalers or {}).items():
        scaler.load_state_dict(state.amp_scaler_states[name])
    if restore_rng:
        _restore_rng(state.rng_state)
    return state.counters
