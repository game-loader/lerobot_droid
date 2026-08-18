# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optional scalar tracking for RL training.

Local JSONL metrics remain the source of truth.  This module only provides a
small boundary for mirroring already-persisted scalar rows to SwanLab.
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


class ScalarTracker(Protocol):
    """Sink for one-based, finite scalar metric rows."""

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Mirror one row at the supplied one-based step."""

    def finish(self) -> None:
        """Finish the remote run without masking training errors."""


@dataclass(frozen=True)
class SwanLabConfig:
    """Configuration for the optional SwanLab 0.9.4 client."""

    project: str
    run_name: str | None
    mode: Literal["online", "offline", "local", "disabled"]
    log_dir: Path
    strict: bool = False


_SWANLAB_MODES = frozenset({"online", "offline", "local", "disabled"})


def _finite_scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping of scalar values")
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"metric name must be a nonempty string, got {name!r}")
        if isinstance(value, bool):
            raise ValueError(f"metric {name!r} must be a finite scalar")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"metric {name!r} must be a finite scalar") from exc
        if not math.isfinite(converted):
            raise ValueError(f"metric {name!r} must be a finite scalar")
        result[name] = converted
    return result


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"step must be a positive integer, got {step!r}")
    return step


class _SwanLabTracker:
    def __init__(self, module: Any, run: Any, *, strict: bool) -> None:
        self._module = module
        self._run = run
        self._strict = strict
        self._disabled = False
        self._finished = False
        self._log_warning_emitted = False
        self._finish_warning_emitted = False

    def _warn_log_failure(self, error: Exception) -> None:
        if self._log_warning_emitted:
            return
        self._log_warning_emitted = True
        warnings.warn(
            f"SwanLab logging failed; disabling remote tracking: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self._disabled or self._finished:
            return
        finite = _finite_scalars(metrics)
        validated_step = _validate_step(step)
        try:
            self._module.log(finite, step=validated_step)
        except Exception as exc:
            if self._strict:
                raise
            self._disabled = True
            self._warn_log_failure(exc)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        finish = getattr(self._run, "finish", None)
        if not callable(finish):
            finish = getattr(self._module, "finish", None)
        if not callable(finish):
            return
        try:
            finish()
        except Exception as exc:
            if self._finish_warning_emitted:
                return
            self._finish_warning_emitted = True
            warnings.warn(
                f"SwanLab finish failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _missing_swanlab_error() -> RuntimeError:
    return RuntimeError(
        "SwanLab logging was requested but swanlab==0.9.4 is unavailable; "
        "install it by running: uv run --with swanlab==0.9.4 python -m "
        "RL.cli.train_offline ..."
    )


def create_swanlab_tracker(
    config: SwanLabConfig, run_config: Mapping[str, Any]
) -> ScalarTracker | None:
    """Create a SwanLab tracker, keeping non-strict failures best-effort.

    Importing SwanLab is deliberately deferred until tracking is requested so
    the disabled path has no optional dependency side effects.
    """

    if config.mode not in _SWANLAB_MODES:
        raise ValueError(f"unsupported SwanLab mode: {config.mode!r}")
    if config.mode == "disabled":
        return None
    if not isinstance(config.project, str) or not config.project.strip():
        raise ValueError("SwanLab project must be nonempty when tracking is enabled")
    if not isinstance(config.log_dir, Path):
        raise ValueError("SwanLab log_dir must be a pathlib.Path")
    if not isinstance(run_config, Mapping):
        raise ValueError("run_config must be a mapping")

    try:
        swanlab = importlib.import_module("swanlab")
    except ImportError as exc:
        raise _missing_swanlab_error() from exc

    try:
        run = swanlab.init(
            project=config.project,
            name=config.run_name,
            mode=config.mode,
            log_dir=str(config.log_dir),
            config=dict(run_config),
        )
    except Exception as exc:
        if config.strict:
            raise
        warnings.warn(
            f"SwanLab initialization failed; disabling remote tracking: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return _SwanLabTracker(swanlab, run, strict=config.strict)
