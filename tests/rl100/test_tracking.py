# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the optional, best-effort SwanLab scalar tracker."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from RL.cli.train_offline import _parser
from RL.tracking import SwanLabConfig, create_swanlab_tracker


class _FakeRun:
    def __init__(self, *, finish_error: Exception | None = None) -> None:
        self.finish_calls = 0
        self.finish_error = finish_error

    def finish(self) -> None:
        self.finish_calls += 1
        if self.finish_error is not None:
            raise self.finish_error


class _FakeSwanLab:
    def __init__(
        self,
        *,
        init_error: Exception | None = None,
        log_error: Exception | None = None,
        finish_error: Exception | None = None,
    ) -> None:
        self.init_error = init_error
        self.log_error = log_error
        self.run = _FakeRun(finish_error=finish_error)
        self.init_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, float], int]] = []

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_calls.append(dict(kwargs))
        if self.init_error is not None:
            raise self.init_error
        return self.run

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.log_calls.append((dict(metrics), step))
        if self.log_error is not None:
            raise self.log_error


def _config(tmp_path: Path, *, strict: bool = False, mode: str = "online") -> SwanLabConfig:
    return SwanLabConfig(
        project="moya-rl100",
        run_name="offline-test",
        mode=mode,  # type: ignore[arg-type]
        log_dir=tmp_path / "swanlog",
        strict=strict,
    )


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeSwanLab) -> None:
    monkeypatch.setitem(sys.modules, "swanlab", fake)


def test_tracker_uses_swanlab_094_names_and_explicit_one_based_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSwanLab()
    _install_fake(monkeypatch, fake)
    run_config = {"dataset": {"episodes": 100}, "seed": 7}

    tracker = create_swanlab_tracker(_config(tmp_path), run_config)

    assert tracker is not None
    assert fake.init_calls == [
        {
            "project": "moya-rl100",
            "name": "offline-test",
            "mode": "online",
            "log_dir": str(tmp_path / "swanlog"),
            "config": run_config,
        }
    ]
    tracker.log({"train/loss": 0.25}, step=1)
    assert fake.log_calls == [({"train/loss": 0.25}, 1)]


def test_disabled_tracker_does_not_import_swanlab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "swanlab", sentinel)

    tracker = create_swanlab_tracker(_config(tmp_path, mode="disabled"), {})

    assert tracker is None
    assert sentinel.__dict__ == {}


def test_missing_swanlab_fails_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setitem(sys.modules, "swanlab", None)

    with pytest.raises(RuntimeError, match=r"uv run --with swanlab==0\.9\.4"):
        create_swanlab_tracker(
            SwanLabConfig(
                project="moya-rl100",
                run_name=None,
                mode="online",
                log_dir=output / "swanlog",
            ),
            {},
        )

    assert not output.exists()


def test_non_strict_initialization_failure_warns_and_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSwanLab(init_error=RuntimeError("init failed"))
    _install_fake(monkeypatch, fake)

    with pytest.warns(RuntimeWarning, match="initialization failed"):
        tracker = create_swanlab_tracker(_config(tmp_path), {})

    assert tracker is None
    assert len(fake.init_calls) == 1


def test_strict_initialization_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSwanLab(init_error=RuntimeError("init failed"))
    _install_fake(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="init failed"):
        create_swanlab_tracker(_config(tmp_path, strict=True), {})


def test_non_strict_log_failure_warns_once_then_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSwanLab(log_error=RuntimeError("log failed"))
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path), {})
    assert tracker is not None

    with pytest.warns(RuntimeWarning, match="logging failed"):
        tracker.log({"loss": 1.0}, step=1)
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        tracker.log({"loss": 0.5}, step=2)

    assert not warning_records
    assert fake.log_calls == [({"loss": 1.0}, 1)]


def test_strict_log_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSwanLab(log_error=RuntimeError("log failed"))
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path, strict=True), {})
    assert tracker is not None

    with pytest.raises(RuntimeError, match="log failed"):
        tracker.log({"loss": 1.0}, step=1)


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_tracker_rejects_non_positive_integer_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: object
) -> None:
    fake = _FakeSwanLab()
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path), {})
    assert tracker is not None

    with pytest.raises(ValueError, match="step"):
        tracker.log({"loss": 1.0}, step=step)  # type: ignore[arg-type]

    assert fake.log_calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-scalar", True])
def test_tracker_rejects_non_finite_or_non_numeric_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    fake = _FakeSwanLab()
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path), {})
    assert tracker is not None

    with pytest.raises(ValueError, match="metric"):
        tracker.log({"loss": value}, step=1)  # type: ignore[dict-item]

    assert fake.log_calls == []


def test_finish_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSwanLab()
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path), {})
    assert tracker is not None

    tracker.finish()
    tracker.finish()

    assert fake.run.finish_calls == 1


def test_finish_failure_is_warning_only_and_does_not_mask_training_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSwanLab(finish_error=RuntimeError("finish failed"))
    _install_fake(monkeypatch, fake)
    tracker = create_swanlab_tracker(_config(tmp_path, strict=True), {})
    assert tracker is not None

    with pytest.raises(ValueError, match="training failed"), pytest.warns(
        RuntimeWarning, match="finish failed"
    ):
        try:
            raise ValueError("training failed")
        finally:
            tracker.finish()

    tracker.finish()
    assert fake.run.finish_calls == 1


def test_offline_cli_parser_exposes_swanlab_controls(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--repo-id",
            "local/test",
            "--summary",
            str(tmp_path / "summary.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--swanlab-project",
            "moya-rl100",
            "--swanlab-run-name",
            "offline-run",
            "--swanlab-mode",
            "local",
            "--swanlab-strict",
        ]
    )

    assert args.swanlab_project == "moya-rl100"
    assert args.swanlab_run_name == "offline-run"
    assert args.swanlab_mode == "local"
    assert args.swanlab_strict is True
