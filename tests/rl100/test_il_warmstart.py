# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
pytest.importorskip("diffusers", exc_type=ModuleNotFoundError)
pytest.importorskip("accelerate", exc_type=ModuleNotFoundError)

from lerobot.scripts.lerobot_train import _policy_processor_factory_kwargs
from RL.cli.train_il_warmstart import build_train_command, processor_fingerprint


def test_fixed_pretrained_stats_are_not_overridden() -> None:
    dataset_stats = {"observation.state": {"min": [0.0], "max": [1.0]}}
    processor_kwargs, postprocessor_kwargs = _policy_processor_factory_kwargs(
        preserve_pretrained_processor_stats=True,
        processor_pretrained_path=Path("checkpoint"),
        resume=False,
        dataset_stats=dataset_stats,
        device_type="cuda",
        rename_map={},
        input_features={"observation.state": object()},
        output_features={"action": object()},
        normalization_mapping={},
    )

    assert "dataset_stats" not in processor_kwargs
    assert "normalizer_processor" not in processor_kwargs["preprocessor_overrides"]
    assert processor_kwargs["preprocessor_overrides"]["device_processor"] == {"device": "cuda"}
    assert postprocessor_kwargs == {}


def test_default_pretrained_training_uses_current_dataset_stats() -> None:
    dataset_stats = {"observation.state": {"min": [0.0], "max": [1.0]}}
    processor_kwargs, postprocessor_kwargs = _policy_processor_factory_kwargs(
        preserve_pretrained_processor_stats=False,
        processor_pretrained_path=Path("checkpoint"),
        resume=False,
        dataset_stats=dataset_stats,
        device_type="cuda",
        rename_map={},
        input_features={"observation.state": object()},
        output_features={"action": object()},
        normalization_mapping={},
    )

    assert processor_kwargs["dataset_stats"] is dataset_stats
    assert processor_kwargs["preprocessor_overrides"]["normalizer_processor"]["stats"] is dataset_stats
    assert postprocessor_kwargs["postprocessor_overrides"]["unnormalizer_processor"]["stats"] is dataset_stats


def _checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"model")
    return root


def test_il_command_enables_fixed_processor_stats(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    args = Namespace(
        checkpoint=checkpoint,
        dataset_root=dataset,
        repo_id="local/moya-round-1",
        output_dir=tmp_path / "output",
        steps=2,
        batch_size=2,
        num_workers=1,
        save_freq=2,
        log_freq=1,
        learning_rate=1e-5,
        seed=1000,
        device="cuda",
        swanlab_project="project",
        swanlab_run_name=None,
        swanlab_mode="disabled",
        dry_run=True,
    )

    command = build_train_command(args)

    assert "--preserve_pretrained_processor_stats=true" in command
    assert "--resume=false" in command
    assert "--env_eval_freq=5000" in command
    assert not any(arg.startswith("--eval_freq=") for arg in command)
    assert f"--policy.path={checkpoint.resolve()}" in command
    assert f"--dataset.root={dataset.resolve()}" in command


def test_il_command_rejects_missing_checkpoint(tmp_path: Path) -> None:
    args = Namespace(checkpoint=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        build_train_command(args)


def _processor_checkpoint(root: Path, *, scalar_auxiliary_stats: bool) -> Path:
    root.mkdir(parents=True)
    configs = {
        "policy_preprocessor.json": {
            "registry_name": "normalizer_processor",
            "features": {
                "observation.state": {"type": "STATE", "shape": [2]},
                "action": {"type": "ACTION", "shape": [1]},
            },
            "state_file": "preprocessor.safetensors",
        },
        "policy_postprocessor.json": {
            "registry_name": "unnormalizer_processor",
            "features": {"action": {"type": "ACTION", "shape": [1]}},
            "state_file": "postprocessor.safetensors",
        },
    }
    norm_map = {"STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    for filename, spec in configs.items():
        payload = {
            "steps": [
                {
                    "registry_name": spec["registry_name"],
                    "config": {
                        "eps": 1e-8,
                        "features": spec["features"],
                        "norm_map": norm_map,
                    },
                    "state_file": spec["state_file"],
                }
            ]
        }
        (root / filename).write_text(json.dumps(payload), encoding="utf-8")

    auxiliary = torch.tensor(42.0) if scalar_auxiliary_stats else torch.tensor([42.0])
    save_file(
        {
            "observation.state.min": torch.tensor([-1.0, -2.0]),
            "observation.state.max": torch.tensor([1.0, 2.0]),
            "observation.state.count": auxiliary,
            "action.min": torch.tensor([-0.5]),
            "action.max": torch.tensor([0.5]),
            "action.count": auxiliary.clone(),
        },
        root / "preprocessor.safetensors",
    )
    save_file(
        {
            "action.min": torch.tensor([-0.5]),
            "action.max": torch.tensor([0.5]),
            "action.count": auxiliary,
        },
        root / "postprocessor.safetensors",
    )
    return root


def test_processor_fingerprint_uses_runtime_normalization_semantics(tmp_path: Path) -> None:
    vector_aux = _processor_checkpoint(tmp_path / "vector_aux", scalar_auxiliary_stats=False)
    scalar_aux = _processor_checkpoint(tmp_path / "scalar_aux", scalar_auxiliary_stats=True)

    assert processor_fingerprint(vector_aux) == processor_fingerprint(scalar_aux)

    state = {
        "observation.state.min": torch.tensor([-1.0, -2.0]),
        "observation.state.max": torch.tensor([1.0, 2.0]),
        "observation.state.count": torch.tensor(42.0),
        "action.min": torch.tensor([-0.5]),
        "action.max": torch.tensor([0.75]),
        "action.count": torch.tensor(42.0),
    }
    save_file(state, scalar_aux / "preprocessor.safetensors")

    assert processor_fingerprint(vector_aux) != processor_fingerprint(scalar_aux)


def test_processor_fingerprint_rejects_unknown_normalization_mode(tmp_path: Path) -> None:
    checkpoint = _processor_checkpoint(tmp_path / "unknown_mode", scalar_auxiliary_stats=False)
    config_path = checkpoint / "policy_preprocessor.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["steps"][0]["config"]["norm_map"]["STATE"] = "NOT_A_MODE"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported normalization mode"):
        processor_fingerprint(checkpoint)
