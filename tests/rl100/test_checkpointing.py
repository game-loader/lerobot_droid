# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression coverage for immutable offline-RL checkpoint bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import CheckpointAdapter
from RL.checkpointing import (
    RLCounters,
    RLProvenance,
    load_rl_checkpoint,
    restore_rl_state,
    save_rl_checkpoint,
)
from RL.config import RLConfig


def _tiny_adapter(root: Path) -> CheckpointAdapter:
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        pretrained_backbone_weights=None,
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = DiffusionPolicy(config)
    policy.save_pretrained(root)
    stats = {
        "observation.state": {
            "min": torch.tensor([-1.0, -2.0, -3.0]),
            "max": torch.tensor([1.0, 2.0, 3.0]),
        },
        "action": {
            "min": torch.tensor([-1.0, 0.0]),
            "max": torch.tensor([1.0, 0.0]),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return CheckpointAdapter.load(root, device="cpu")


def _provenance(adapter: CheckpointAdapter) -> RLProvenance:
    return RLProvenance(
        stage="offline",
        root_base_path="base",
        root_base_hash="a" * 64,
        input_checkpoint="input",
        input_checkpoint_hash="b" * 64,
        processor_fingerprint=adapter.processor_fingerprint(),
        policy_type="diffusion",
        policy_config_hash="c" * 64,
        dataset_root="dataset",
        dataset_repo_id="example/repo",
        dataset_summary_path="summary.json",
        dataset_summary_hash="d" * 64,
        feature_keys=("observation.state",),
        state_key="observation.state",
        state_dim=3,
        chunk_size=2,
        action_dim=2,
        active_action_mask=(True, False),
        lerobot_commit="e" * 40,
        rl100_url="https://example.invalid/rl100",
    )


def _step_optimizer(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in module.parameters()).backward()
    optimizer.step()


def test_immutable_bundle_round_trip_keeps_standard_policy_and_safe_state(tmp_path: Path) -> None:
    source = _tiny_adapter(tmp_path / "source")
    old_policy = DiffusionPolicy(source.policy.config)
    old_policy.load_state_dict(source.policy.state_dict(), strict=True)
    critic = torch.nn.Linear(3, 1)
    actor_optimizer = torch.optim.Adam(source.policy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    _step_optimizer(critic, critic_optimizer)
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"loss":1.0}\n', encoding="utf-8")
    destination = tmp_path / "run" / "checkpoints" / "000001"
    config = RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2)

    save_rl_checkpoint(
        destination,
        current_policy=source,
        old_policy=old_policy,
        iql=critic,
        actor_optimizer=actor_optimizer,
        optimizers={"critic": critic_optimizer},
        counters=RLCounters(global_updates=1, actor_updates=1, metrics_rows=1),
        provenance=_provenance(source),
        rl_config=config,
        metrics_path=metrics_path,
        trainer_state={"epoch": 1},
    )

    reloaded_adapter = CheckpointAdapter.load(destination / "pretrained_model", device="cpu")
    assert reloaded_adapter.processor_fingerprint() == source.processor_fingerprint()
    assert (destination / "pretrained_model" / "model.safetensors").is_file()
    assert (destination / "manifest.json").is_file()
    state_payload = torch.load(destination / "rl_state.pt", map_location="cpu", weights_only=True)
    assert isinstance(state_payload, dict)
    assert "current_policy" not in state_payload["modules"]

    loaded = load_rl_checkpoint(
        destination,
        device="cpu",
        expected_provenance=_provenance(source),
        expected_config=config,
    )
    restored_old = DiffusionPolicy(source.policy.config)
    restored_critic = torch.nn.Linear(3, 1)
    restored_actor_optimizer = torch.optim.Adam(loaded.current.policy.parameters(), lr=1e-4)
    restored_critic_optimizer = torch.optim.Adam(restored_critic.parameters(), lr=1e-3)
    restore_rl_state(
        loaded,
        old_policy=restored_old,
        iql=restored_critic,
        actor_optimizer=restored_actor_optimizer,
        optimizers={"critic": restored_critic_optimizer},
    )

    for expected, actual in zip(old_policy.parameters(), restored_old.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(critic.parameters(), restored_critic.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert restored_critic_optimizer.state_dict()["state"]


def test_checkpoint_destination_is_never_overwritten(tmp_path: Path) -> None:
    source = _tiny_adapter(tmp_path / "source")
    destination = tmp_path / "checkpoints" / "000001"
    kwargs = {
        "current_policy": source,
        "counters": RLCounters(),
        "provenance": _provenance(source),
        "rl_config": RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2),
    }

    save_rl_checkpoint(destination, **kwargs)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError):
        save_rl_checkpoint(destination, **kwargs)
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_checkpoint_rejects_changed_processor_path_without_reading_outside(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = _tiny_adapter(source_root)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"external")
    processor_path = source_root / "policy_preprocessor.json"
    payload = json.loads(processor_path.read_text(encoding="utf-8"))
    normalizer = next(step for step in payload["steps"] if "state_file" in step)
    normalizer["state_file"] = f"../{outside.name}"
    processor_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="state_file|inside|relative"):
        save_rl_checkpoint(
            tmp_path / "checkpoints" / "000001",
            current_policy=source,
            counters=RLCounters(),
            provenance=_provenance(source),
            rl_config=RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2),
        )


def test_checkpoint_rejects_symlinked_processor_state(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = _tiny_adapter(source_root)
    payload = json.loads((source_root / "policy_preprocessor.json").read_text(encoding="utf-8"))
    state_name = next(step["state_file"] for step in payload["steps"] if "state_file" in step)
    state_path = source_root / state_name
    real_path = source_root / "real-processor-state.safetensors"
    state_path.rename(real_path)
    state_path.symlink_to(real_path.name)

    with pytest.raises(ValueError, match="symlink"):
        save_rl_checkpoint(
            tmp_path / "checkpoints" / "000001",
            current_policy=source,
            counters=RLCounters(),
            provenance=_provenance(source),
            rl_config=RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2),
        )


def test_checkpoint_rejects_processor_fingerprint_provenance_mismatch(tmp_path: Path) -> None:
    source = _tiny_adapter(tmp_path / "source")
    provenance = _provenance(source)
    provenance = RLProvenance(
        **(provenance.to_dict() | {"processor_fingerprint": "f" * 64})
    )

    with pytest.raises(ValueError, match="processor fingerprint"):
        save_rl_checkpoint(
            tmp_path / "checkpoints" / "000001",
            current_policy=source,
            counters=RLCounters(),
            provenance=provenance,
            rl_config=RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2),
        )


def test_checkpoint_metrics_snapshot_must_match_counter(tmp_path: Path) -> None:
    source = _tiny_adapter(tmp_path / "source")
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"loss":1.0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="metrics_rows"):
        save_rl_checkpoint(
            tmp_path / "checkpoints" / "000001",
            current_policy=source,
            counters=RLCounters(metrics_rows=2),
            provenance=_provenance(source),
            rl_config=RLConfig(state_dim=3, action_dim=2, chunk_size=2, n_obs_steps=2),
            metrics_path=metrics_path,
        )
