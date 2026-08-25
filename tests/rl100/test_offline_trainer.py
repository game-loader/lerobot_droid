# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Smoke tests for the offline diffusion-RL training orchestrator."""

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
from RL.algorithms.amq import AMQEvaluator
from RL.algorithms.dynamics import PolicyPromotionGate, StateDynamicsEnsemble
from RL.algorithms.iql import IQL
from RL.cli.train_offline import _save_eval_policy_snapshot
from RL.config import TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.trainers.offline import OfflineTrainer
from RL.types import DecisionBatch, ObservationBatch


class _RecordingTracker:
    def __init__(self, metrics_path: Path | None = None) -> None:
        self.metrics_path = metrics_path
        self.calls: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        if self.metrics_path is not None:
            rows = self.metrics_path.read_text(encoding="utf-8").splitlines()
            assert len(rows) == step
        self.calls.append((dict(metrics), step))

    def finish(self) -> None:
        return None


def _adapter(root: Path) -> DiffusionRLAdapter:
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
        "observation.state": {"min": torch.full((3,), -2.0), "max": torch.full((3,), 2.0)},
        "action": {"min": torch.tensor([-2.0, 0.0]), "max": torch.tensor([2.0, 0.0])},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return DiffusionRLAdapter(
        CheckpointAdapter.load(root, device="cpu"),
        TraceConfig(num_inference_steps=2, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )


def _batch() -> DecisionBatch:
    return DecisionBatch(
        observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.5)}),
        next_observation=ObservationBatch({"observation.state": torch.full((2, 2, 3), 0.75)}),
        action=torch.tensor(
            [
                [[0.25, 0.0], [0.5, 0.0]],
                [[-0.25, 0.0], [99.0, 0.0]],
            ]
        ),
        action_valid=torch.tensor([[True, True], [True, False]]),
        reward=torch.tensor([[0.0], [1.0]]),
        done=torch.tensor([[False], [True]]),
        discount=torch.full((2, 1), 0.9),
    )


def test_eval_policy_snapshot_is_loadable_with_processor_artifacts(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "source")

    snapshot = _save_eval_policy_snapshot(adapter.checkpoint, tmp_path / "snapshot")

    assert (snapshot / "model.safetensors").is_file()
    assert (snapshot / "policy_preprocessor.json").is_file()
    assert (snapshot / "policy_postprocessor.json").is_file()
    loaded = CheckpointAdapter.load(snapshot, device="cpu")
    for expected, actual in zip(adapter.policy.parameters(), loaded.policy.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_offline_train_step_updates_current_but_not_old_without_sync(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        actor_clip_ratio=0.2,
        old_policy_sync_interval=0,
        debug=True,
    )
    current_before = [parameter.detach().clone() for parameter in current.policy.parameters()]
    old_before = [parameter.detach().clone() for parameter in old.policy.parameters()]

    metrics = trainer.train_step(_batch(), generator=current.make_generator(7))

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert "info/actor/ratio_q05" in metrics
    assert "info/actor/denoise_00/ratio_q95" in metrics
    assert "info/actor/denoise_00/sigma_inverse_square" in metrics
    assert "actor/ratio_q05" not in metrics
    assert metrics["info/actor/old_replay_abs_delta_max"] < 1e-5
    assert trainer.counters.iql_updates == 1
    assert trainer.counters.actor_updates == 1
    assert trainer.counters.old_policy_syncs == 0
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["progress/metrics_rows"] == 1
    assert row["progress/phase_id"] == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(current_before, current.policy.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after) for before, after in zip(old_before, old.policy.parameters(), strict=True)
    )
    normalized = trainer.normalized_batch(_batch())
    torch.testing.assert_close(
        normalized.observation.features["observation.state"],
        current.checkpoint.normalize_observation(_batch().observation).features["observation.state"],
    )
    torch.testing.assert_close(normalized.action, current.checkpoint.normalize_action(_batch().action))


def test_offline_actor_diagnostics_hide_per_denoising_metrics_by_default(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        old_policy_sync_interval=0,
    )

    metrics = trainer.train_actor_step(_batch(), generator=current.make_generator(13))

    assert "info/actor/ratio_q05" in metrics
    assert not any("denoise_" in key for key in metrics)
    assert "info/actor/post_update/approx_kl" in metrics


def test_ppo_epochs_replay_one_behavior_trace_and_report_post_update_kl(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-3),
        metrics_path=tmp_path / "metrics.jsonl",
        old_policy_sync_interval=0,
        ppo_epochs=2,
    )
    old_before = [parameter.detach().clone() for parameter in old.policy.parameters()]

    metrics = trainer.train_actor_step(_batch(), generator=current.make_generator(11))

    assert trainer.counters.actor_updates == 2
    assert trainer.counters.decisions_seen == 2
    assert metrics["info/actor/ppo_epochs"] == 2
    assert metrics["info/actor/old_policy_sync_count"] == 0
    assert metrics["info/actor/ratio_q95"] != pytest.approx(1.0)
    assert metrics["actor/approx_kl"] != pytest.approx(0.0)
    assert all(
        torch.equal(before, after) for before, after in zip(old_before, old.policy.parameters(), strict=True)
    )


def test_state_amq_is_paired_and_promotion_is_explicit(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    encoder = StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4)
    iql = IQL(
        feature_encoder=encoder,
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    dynamics = StateDynamicsEnsemble(
        feature_encoder=encoder,
        state_dim=3,
        n_obs_steps=2,
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        ensemble_size=2,
    )
    evaluator = AMQEvaluator(
        dynamics=dynamics,
        iql=iql,
        candidate_policy=current,
        behavior_policy=old,
        rollout_horizon=1,
    )
    candidate, behavior = evaluator.paired(_batch().observation, seed=17)
    assert candidate == behavior

    with pytest.raises(ValueError, match="old_policy_sync_interval"):
        OfflineTrainer(
            current_policy=current,
            old_policy=old,
            iql=iql,
            actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
            metrics_path=tmp_path / "invalid-metrics.jsonl",
            old_policy_sync_interval=1,
            dynamics=dynamics,
            amq_evaluator=evaluator,
            promotion_gate=PolicyPromotionGate(relative_margin=0.05, max_validation_loss=100.0),
        )

    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        old_policy_sync_interval=0,
        dynamics=dynamics,
        amq_evaluator=evaluator,
        promotion_gate=PolicyPromotionGate(relative_margin=0.05, max_validation_loss=100.0),
    )
    trainer.train_dynamics_step(_batch())
    amq_metrics = trainer.evaluate_amq(_batch(), seed=17)
    for key in (
        "info/amq/return_delta",
        "info/amq/delta_threshold",
        "info/amq/candidate_predicted_reward",
        "info/amq/behavior_predicted_done",
        "info/amq/behavior_rollout_disagreement",
    ):
        assert key in amq_metrics and torch.isfinite(torch.tensor(amq_metrics[key]))
    old_before = [parameter.detach().clone() for parameter in old.policy.parameters()]
    assert not trainer.promote_behavior_policy(False)
    assert all(
        torch.equal(before, after) for before, after in zip(old_before, old.policy.parameters(), strict=True)
    )
    with torch.no_grad():
        next(current.policy.parameters()).add_(0.1)
    assert trainer.promote_behavior_policy(True)
    assert trainer.counters.promotion_attempts == 3
    assert trainer.counters.promotions == 1
    for current_parameter, old_parameter in zip(
        current.policy.parameters(), old.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(current_parameter, old_parameter)


def test_observation_normalization_round_trip_for_amq_state(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "policy")
    raw = ObservationBatch({"observation.state": torch.tensor([[[0.25, -0.5, 1.0], [1.5, 0.0, -1.0]]])})
    normalized = adapter.checkpoint.normalize_observation(raw)
    restored = adapter.checkpoint.unnormalize_observation(normalized)
    torch.testing.assert_close(restored.features["observation.state"], raw.features["observation.state"])


def test_offline_phase_rows_are_one_based_and_mirrored_after_local_flush(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    metrics_path = tmp_path / "metrics.jsonl"
    tracker = _RecordingTracker(metrics_path)
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=metrics_path,
        actor_clip_ratio=0.2,
        tracker=tracker,
    )

    trainer.record_metrics(trainer.train_iql_step(_batch()), 0)
    trainer.record_metrics(trainer.train_actor_step(_batch(), generator=current.make_generator(7)), 1)

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [row["progress/metrics_rows"] for row in rows] == [1, 2]
    assert [row["progress/phase_id"] for row in rows] == [0, 1]
    assert [step for _metrics, step in tracker.calls] == [1, 2]
    assert rows[0]["progress/iql_updates"] == 1
    assert rows[0]["progress/actor_updates"] == 0
    assert rows[0]["progress/samples_seen"] == 2
    assert rows[0]["progress/decisions_seen"] == 0
    assert rows[1]["progress/iql_updates"] == 1
    assert rows[1]["progress/actor_updates"] == 1
    assert rows[1]["progress/samples_seen"] == 2
    assert rows[1]["progress/decisions_seen"] == 2
    assert trainer.counters.metrics_rows == 2
    assert trainer.counters.old_policy_syncs == 1


def test_strict_tracker_failure_happens_after_local_row_is_committed(tmp_path: Path) -> None:
    class FailingTracker(_RecordingTracker):
        def log(self, metrics: dict[str, float], *, step: int) -> None:
            super().log(metrics, step=step)
            raise RuntimeError("remote log failed")

    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    iql = IQL(
        feature_encoder=StateFeatureEncoder(state_dim=3, n_obs_steps=2, hidden_dims=(8,), output_dim=4),
        action_dim=2,
        chunk_size=2,
        active_action_mask=current.checkpoint.active_action_mask,
        hidden_dims=(8,),
        expectile=0.7,
        tau=0.01,
        q_lr=1e-3,
        v_lr=1e-3,
    )
    metrics_path = tmp_path / "metrics.jsonl"
    trainer = OfflineTrainer(
        current_policy=current,
        old_policy=old,
        iql=iql,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=metrics_path,
        actor_clip_ratio=0.2,
        tracker=FailingTracker(metrics_path),
    )

    with pytest.raises(RuntimeError, match="remote log failed"):
        trainer.record_metrics(trainer.train_iql_step(_batch()), 0)

    rows = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert trainer.counters.metrics_rows == 1
