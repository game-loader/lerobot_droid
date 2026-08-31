# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

"""Offline IQL and denoising-PPO orchestration for LeRobot Diffusion Policy."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from RL.algorithms.amq import AMQEvaluator
from RL.algorithms.dynamics import (
    DP3FeatureDynamicsEnsemble,
    PolicyPromotionGate,
    PromotionDecision,
    StateDynamicsEnsemble,
)
from RL.algorithms.iql import IQL
from RL.algorithms.ppo import (
    denoising_ppo_loss,
    denoising_ppo_metrics,
)
from RL.checkpointing import RLCounters, RLProvenance, save_rl_checkpoint
from RL.config import RLConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.tracking import ScalarTracker
from RL.types import DecisionBatch


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"metric {name!r} is non-finite: {value!r}")
        result[name] = converted
    return result


def _info_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"info/{prefix}/{name}": float(value) for name, value in metrics.items()}


def _validate_optimizer_parameters(
    optimizer: torch.optim.Optimizer, module: torch.nn.Module, *, name: str
) -> None:
    module_parameters = list(module.parameters())
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    module_ids = [id(parameter) for parameter in module_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(module_ids):
        raise ValueError(f"{name} optimizer parameters must exactly match the module parameters")


class OfflineTrainer:
    """Own old/current diffusion policies and their offline RL optimizers.

    Dataset records may be raw (the normal CLI path).  The IQL path receives a
    processor-normalized copy, while the diffusion adapter receives raw
    observations and applies the exact checkpoint processor itself.
    """

    def __init__(
        self,
        *,
        current_policy: DiffusionRLAdapter,
        old_policy: DiffusionRLAdapter,
        iql: IQL,
        actor_optimizer: torch.optim.Optimizer,
        metrics_path: str | Path,
        actor_clip_ratio: float = 0.2,
        gradient_clip_norm: float = 1.0,
        old_policy_sync_interval: int = 1,
        ppo_epochs: int = 1,
        dynamics: StateDynamicsEnsemble | DP3FeatureDynamicsEnsemble | None = None,
        amq_evaluator: AMQEvaluator | None = None,
        promotion_gate: PolicyPromotionGate | None = None,
        tracker: ScalarTracker | None = None,
        debug: bool = False,
    ) -> None:
        if not isinstance(current_policy, DiffusionRLAdapter) or not isinstance(
            old_policy, DiffusionRLAdapter
        ):
            raise ValueError("current_policy and old_policy must be DiffusionRLAdapter instances")
        if not isinstance(iql, IQL):
            raise ValueError("iql must be an IQL instance")
        if current_policy.policy is old_policy.policy:
            raise ValueError("current and old policies must be independent modules")
        current_policy.assert_transition_compatible(old_policy)
        if not torch.equal(
            current_policy.checkpoint.active_action_mask,
            old_policy.checkpoint.active_action_mask,
        ):
            raise ValueError("current and old policies must use the same active action mask")
        if not isinstance(actor_optimizer, torch.optim.Optimizer):
            raise ValueError("actor_optimizer must be a torch optimizer")
        _validate_optimizer_parameters(actor_optimizer, current_policy.policy, name="actor")
        if not isinstance(metrics_path, (str, Path)):
            raise ValueError("metrics_path must be a path")
        if isinstance(actor_clip_ratio, bool) or not isinstance(actor_clip_ratio, (int, float)):
            raise ValueError("actor_clip_ratio must be a finite number")
        if not math.isfinite(float(actor_clip_ratio)) or not 0 <= float(actor_clip_ratio) < 1:
            raise ValueError("actor_clip_ratio must lie in [0, 1)")
        if isinstance(gradient_clip_norm, bool) or not isinstance(gradient_clip_norm, (int, float)):
            raise ValueError("gradient_clip_norm must be a finite number")
        if not math.isfinite(float(gradient_clip_norm)) or float(gradient_clip_norm) <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if (
            isinstance(old_policy_sync_interval, bool)
            or not isinstance(old_policy_sync_interval, int)
            or old_policy_sync_interval < 0
        ):
            raise ValueError("old_policy_sync_interval must be a nonnegative integer")
        if isinstance(ppo_epochs, bool) or not isinstance(ppo_epochs, int) or ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        if not isinstance(debug, bool):
            raise ValueError("debug must be a bool")

        self.current_policy = current_policy
        self.old_policy = old_policy
        self.iql = iql
        self.actor_optimizer = actor_optimizer
        self.metrics_path = Path(metrics_path)
        self.actor_clip_ratio = float(actor_clip_ratio)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.old_policy_sync_interval = old_policy_sync_interval
        self.ppo_epochs = ppo_epochs
        self.dynamics = dynamics
        if amq_evaluator is not None and not isinstance(amq_evaluator, AMQEvaluator):
            raise ValueError("amq_evaluator must be an AMQEvaluator")
        if promotion_gate is not None and not isinstance(promotion_gate, PolicyPromotionGate):
            raise ValueError("promotion_gate must be a PolicyPromotionGate")
        if (amq_evaluator is None) != (promotion_gate is None):
            raise ValueError("amq_evaluator and promotion_gate must be provided together")
        if amq_evaluator is not None:
            if old_policy_sync_interval != 0:
                raise ValueError(
                    "old_policy_sync_interval must be 0 when AM-Q promotion is enabled; "
                    "behavior snapshots are gate-controlled"
                )
            if (
                amq_evaluator.candidate_policy is not current_policy
                or amq_evaluator.behavior_policy is not old_policy
            ):
                raise ValueError(
                    "AMQEvaluator must reference the trainer's current and old policies"
                )
        self.amq_evaluator = amq_evaluator
        self.promotion_gate = promotion_gate
        self.tracker = tracker
        self.debug = debug
        self.counters = RLCounters()
        self._replay_verified = False
        self._replay_info = {
            "old_replay_abs_delta_max": 0.0,
            "old_replay_abs_delta_mean": 0.0,
        }
        self._transition_info = (
            current_policy.denoising_step_diagnostics() if debug else ()
        )
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.old_policy.policy.eval()
        for parameter in self.old_policy.policy.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.current_policy.policy.parameters()).device

    def normalized_batch(self, batch: DecisionBatch) -> DecisionBatch:
        if not isinstance(batch, DecisionBatch):
            raise ValueError(f"batch must be a DecisionBatch, got {type(batch).__name__}")
        device = self.device
        normalized_observation = self.current_policy.checkpoint.normalize_observation(
            batch.observation.to(device, non_blocking=True), convert_visual_uint8=True
        )
        normalized_next = self.current_policy.checkpoint.normalize_observation(
            batch.next_observation.to(device, non_blocking=True), convert_visual_uint8=True
        )
        normalized_action = self.current_policy.checkpoint.normalize_action(
            batch.action.to(device, non_blocking=True)
        )
        return dataclasses.replace(
            batch,
            observation=normalized_observation,
            next_observation=normalized_next,
            action=normalized_action,
        )

    def _write_metrics_row(self, row: Mapping[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()

    @staticmethod
    def _validate_phase_id(phase_id: int) -> int:
        if isinstance(phase_id, bool) or not isinstance(phase_id, int) or phase_id < 0:
            raise ValueError(f"phase_id must be a nonnegative integer, got {phase_id!r}")
        return phase_id

    def record_metrics(
        self, metrics: Mapping[str, float], phase_id: int = 2
    ) -> dict[str, float]:
        """Persist one post-update row, then mirror it remotely.

        The local row and counter are committed before a tracker is called so
        strict remote failures leave an auditable partial run.
        """

        finite = _finite_metrics(dict(metrics))
        phase_id = self._validate_phase_id(phase_id)
        row_number = self.counters.metrics_rows + 1
        row: dict[str, float] = dict(finite)
        row.update(
            {
                "progress/metrics_rows": float(row_number),
                "progress/iql_updates": float(self.counters.iql_updates),
                "progress/actor_updates": float(self.counters.actor_updates),
                "progress/samples_seen": float(self.counters.samples_seen),
                "progress/decisions_seen": float(self.counters.decisions_seen),
                "progress/old_policy_syncs": float(self.counters.old_policy_syncs),
                "progress/phase_id": float(phase_id),
            }
        )
        row = _finite_metrics(row)
        self._write_metrics_row(row)
        self.counters = dataclasses.replace(self.counters, metrics_rows=row_number)
        if self.tracker is not None:
            self.tracker.log(row, step=row_number)
        return row

    def train_iql_step(self, batch: DecisionBatch) -> dict[str, float]:
        normalized = self.normalized_batch(batch)
        update_metrics = self.iql.update(normalized)
        if self.dynamics is not None:
            self.dynamics.sync_feature_encoder(self.iql.feature_encoder)
        metrics = {f"iql/{name}": value for name, value in update_metrics.items()}
        self.counters = dataclasses.replace(
            self.counters,
            iql_updates=self.counters.iql_updates + 1,
            samples_seen=self.counters.samples_seen + batch.action.shape[0],
        )
        return _finite_metrics(metrics)

    def _clip_actor_gradients(self) -> None:
        norm = torch.nn.utils.clip_grad_norm_(
            self.current_policy.policy.parameters(), self.gradient_clip_norm
        )
        if not torch.isfinite(norm).item():
            raise ValueError("actor gradients must be finite before optimizer step")

    def train_actor_step(
        self,
        batch: DecisionBatch,
        *,
        generator: torch.Generator | None = None,
        ppo_epochs: int | None = None,
    ) -> dict[str, float]:
        epochs = self.ppo_epochs if ppo_epochs is None else ppo_epochs
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("ppo_epochs must be a positive integer")
        normalized = self.normalized_batch(batch)
        with torch.no_grad():
            trace = self.old_policy.sample_trace(batch.observation, generator=generator)
            if not self._replay_verified:
                self._replay_info = self.old_policy.verify_trace_replay(batch.observation, trace)
                self._replay_verified = True
            sampled_action = trace.final_actions[:, self.current_policy.execution_slice, :]
            advantage = self.iql.advantage(
                normalized.observation,
                sampled_action,
                normalized.action_valid,
                normalize=True,
            ).reshape(-1)

        losses: list[Tensor] = []
        last_post_log_prob: Tensor | None = None
        try:
            # A behavior snapshot owns one complete stochastic trace. Every
            # candidate epoch replays that exact trace and creates fresh U-Net
            # graphs; no graph is reused after an optimizer step.
            for _epoch in range(epochs):
                self.actor_optimizer.zero_grad(set_to_none=True)
                epoch_losses: list[Tensor] = []
                try:
                    for index, new_step in enumerate(
                        self.current_policy.iter_recomputed_log_prob(batch.observation, trace)
                    ):
                        new_executable = self.current_policy.executable_log_prob(new_step)
                        old_executable = self.current_policy.executable_log_prob(
                            trace.old_log_prob[index : index + 1]
                        )
                        loss, _metrics = denoising_ppo_loss(
                            new_executable,
                            old_executable,
                            advantage,
                            step_mask=normalized.action_valid,
                            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                            clip_ratio=self.actor_clip_ratio,
                        )
                        if not torch.isfinite(loss).item():
                            raise ValueError("actor loss must be finite before backward")
                        (loss / len(self.current_policy.timesteps)).backward()
                        epoch_losses.append(loss.detach())
                    self._clip_actor_gradients()
                except Exception:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    raise
                self.actor_optimizer.step()
                if not all(
                    torch.isfinite(parameter).all().item()
                    for parameter in self.current_policy.policy.parameters()
                ):
                    raise ValueError("actor optimizer produced non-finite policy parameters")
                losses.extend(epoch_losses)
                # This probe is deliberately after optimizer.step and before
                # any behavior synchronization, so ratio/KL compare distinct
                # policies on exactly the stored old transition.
                with torch.no_grad():
                    last_post_log_prob = self.current_policy.recompute_log_prob(
                        batch.observation, trace
                    )

        except Exception:
            self.actor_optimizer.zero_grad(set_to_none=True)
            raise

        assert last_post_log_prob is not None
        self.counters = dataclasses.replace(
            self.counters,
            actor_updates=self.counters.actor_updates + epochs,
            decisions_seen=self.counters.decisions_seen + batch.action.shape[0],
        )
        old_log_prob = trace.old_log_prob.to(
            device=last_post_log_prob.device, dtype=last_post_log_prob.dtype
        )
        post_new_executable = self.current_policy.executable_log_prob(last_post_log_prob)
        post_old_executable = self.current_policy.executable_log_prob(old_log_prob)
        aggregate = denoising_ppo_metrics(
            post_new_executable,
            post_old_executable,
            step_mask=normalized.action_valid,
            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
            clip_ratio=self.actor_clip_ratio,
        )
        step_info: dict[str, float] = {}
        if self.debug:
            for index in range(len(self.current_policy.timesteps)):
                step_info.update(
                    _info_metrics(
                        f"actor/denoise_{index:02d}",
                        denoising_ppo_metrics(
                            post_new_executable[index : index + 1],
                            post_old_executable[index : index + 1],
                            step_mask=normalized.action_valid,
                            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                            clip_ratio=self.actor_clip_ratio,
                        ),
                    )
                )
        replay_info = dict(self._replay_info)
        if (
            self.old_policy_sync_interval > 0
            and self.counters.actor_updates % self.old_policy_sync_interval == 0
        ):
            self.sync_old_policy()
        metrics = {
            "actor/loss": float(torch.stack(losses).mean().item()),
            "actor/ratio_mean": aggregate["ratio_mean"],
            "actor/clip_fraction": aggregate["clip_fraction"],
            "actor/approx_kl": aggregate["approx_kl"],
            "info/actor/old_policy_sync_age_updates": float(
                self.counters.actor_updates - self.counters.last_old_policy_sync_actor_update
            ),
            "info/actor/old_policy_sync_count": float(self.counters.old_policy_syncs),
            "info/actor/ppo_epochs": float(epochs),
        }
        metrics.update(step_info)
        metrics.update(_info_metrics("actor", aggregate))
        metrics.update(_info_metrics("actor/post_update", aggregate))
        if self.debug:
            for index in range(len(self.current_policy.timesteps)):
                step_metrics = denoising_ppo_metrics(
                    post_new_executable[index : index + 1],
                    post_old_executable[index : index + 1],
                    step_mask=normalized.action_valid,
                    action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                    clip_ratio=self.actor_clip_ratio,
                )
                metrics.update(_info_metrics(f"actor/post_update/denoise_{index:02d}", step_metrics))
            for diagnostic in self._transition_info:
                step = int(diagnostic["step"])
                metrics.update(_info_metrics(f"actor/denoise_{step:02d}", diagnostic))
        metrics.update(_info_metrics("actor", replay_info))
        return _finite_metrics(metrics)

    def sync_old_policy(self) -> None:
        self.old_policy.policy.load_state_dict(self.current_policy.policy.state_dict(), strict=True)
        self.counters = dataclasses.replace(
            self.counters,
            old_policy_syncs=self.counters.old_policy_syncs + 1,
            last_old_policy_sync_actor_update=self.counters.actor_updates,
        )
        # The old trace is now generated by a new behavior snapshot. Force a
        # fresh replay contract check on the next actor step instead of
        # carrying the first snapshot's diagnostic across promotions.
        self._replay_verified = False
        self._replay_info = {
            "old_replay_abs_delta_max": 0.0,
            "old_replay_abs_delta_mean": 0.0,
        }

    def promote_behavior_policy(self, decision: PromotionDecision | bool = True) -> bool:
        """Apply an AM-Q decision and update the behavior snapshot only on accept."""

        if isinstance(decision, PromotionDecision):
            accepted = decision.promote
        elif isinstance(decision, bool):
            accepted = decision
        else:
            raise ValueError("decision must be a PromotionDecision or bool")
        self.counters = dataclasses.replace(
            self.counters, promotion_attempts=self.counters.promotion_attempts + 1
        )
        if not accepted:
            return False
        self.sync_old_policy()
        self.counters = dataclasses.replace(
            self.counters,
            promotions=self.counters.promotions + 1,
            last_promotion_actor_update=self.counters.actor_updates,
        )
        return True

    @staticmethod
    def _promotion_reason_code(reason: str) -> float:
        return {
            "promote": 1.0,
            "insufficient_return": 2.0,
            "dynamics_validation_loss": 3.0,
            "rollout_disagreement": 4.0,
            "missing_rollout_disagreement": 5.0,
        }.get(reason, 0.0)

    def evaluate_amq(
        self,
        batch: DecisionBatch,
        *,
        seed: int = 0,
    ) -> dict[str, float]:
        """Evaluate paired candidate/behavior AM-Q and conditionally promote."""

        if self.amq_evaluator is None or self.promotion_gate is None or self.dynamics is None:
            raise RuntimeError("AM-Q evaluation requires dynamics, evaluator, and promotion gate")
        normalized = self.normalized_batch(batch)
        candidate, behavior = self.amq_evaluator.paired(
            batch.observation, normalized=False, seed=seed
        )
        critic_return = float(
            self.iql.q_value(
                normalized.observation, normalized.action, normalized.action_valid
            ).mean().item()
        )
        validation_loss = self.dynamics.validation_loss(normalized)
        decision = self.promotion_gate.decide(
            candidate_return=candidate.amq_return,
            behavior_return=behavior.amq_return,
            critic_return=critic_return,
            dynamics_validation_loss=validation_loss,
            rollout_disagreement=candidate.max_disagreement,
        )
        self.promote_behavior_policy(decision)
        delta_return = decision.candidate_return - decision.behavior_return
        metrics = {
            "info/amq/candidate_return": decision.candidate_return,
            "info/amq/behavior_return": decision.behavior_return,
            "info/amq/critic_return": decision.critic_return,
            "info/amq/required_return": decision.required_return,
            "info/amq/return_delta": delta_return,
            "info/amq/delta_threshold": decision.required_return - decision.reference_return,
            "info/amq/reference_return": decision.reference_return,
            "info/amq/relative_margin": decision.relative_margin,
            "info/amq/dynamics_validation_loss": decision.dynamics_validation_loss,
            "info/amq/max_validation_loss": decision.max_validation_loss,
            "info/amq/max_rollout_disagreement": (
                float(decision.max_rollout_disagreement)
                if decision.max_rollout_disagreement is not None
                else -1.0
            ),
            "info/amq/rollout_disagreement": candidate.max_disagreement,
            "info/amq/candidate_predicted_reward": candidate.mean_predicted_reward,
            "info/amq/candidate_predicted_done": candidate.mean_predicted_done,
            "info/amq/candidate_rollout_horizon": float(candidate.rollout_horizon),
            "info/amq/behavior_predicted_reward": behavior.mean_predicted_reward,
            "info/amq/behavior_predicted_done": behavior.mean_predicted_done,
            "info/amq/behavior_rollout_horizon": float(behavior.rollout_horizon),
            "info/amq/behavior_rollout_disagreement": behavior.max_disagreement,
            "info/amq/promoted": float(decision.promote),
            "info/amq/reason_code": self._promotion_reason_code(decision.reason),
            "info/amq/promotion_attempts": float(self.counters.promotion_attempts),
            "info/amq/promotions": float(self.counters.promotions),
        }
        return _finite_metrics(metrics)

    def train_dynamics_step(self, batch: DecisionBatch) -> dict[str, float]:
        if self.dynamics is None:
            raise RuntimeError("dynamics is not configured")
        self.dynamics.sync_feature_encoder(self.iql.feature_encoder)
        metrics = {
            f"dynamics/{name}": value
            for name, value in self.dynamics.update(self.normalized_batch(batch)).items()
        }
        self.counters = dataclasses.replace(
            self.counters, dynamics_updates=self.counters.dynamics_updates + 1
        )
        return _finite_metrics(metrics)

    def train_step(
        self, batch: DecisionBatch, *, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        metrics = {}
        metrics.update(self.train_iql_step(batch))
        metrics.update(self.train_actor_step(batch, generator=generator))
        if self.dynamics is not None:
            metrics.update(self.train_dynamics_step(batch))
        self.counters = dataclasses.replace(
            self.counters, global_updates=self.counters.global_updates + 1
        )
        self.record_metrics(metrics, phase_id=2)
        return _finite_metrics(metrics)

    def save_checkpoint(
        self,
        destination: str | Path,
        *,
        provenance: RLProvenance,
        rl_config: RLConfig,
    ) -> Path:
        named_optimizers: dict[str, torch.optim.Optimizer] = {
            "iql_q": self.iql.q_optimizer,
            "iql_v": self.iql.v_optimizer,
        }
        if self.dynamics is not None:
            named_optimizers.update(
                {f"dynamics_{index}": optimizer for index, optimizer in enumerate(self.dynamics.optimizers)}
            )
        return save_rl_checkpoint(
            destination,
            current_policy=self.current_policy.checkpoint,
            old_policy=self.old_policy.policy,
            iql=self.iql,
            actor_optimizer=self.actor_optimizer,
            optimizers=named_optimizers,
            counters=self.counters,
            provenance=provenance,
            rl_config=rl_config,
            metrics_path=self.metrics_path,
            dynamics=self.dynamics,
        )
