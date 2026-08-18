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

from RL.algorithms.dynamics import StateDynamicsEnsemble
from RL.algorithms.iql import IQL
from RL.algorithms.ppo import denoising_ppo_loss
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
        old_policy_sync_interval: int = 0,
        dynamics: StateDynamicsEnsemble | None = None,
        tracker: ScalarTracker | None = None,
    ) -> None:
        if not isinstance(current_policy, DiffusionRLAdapter) or not isinstance(
            old_policy, DiffusionRLAdapter
        ):
            raise ValueError("current_policy and old_policy must be DiffusionRLAdapter instances")
        if not isinstance(iql, IQL):
            raise ValueError("iql must be an IQL instance")
        if current_policy.policy is old_policy.policy:
            raise ValueError("current and old policies must be independent modules")
        if current_policy.timesteps != old_policy.timesteps:
            raise ValueError("current and old policies must use the same DDIM schedule")
        if not torch.equal(
            current_policy.checkpoint.active_action_mask,
            old_policy.checkpoint.active_action_mask,
        ):
            raise ValueError("current and old policies must use the same active action mask")
        if not isinstance(actor_optimizer, torch.optim.Optimizer):
            raise ValueError("actor_optimizer must be a torch optimizer")
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

        self.current_policy = current_policy
        self.old_policy = old_policy
        self.iql = iql
        self.actor_optimizer = actor_optimizer
        self.metrics_path = Path(metrics_path)
        self.actor_clip_ratio = float(actor_clip_ratio)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.old_policy_sync_interval = old_policy_sync_interval
        self.dynamics = dynamics
        self.tracker = tracker
        self.counters = RLCounters()
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
        normalized_observation = self.current_policy.checkpoint.normalize_observation(
            batch.observation
        )
        normalized_next = self.current_policy.checkpoint.normalize_observation(
            batch.next_observation
        )
        normalized_action = self.current_policy.checkpoint.normalize_action(batch.action)
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
        metrics = {
            f"iql/{name}": value for name, value in self.iql.update(normalized).items()
        }
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
        self, batch: DecisionBatch, *, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        normalized = self.normalized_batch(batch)
        with torch.no_grad():
            trace = self.old_policy.sample_trace(batch.observation, generator=generator)
            sampled_action = trace.final_actions[:, self.current_policy.execution_slice, :]
            advantage = self.iql.advantage(
                normalized.observation,
                sampled_action,
                normalized.action_valid,
                normalize=True,
            ).reshape(-1)

        self.actor_optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        ratio_values: list[float] = []
        clip_values: list[float] = []
        kl_values: list[float] = []
        try:
            for index, new_step in enumerate(
                self.current_policy.iter_recomputed_log_prob(batch.observation, trace)
            ):
                new_executable = self.current_policy.executable_log_prob(new_step)
                old_executable = self.current_policy.executable_log_prob(
                    trace.old_log_prob[index : index + 1]
                )
                loss, metrics = denoising_ppo_loss(
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
                losses.append(loss.detach())
                ratio_values.append(metrics["ratio_mean"])
                clip_values.append(metrics["clip_fraction"])
                kl_values.append(metrics["approx_kl"])
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
        self.counters = dataclasses.replace(
            self.counters,
            actor_updates=self.counters.actor_updates + 1,
            decisions_seen=self.counters.decisions_seen + batch.action.shape[0],
        )
        if (
            self.old_policy_sync_interval > 0
            and self.counters.actor_updates % self.old_policy_sync_interval == 0
        ):
            self.sync_old_policy()
        metrics = {
            "actor/loss": float(torch.stack(losses).mean().item()),
            "actor/ratio_mean": sum(ratio_values) / len(ratio_values),
            "actor/clip_fraction": sum(clip_values) / len(clip_values),
            "actor/approx_kl": sum(kl_values) / len(kl_values),
        }
        return _finite_metrics(metrics)

    def sync_old_policy(self) -> None:
        self.old_policy.policy.load_state_dict(self.current_policy.policy.state_dict(), strict=True)
        self.counters = dataclasses.replace(
            self.counters,
            old_policy_syncs=self.counters.old_policy_syncs + 1,
            last_old_policy_sync_actor_update=self.counters.actor_updates,
        )

    def train_step(
        self, batch: DecisionBatch, *, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        metrics = {}
        metrics.update(self.train_iql_step(batch))
        metrics.update(self.train_actor_step(batch, generator=generator))
        if self.dynamics is not None:
            dynamics_metrics = self.dynamics.update(self.normalized_batch(batch))
            metrics.update({f"dynamics/{name}": value for name, value in dynamics_metrics.items()})
            self.counters = dataclasses.replace(
                self.counters, dynamics_updates=self.counters.dynamics_updates + 1
            )
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
