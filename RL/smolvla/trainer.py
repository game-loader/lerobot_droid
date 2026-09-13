"""Token-attention IQL + flow-subpolicy PPO + coherent prefix-token AM-Q."""

from dataclasses import dataclass

import torch

from RL.algorithms.dynamics import PolicyPromotionGate
from RL.algorithms.ppo import denoising_ppo_loss, denoising_ppo_metrics
from RL.smolvla.critics import TokenIQL
from RL.smolvla.dynamics import TokenDynamics
from RL.smolvla.features import Condition
from RL.types import DecisionBatch


@dataclass
class EncodedBatch:
    current: Condition
    following: Condition
    action: torch.Tensor
    valid: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    discount: torch.Tensor

    def decision(self):
        return DecisionBatch(
            self.current.observation(),
            self.following.observation(),
            self.action,
            self.valid,
            self.reward,
            self.done,
            self.discount,
        )

    def dynamics_args(self):
        return self.current, self.following, self.action, self.valid, self.reward, self.done


class SmolVLAOfflineTrainer:
    def __init__(
        self,
        encoder,
        current,
        behavior,
        active_mask,
        example: Condition,
        *,
        actor_lr=1e-7,
        critic_lr=3e-4,
        hidden=128,
        critic_layers=2,
        critic_heads=4,
        value_tokens=2,
        ensemble=3,
        dynamics_hidden=64,
        dynamics_layers=2,
        dynamics_heads=4,
        clip_ratio=0.1,
        relative_margin=0.05,
        max_validation_loss=1.0,
        max_disagreement=0.1,
    ):
        if current.policy.model.vlm_with_expert.vlm is not behavior.policy.model.vlm_with_expert.vlm:
            raise ValueError("Current/behavior must share the same frozen SmolVLM instance")
        if current.policy.model.state_proj is not behavior.policy.model.state_proj:
            raise ValueError("Current/behavior must share the same frozen state projection")
        if actor_lr <= 0 or not 0 < clip_ratio < 1:
            raise ValueError("Invalid actor learning rate or clip ratio")
        self.encoder, self.current, self.behavior = encoder, current, behavior
        self.active_mask = active_mask
        self.clip_ratio = clip_ratio
        self.actor_optimizer = torch.optim.Adam(current.actor_parameters(), lr=actor_lr)
        self.iql = TokenIQL(
            token_shape=example.tokens.shape[1:],
            action_dim=current.action_dim,
            execution_steps=current.execution_steps,
            active_mask=active_mask,
            hidden=hidden,
            layers=critic_layers,
            heads=critic_heads,
            value_tokens=value_tokens,
            expectile=0.7,
            tau=0.005,
            q_lr=critic_lr,
            v_lr=critic_lr,
        ).to(example.kv.device)
        self.dynamics = TokenDynamics(
            example.prefix.shape[1:],
            current.action_dim,
            current.execution_steps,
            active_mask,
            ensemble=ensemble,
            hidden=dynamics_hidden,
            layers=dynamics_layers,
            heads=dynamics_heads,
        ).to(example.kv.device)
        self.gate = PolicyPromotionGate(
            relative_margin=relative_margin,
            max_validation_loss=max_validation_loss,
            max_rollout_disagreement=max_disagreement,
            use_critic_reference=False,
            inclusive_margin=True,
        )
        self.actor_updates = self.promotions = 0
        self._replay_verified = False
        self._replay_error = 0.0

    def parameter_counts(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "q1": count(self.iql.q1),
            "q2": count(self.iql.q2),
            "v": count(self.iql.value),
            "qv_trainable": sum(p.numel() for p in self.iql.parameters() if p.requires_grad),
            "target_q_frozen": count(self.iql.target_q1) + count(self.iql.target_q2),
            "dynamics_trainable": count(self.dynamics),
        }

    def encode_batch(self, raw, task):
        current, action = self.encoder.encode({**raw["observation"], "action": raw["action"]}, task)
        following, _ = self.encoder.encode(raw["next_observation"], task)
        if not torch.equal(current.mask, following.mask) or not torch.equal(
            current.dynamic_mask, following.dynamic_mask
        ):
            raise ValueError("Task/token layout must stay fixed across a decision transition")
        device = current.kv.device
        return EncodedBatch(
            current,
            following,
            action,
            raw["valid"].to(device),
            raw["reward"].to(device),
            raw["done"].to(device),
            raw["discount"].to(device),
        )

    def update_actor(self, batch: EncodedBatch, *, seed=0):
        self.current.policy.eval()
        generator = torch.Generator(device=batch.current.kv.device).manual_seed(seed)
        trace = self.behavior.sample(batch.current, generator)
        with torch.no_grad():
            # Identity replay is checked against the frozen behavior, not an evolving candidate.
            if not self._replay_verified:
                self._replay_error = max(
                    (self.behavior.replay_step(batch.current, trace, i) - trace.log_probs[i])
                    .abs()
                    .max()
                    .item()
                    for i in range(self.current.flow.steps)
                )
                if self._replay_error > 1e-4:
                    raise RuntimeError(f"Behavior log-prob replay mismatch: {self._replay_error}")
                self._replay_verified = True
            advantage = self.iql.advantage(
                batch.current.observation(), self.current.executed(trace.actions), batch.valid
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        losses = []
        for i in range(self.current.flow.steps):
            new = self.current.executed(self.current.replay_step(batch.current, trace, i)).unsqueeze(0)
            old = self.current.executed(trace.log_probs[i]).unsqueeze(0)
            loss, _ = denoising_ppo_loss(
                new,
                old,
                advantage,
                step_mask=batch.valid,
                action_dim_mask=self.active_mask,
                clip_ratio=self.clip_ratio,
            )
            (loss / self.current.flow.steps).backward()
            losses.append(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.current.actor_parameters(), 1.0, error_if_nonfinite=True
        )
        self.actor_optimizer.step()
        self.actor_updates += 1
        with torch.no_grad():
            post = torch.stack(
                [
                    self.current.executed(self.current.replay_step(batch.current, trace, i))
                    for i in range(self.current.flow.steps)
                ]
            )
            metrics = denoising_ppo_metrics(
                post,
                self.current.executed(trace.log_probs),
                step_mask=batch.valid,
                action_dim_mask=self.active_mask,
                clip_ratio=self.clip_ratio,
            )
        return {
            "actor/loss": torch.stack(losses).mean().item(),
            "actor/grad_norm": grad_norm.item(),
            "actor/replay_error": self._replay_error,
            **{f"actor/{k}": v for k, v in metrics.items()},
        }

    @torch.no_grad()
    def amq_score(self, adapter, initial: Condition, *, horizon=5, seed=0):
        if horizon < 1:
            raise ValueError("AM-Q horizon must be positive")
        condition = initial
        generator = torch.Generator(device=initial.kv.device).manual_seed(seed)
        total = torch.zeros(initial.kv.shape[0], device=initial.kv.device)
        alive = torch.ones_like(total)
        disagreement = 0.0
        valid = torch.ones(
            (initial.kv.shape[0], adapter.execution_steps), dtype=torch.bool, device=initial.kv.device
        )
        for _ in range(horizon):
            if not (alive > 0).any():
                break
            trace = adapter.sample(condition, generator)
            action = adapter.executed(trace.actions)
            # Actor K/V and Critic tokens come from the SAME real/predicted prefix.
            q = self.iql.min_q(condition.observation(), action, valid).squeeze(-1)
            total += alive * q
            condition, outcomes, variance = self.dynamics.predict(
                condition, action, valid, encoder=self.encoder
            )
            disagreement = max(disagreement, torch.where(alive > 0, variance, 0).max().item())
            done_probability = outcomes[:, 1]
            alive *= (done_probability < 0.5).float() * (1 - done_probability)
        return total.mean().item(), disagreement

    def evaluate_and_promote(self, validation: EncodedBatch, *, horizon=5, seed=0):
        val_loss = self.dynamics.validation_loss(*validation.dynamics_args())
        candidate, disagreement = self.amq_score(self.current, validation.current, horizon=horizon, seed=seed)
        baseline, old_disagreement = self.amq_score(
            self.behavior, validation.current, horizon=horizon, seed=seed
        )
        decision = self.gate.decide(
            candidate_return=candidate,
            behavior_return=baseline,
            critic_return=baseline,
            dynamics_validation_loss=val_loss,
            rollout_disagreement=max(disagreement, old_disagreement),
        )
        if decision.promote:
            self.current.synchronize_to(self.behavior)
            self.promotions += 1
            self._replay_verified = False
        return {
            "amq/candidate": candidate,
            "amq/behavior": baseline,
            "amq/validation_loss": val_loss,
            "amq/disagreement": max(disagreement, old_disagreement),
            "amq/promoted": decision.promote,
            "amq/reason": decision.reason,
            "amq/promotions": self.promotions,
        }
