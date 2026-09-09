"""Double-Q IQL with independent compact value-query Transformer heads."""

import copy

import torch
import torch.nn.functional as functional
from torch import nn

from RL.algorithms.iql import (
    _finite_float,
    _polyak_update,
    _positive_int,
    _require_finite_loss,
    compute_td_target,
    expectile_loss,
)
from RL.smolvla.attention import QueryTransformer, TokenMemory
from RL.types import DecisionBatch


class ValueTokenHead(nn.Module):
    def __init__(self, token_shape, *, hidden=128, layers=2, heads=4, value_tokens=2, **action_config):
        super().__init__()
        _positive_int("value_tokens", value_tokens)
        self.memory = TokenMemory(token_shape, hidden, **action_config)
        self.value_tokens = nn.Parameter(torch.randn(1, value_tokens, hidden) * 0.02)
        self.transformer = QueryTransformer(hidden, layers, heads)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, observation, actions=None, valid=None):
        memory, mask = self.memory(observation, actions, valid)
        queries = self.value_tokens.expand(memory.shape[0], -1, -1)
        values = self.transformer(queries, memory, mask)
        return self.readout(values.mean(1))


class TokenIQL(nn.Module):
    """Same expectile/TD/Polyak math as generic IQL; no flattened K/V MLP.

    Q1/Q2/V have disjoint trainable parameters. Target Qs are frozen copies.
    All heads consume the exact same detached VLM tokens; their small learned
    projections are value heads, not trainable replacements for SmolVLM.
    """

    def __init__(
        self,
        token_shape,
        action_dim,
        execution_steps,
        active_mask,
        *,
        hidden=128,
        layers=2,
        heads=4,
        value_tokens=2,
        expectile=0.7,
        tau=0.005,
        q_lr=3e-4,
        v_lr=3e-4,
    ):
        super().__init__()
        self.expectile = _finite_float("expectile", expectile)
        self.tau = _finite_float("tau", tau)
        if not 0 < self.expectile < 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid IQL expectile/tau")
        if _finite_float("q_lr", q_lr) <= 0 or _finite_float("v_lr", v_lr) <= 0:
            raise ValueError("IQL learning rates must be positive")
        config = {"hidden": hidden, "layers": layers, "heads": heads, "value_tokens": value_tokens}
        action = {"action_dim": action_dim, "execution_steps": execution_steps, "active_mask": active_mask}
        self.q1 = ValueTokenHead(token_shape, **config, **action)
        self.q2 = ValueTokenHead(token_shape, **config, **action)
        self.value = ValueTokenHead(token_shape, **config)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False).eval()
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=q_lr)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=v_lr)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.target_q1.eval()
        self.target_q2.eval()
        return self

    def update(self, batch: DecisionBatch):
        if not isinstance(batch, DecisionBatch):
            raise ValueError("Expected a DecisionBatch")
        self.train()
        b = batch.to(self.device)
        with torch.no_grad():
            q_bar = torch.minimum(
                self.target_q1(b.observation, b.action, b.action_valid),
                self.target_q2(b.observation, b.action, b.action_valid),
            )
            td = compute_td_target(
                reward=b.reward,
                discount=b.discount,
                done=b.done,
                next_value=self.value(b.next_observation),
            )
        value = self.value(b.observation)
        v_loss = expectile_loss(q_bar - value, expectile=self.expectile)
        q1 = self.q1(b.observation, b.action, b.action_valid)
        q2 = self.q2(b.observation, b.action, b.action_valid)
        q_loss = functional.mse_loss(q1, td) + functional.mse_loss(q2, td)
        _require_finite_loss("q_loss", q_loss)
        _require_finite_loss("v_loss", v_loss)
        self.q_optimizer.zero_grad(set_to_none=True)
        self.v_optimizer.zero_grad(set_to_none=True)
        try:
            v_loss.backward()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 10.0, error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(
                [*self.q1.parameters(), *self.q2.parameters()], 10.0, error_if_nonfinite=True
            )
        except Exception:
            self.q_optimizer.zero_grad(set_to_none=True)
            self.v_optimizer.zero_grad(set_to_none=True)
            raise
        self.v_optimizer.step()
        self.q_optimizer.step()
        _polyak_update(self.target_q1, self.q1, self.tau)
        _polyak_update(self.target_q2, self.q2, self.tau)
        return {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "q_mean": torch.minimum(q1, q2).detach().mean().item(),
            "v_mean": value.detach().mean().item(),
            "td_target_mean": td.mean().item(),
        }

    @torch.no_grad()
    def min_q(self, observation, action, action_valid):
        self.eval()
        obs = observation.to(self.device)
        action, action_valid = action.to(self.device), action_valid.to(self.device)
        result = torch.minimum(self.q1(obs, action, action_valid), self.q2(obs, action, action_valid))
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite Q value")
        return result

    @torch.no_grad()
    def advantage(self, observation, action, action_valid, *, normalize=True, eps=1e-6):
        if not isinstance(normalize, bool) or _finite_float("eps", eps) <= 0:
            raise ValueError("Invalid advantage normalization")
        raw = self.min_q(observation, action, action_valid) - self.value(observation.to(self.device))
        if normalize and raw.shape[0] > 1:
            centered = raw - raw.mean()
            variance = centered.square().mean()
            if variance.item() > eps:
                raw = centered / torch.sqrt(variance + eps)
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite advantage")
        return raw.detach()
