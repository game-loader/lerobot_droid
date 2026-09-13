"""Attention ensemble predicting prefix tokens, not unconstrained per-layer K/V.

Future-slot queries plus reward/done queries attend to the same frozen
contextual tokens used by Q/V and ordered executed-action tokens. During AM-Q,
one shared frozen SmolVLM pass regenerates coherent Actor K/V and Critic tokens.
"""

import torch
import torch.nn.functional as functional
from torch import nn

from RL.algorithms.iql import _finite_float, _positive_int
from RL.smolvla.attention import QueryTransformer, TokenMemory


def prefix_scale(condition):
    """Detached per-token RMS: prefix magnitudes differ between image and state."""
    return condition.prefix.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(1.0)


def dynamic_error_mean(error, mask):
    """Balance visual and final-state groups, so one state token isn't swamped.

    error is [B,N,D]; returns [B]. Padding, language and image markers are
    excluded. Terminal targets are removed by the caller's mask.
    """
    token_error = error.mean(-1)
    visual_count = mask[:, :-1].sum(-1)
    visual = torch.where(mask[:, :-1], token_error[:, :-1], 0).sum(-1) / visual_count.clamp_min(1)
    state = torch.where(mask[:, -1], token_error[:, -1], 0)
    groups = (visual_count > 0).float() + mask[:, -1].float()
    return (visual + state) / groups.clamp_min(1)


class PrefixTransitionHead(nn.Module):
    def __init__(self, token_shape, action_dim, execution_steps, active_mask, hidden, layers, heads):
        super().__init__()
        count, width = token_shape
        self.memory = TokenMemory(
            token_shape,
            hidden,
            action_dim=action_dim,
            execution_steps=execution_steps,
            active_mask=active_mask,
        )
        self.future_queries = nn.Parameter(torch.randn(1, count, hidden) * 0.02)
        self.outcome_queries = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
        self.transformer = QueryTransformer(hidden, layers, heads)
        self.delta = nn.Linear(hidden, width)
        self.outcome = nn.Linear(hidden, 1)
        # Identity initialization is only an initialization, NOT evidence that
        # dynamics is trained; phase and held-out checks remain mandatory.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, condition, actions, valid):
        memory, memory_mask = self.memory(condition.observation(), actions, valid)
        count = condition.prefix.shape[1]
        future = memory[:, :count] + self.future_queries
        outcomes = self.outcome_queries.expand(memory.shape[0], -1, -1)
        queries = torch.cat((future, outcomes), 1)
        query_mask = torch.cat(
            (condition.dynamic_mask, torch.ones_like(outcomes[:, :, 0], dtype=torch.bool)), 1
        )
        hidden = self.transformer(queries, memory, memory_mask, query_mask)
        prefix = condition.prefix.detach().float()
        predicted = prefix + prefix_scale(condition) * self.delta(hidden[:, :count])
        predicted = torch.where(condition.dynamic_mask[..., None], predicted, prefix)
        return predicted, self.outcome(hidden[:, count:]).squeeze(-1)


class TokenDynamics(nn.Module):
    def __init__(
        self,
        token_shape,
        action_dim,
        execution_steps,
        active_mask,
        *,
        ensemble=3,
        hidden=64,
        layers=2,
        heads=4,
        lr=3e-4,
    ):
        super().__init__()
        _positive_int("ensemble", ensemble)
        if ensemble < 2 or _finite_float("lr", lr) <= 0:
            raise ValueError("Dynamics needs at least two heads and a positive learning rate")
        self.heads = nn.ModuleList(
            PrefixTransitionHead(token_shape, action_dim, execution_steps, active_mask, hidden, layers, heads)
            for _ in range(ensemble)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, condition, actions, valid):
        predictions = [head(condition, actions, valid) for head in self.heads]
        return torch.stack([p[0] for p in predictions]), torch.stack([p[1] for p in predictions])

    @staticmethod
    def losses(predicted, outcomes, current, following, reward, done):
        if not torch.equal(current.mask, following.mask) or not torch.equal(
            current.dynamic_mask, following.dynamic_mask
        ):
            raise ValueError("Dynamics requires a fixed task/token layout")
        # No true next observation exists for terminal windows. Do not learn
        # their placeholder frame, including in the denominator of the loss.
        mask = following.dynamic_mask & ~done
        squared = ((predicted - following.prefix.detach().float()) / prefix_scale(current)).square()
        per_sample = dynamic_error_mean(squared, mask)
        feature_loss = per_sample.sum() / mask.any(-1).sum().clamp_min(1)
        reward_loss = functional.mse_loss(outcomes[:, :1].sigmoid(), reward)
        done_loss = functional.binary_cross_entropy_with_logits(outcomes[:, 1:], done.float())
        return feature_loss, reward_loss + done_loss

    def update(self, current, following, actions, valid, reward, done):
        self.train()
        feature_losses, outcome_losses = [], []
        self.optimizer.zero_grad(set_to_none=True)
        for head in self.heads:
            indices = torch.randint(
                current.prefix.shape[0], (current.prefix.shape[0],), device=current.kv.device
            )
            c, n = current.select(indices), following.select(indices)
            predicted, outcomes = head(c, actions[indices], valid[indices])
            feature_loss, outcome_loss = self.losses(
                predicted, outcomes, c, n, reward[indices], done[indices]
            )
            loss = (feature_loss + outcome_loss) / len(self.heads)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite dynamics loss")
            loss.backward()
            feature_losses.append(feature_loss.detach())
            outcome_losses.append(outcome_loss.detach())
        nn.utils.clip_grad_norm_(self.parameters(), 10, error_if_nonfinite=True)
        self.optimizer.step()
        return {
            "dynamics/features": torch.stack(feature_losses).mean().item(),
            "dynamics/outcomes": torch.stack(outcome_losses).mean().item(),
        }

    @torch.no_grad()
    def validation_loss(self, current, following, actions, valid, reward, done):
        self.eval()
        predicted, outcomes = self(current, actions, valid)
        losses = [
            sum(self.losses(p, o, current, following, reward, done))
            for p, o in zip(predicted, outcomes, strict=True)
        ]
        return torch.stack(losses).mean().item()

    @torch.no_grad()
    def predict(self, condition, actions, valid, *, encoder):
        self.eval()
        predicted, outcomes = self(condition, actions, valid)
        variance = predicted.var(0, unbiased=False) / prefix_scale(condition).square()
        disagreement = dynamic_error_mean(variance, condition.dynamic_mask)
        # torch.where AFTER ensemble averaging preserves fixed tokens bit-for-bit.
        prefix = torch.where(
            condition.dynamic_mask[..., None], predicted.mean(0).to(condition.prefix.dtype), condition.prefix
        )
        following = encoder.from_prefix(prefix, condition.mask, condition.dynamic_mask)
        return following, outcomes.sigmoid().mean(0), disagreement
