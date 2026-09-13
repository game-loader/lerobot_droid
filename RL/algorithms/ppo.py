# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Masked PPO objective over stochastic diffusion denoising transitions."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_LOG_RATIO_EXACT_LIMIT = 20.0


def _stable_probability_ratio(log_ratio: Tensor) -> Tensor:
    """Exponentiate exactly in the useful range and use a finite smooth tail."""

    capped = log_ratio.clamp(max=_LOG_RATIO_EXACT_LIMIT)
    exact = torch.exp(capped)
    excess = (log_ratio - _LOG_RATIO_EXACT_LIMIT).clamp_min(0.0)
    limit_ratio = math.exp(_LOG_RATIO_EXACT_LIMIT)
    upper_tail = limit_ratio * (1.0 + excess / (1.0 + excess))
    return torch.where(log_ratio <= _LOG_RATIO_EXACT_LIMIT, exact, upper_tail)


def _validate_log_prob(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [denoise,batch,step,dim], got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype, got {value.dtype}")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be nonzero, got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def reduce_event_log_prob(log_prob: Tensor, *, step_mask: Tensor, action_dim_mask: Tensor) -> Tensor:
    """Sum valid executable action events while preserving denoising and batch axes."""

    _validate_log_prob("log_prob", log_prob)
    _, batch_size, action_steps, action_dim = log_prob.shape
    if not isinstance(step_mask, Tensor) or step_mask.dtype != torch.bool:
        raise ValueError("step_mask must be a boolean torch.Tensor")
    if step_mask.shape != (batch_size, action_steps):
        raise ValueError(
            f"step_mask must have shape {(batch_size, action_steps)}, got {tuple(step_mask.shape)}"
        )
    if not isinstance(action_dim_mask, Tensor) or action_dim_mask.dtype != torch.bool:
        raise ValueError("action_dim_mask must be a boolean torch.Tensor")
    if action_dim_mask.shape != (action_dim,):
        raise ValueError(
            f"action_dim_mask must have shape {(action_dim,)}, got {tuple(action_dim_mask.shape)}"
        )
    if not action_dim_mask.any().item():
        raise ValueError("action_dim_mask must select at least one action dimension")
    if not step_mask.any(dim=1).all().item():
        raise ValueError("step_mask must select at least one action step per batch item")

    event_mask = step_mask.to(log_prob.device).unsqueeze(0).unsqueeze(-1)
    event_mask = event_mask & action_dim_mask.to(log_prob.device).view(1, 1, 1, -1)
    reduced = torch.where(event_mask, log_prob.float(), 0.0).sum(dim=(-1, -2))
    if not torch.isfinite(reduced).all().item():
        raise ValueError("reduced log probability contains non-finite values")
    return reduced


def _validate_clip_ratio(clip_ratio: float) -> float:
    if not isinstance(clip_ratio, (int, float)) or isinstance(clip_ratio, bool):
        raise ValueError(f"clip_ratio must be a finite number, got {clip_ratio!r}")
    clip_ratio = float(clip_ratio)
    if not math.isfinite(clip_ratio) or not 0 <= clip_ratio < 1:
        raise ValueError(f"clip_ratio must be finite and in [0, 1), got {clip_ratio!r}")
    return clip_ratio


def _quantile(values: Tensor, probability: float) -> float:
    """Return a finite scalar quantile for diagnostics on CPU or CUDA."""

    flattened = values.detach().float().reshape(-1)
    return float(torch.quantile(flattened, probability).item())


def _ppo_metrics_from_terms(
    *,
    old_reduced: Tensor,
    new_reduced: Tensor,
    ratio: Tensor,
    step_mask: Tensor,
    action_dim_mask: Tensor,
    clip_ratio: float,
) -> dict[str, float]:
    delta_log_prob = new_reduced - old_reduced
    clip_fraction = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()
    approx_kl = ((ratio - 1.0) - delta_log_prob).mean()
    event_count = (
        step_mask.to(device=old_reduced.device, dtype=torch.float32).sum(dim=1)
        * action_dim_mask.to(device=old_reduced.device, dtype=torch.float32).sum()
    )
    metrics = {
        # These three are the compact, backwards-compatible training metrics.
        "ratio_mean": float(ratio.detach().mean().item()),
        "clip_fraction": float(clip_fraction.detach().item()),
        "approx_kl": float(approx_kl.detach().item()),
        # The remaining values are deliberately diagnostic and are namespaced
        # under info/ by the trainers before being sent to trackers.
        "ratio_q05": _quantile(ratio, 0.05),
        "ratio_q50": _quantile(ratio, 0.50),
        "ratio_q95": _quantile(ratio, 0.95),
        "ratio_max": float(ratio.detach().max().item()),
        "delta_logprob_mean": float(delta_log_prob.detach().mean().item()),
        "delta_logprob_std": float(delta_log_prob.detach().std(unbiased=False).item()),
        "delta_logprob_q05": _quantile(delta_log_prob, 0.05),
        "delta_logprob_q50": _quantile(delta_log_prob, 0.50),
        "delta_logprob_q95": _quantile(delta_log_prob, 0.95),
        "delta_logprob_min": float(delta_log_prob.detach().min().item()),
        "delta_logprob_max": float(delta_log_prob.detach().max().item()),
        "ratio_capped_fraction": float(
            (delta_log_prob.detach() > _LOG_RATIO_EXACT_LIMIT).float().mean().item()
        ),
        "old_joint_logprob_mean": float(old_reduced.detach().mean().item()),
        "new_joint_logprob_mean": float(new_reduced.detach().mean().item()),
        "event_count_mean": float(event_count.mean().item()),
        "event_count_min": float(event_count.min().item()),
        "event_count_max": float(event_count.max().item()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"PPO metrics contain non-finite values: {metrics}")
    return metrics


def denoising_ppo_metrics(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    *,
    step_mask: Tensor,
    action_dim_mask: Tensor,
    clip_ratio: float,
) -> dict[str, float]:
    """Summarize joint diffusion ratios without constructing an autograd loss.

    This is used by trainers to aggregate diagnostics over all denoising steps
    and minibatches while keeping the objective's event reduction in one place.
    """

    _validate_log_prob("new_log_prob", new_log_prob)
    _validate_log_prob("old_log_prob", old_log_prob)
    if new_log_prob.shape != old_log_prob.shape:
        raise ValueError(
            "new_log_prob and old_log_prob must have matching shapes, "
            f"got {tuple(new_log_prob.shape)} and {tuple(old_log_prob.shape)}"
        )
    clip_ratio = _validate_clip_ratio(clip_ratio)
    old_reduced = reduce_event_log_prob(
        old_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask
    ).detach()
    new_reduced = reduce_event_log_prob(
        new_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask
    ).detach()
    return denoising_ppo_reduced_metrics(
        new_reduced,
        old_reduced,
        step_mask=step_mask,
        action_dim_mask=action_dim_mask,
        clip_ratio=clip_ratio,
    )


def denoising_ppo_reduced_metrics(
    new_reduced: Tensor,
    old_reduced: Tensor,
    *,
    step_mask: Tensor,
    action_dim_mask: Tensor,
    clip_ratio: float,
) -> dict[str, float]:
    """Summarize already joint-reduced log-probabilities.

    ``new_reduced`` and ``old_reduced`` have shape ``[denoise, batch]``.  The
    helper lets trainers keep diagnostics compact instead of transferring the
    full action-chunk event tensor between devices.
    """

    for name, value in (("new_reduced", new_reduced), ("old_reduced", old_reduced)):
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must have shape [denoise,batch]")
        if not value.is_floating_point() or value.numel() == 0:
            raise ValueError(f"{name} must be a nonempty floating-point tensor")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if new_reduced.shape != old_reduced.shape:
        raise ValueError(
            "new_reduced and old_reduced must have matching shapes, "
            f"got {tuple(new_reduced.shape)} and {tuple(old_reduced.shape)}"
        )
    if not isinstance(step_mask, Tensor) or step_mask.dtype != torch.bool or step_mask.ndim != 2:
        raise ValueError("step_mask must be a boolean tensor with shape [batch,action_steps]")
    if step_mask.shape[0] != new_reduced.shape[1]:
        raise ValueError(
            "step_mask batch dimension must match reduced log probabilities, "
            f"got {step_mask.shape[0]} and {new_reduced.shape[1]}"
        )
    if not step_mask.any(dim=1).all().item():
        raise ValueError("step_mask must select at least one action step per batch item")
    if (
        not isinstance(action_dim_mask, Tensor)
        or action_dim_mask.dtype != torch.bool
        or action_dim_mask.ndim != 1
        or not action_dim_mask.any().item()
    ):
        raise ValueError("action_dim_mask must be a nonempty boolean vector")
    new_reduced = new_reduced.detach()
    old_reduced = old_reduced.detach()
    clip_ratio = _validate_clip_ratio(clip_ratio)
    log_ratio = new_reduced - old_reduced
    if not torch.isfinite(log_ratio).all().item():
        raise ValueError("PPO log ratio contains non-finite values")
    ratio = _stable_probability_ratio(log_ratio)
    if not torch.isfinite(ratio).all().item():
        raise ValueError("PPO ratio contains non-finite values")
    return _ppo_metrics_from_terms(
        old_reduced=old_reduced,
        new_reduced=new_reduced,
        ratio=ratio,
        step_mask=step_mask,
        action_dim_mask=action_dim_mask,
        clip_ratio=clip_ratio,
    )


def denoising_ppo_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    *,
    step_mask: Tensor,
    action_dim_mask: Tensor,
    clip_ratio: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute one clipped PPO ratio per denoising sub-policy and sample."""

    _validate_log_prob("new_log_prob", new_log_prob)
    _validate_log_prob("old_log_prob", old_log_prob)
    if new_log_prob.shape != old_log_prob.shape:
        raise ValueError(
            "new_log_prob and old_log_prob must have matching shapes, "
            f"got {tuple(new_log_prob.shape)} and {tuple(old_log_prob.shape)}"
        )
    clip_ratio = _validate_clip_ratio(clip_ratio)
    if not isinstance(advantage, Tensor) or not advantage.is_floating_point():
        raise ValueError("advantage must be a floating-point torch.Tensor")
    batch_size = new_log_prob.shape[1]
    if advantage.numel() != batch_size:
        raise ValueError(
            f"advantage must contain one value per batch item ({batch_size}), "
            f"got shape {tuple(advantage.shape)}"
        )
    if not torch.isfinite(advantage).all().item():
        raise ValueError("advantage must contain only finite values")

    old_reduced = reduce_event_log_prob(
        old_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask
    ).detach()
    new_reduced = reduce_event_log_prob(new_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask)
    log_ratio = new_reduced - old_reduced
    if not torch.isfinite(log_ratio).all().item():
        raise ValueError("PPO log ratio contains non-finite values")
    ratio = _stable_probability_ratio(log_ratio)
    if not torch.isfinite(ratio).all().item():
        raise ValueError("PPO ratio contains non-finite values")

    expanded_advantage = advantage.to(device=ratio.device, dtype=ratio.dtype).reshape(1, -1)
    expanded_advantage = expanded_advantage.expand_as(ratio)
    unclipped = ratio * expanded_advantage
    clipped_log_ratio = log_ratio.clamp(min=math.log1p(-clip_ratio), max=math.log1p(clip_ratio))
    clipped = torch.exp(clipped_log_ratio) * expanded_advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("PPO loss is non-finite")

    metrics = _ppo_metrics_from_terms(
        old_reduced=old_reduced,
        new_reduced=new_reduced,
        ratio=ratio,
        step_mask=step_mask,
        action_dim_mask=action_dim_mask,
        clip_ratio=clip_ratio,
    )
    return loss, metrics
