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

import pytest
import torch

from RL.algorithms.ppo import (
    denoising_ppo_loss,
    denoising_ppo_metrics,
    denoising_ppo_reduced_metrics,
    reduce_event_log_prob,
)


def test_masked_log_probability_excludes_padding_and_constant_dimensions() -> None:
    log_prob = torch.ones(3, 2, 4, 14)
    log_prob[:, :, :, 3:12] = 100.0
    log_prob[:, 0, 2:] = 100.0
    log_prob[:, 1, 3:] = 100.0
    step_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    dim_mask = torch.zeros(14, dtype=torch.bool)
    dim_mask[[0, 1, 2, 12, 13]] = True

    reduced = reduce_event_log_prob(
        log_prob, step_mask=step_mask, action_dim_mask=dim_mask
    )

    assert reduced.shape == (3, 2)
    assert reduced[:, 0].tolist() == [10.0, 10.0, 10.0]
    assert reduced[:, 1].tolist() == [15.0, 15.0, 15.0]


def test_denoising_ppo_unit_ratio_and_metrics_are_finite() -> None:
    old_log_prob = torch.randn(4, 3, 2, 5)
    advantage = torch.tensor([1.0, -0.5, 0.25])

    loss, metrics = denoising_ppo_loss(
        old_log_prob.clone().requires_grad_(),
        old_log_prob,
        advantage,
        step_mask=torch.ones(3, 2, dtype=torch.bool),
        action_dim_mask=torch.ones(5, dtype=torch.bool),
        clip_ratio=0.2,
    )

    assert loss.item() == pytest.approx(-advantage.mean().item())
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["clip_fraction"] == pytest.approx(0.0)
    assert metrics["approx_kl"] == pytest.approx(0.0)
    assert metrics["ratio_capped_fraction"] == pytest.approx(0.0)
    for name in (
        "ratio_q05",
        "ratio_q50",
        "ratio_q95",
        "ratio_max",
        "delta_logprob_mean",
        "delta_logprob_q05",
        "delta_logprob_q50",
        "delta_logprob_q95",
        "delta_logprob_max",
        "old_joint_logprob_mean",
        "new_joint_logprob_mean",
    ):
        assert name in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_denoising_ppo_reports_joint_chunk_ratio_quantiles() -> None:
    # Each sample has four executable events.  The ratio must use the
    # complete T_a x D joint log-probability before exponentiation.
    old_log_prob = torch.zeros(1, 4, 2, 2)
    deltas = torch.tensor([0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(4.0)), -torch.log(torch.tensor(2.0))])
    new_log_prob = deltas.reshape(1, 4, 1, 1).expand_as(old_log_prob) / 4.0

    _loss, metrics = denoising_ppo_loss(
        new_log_prob,
        old_log_prob,
        torch.ones(4),
        step_mask=torch.ones(4, 2, dtype=torch.bool),
        action_dim_mask=torch.ones(2, dtype=torch.bool),
        clip_ratio=0.2,
    )

    expected_ratio = deltas.exp()
    expected_delta = deltas
    assert metrics["ratio_mean"] == pytest.approx(expected_ratio.mean().item())
    assert metrics["ratio_q05"] == pytest.approx(torch.quantile(expected_ratio, 0.05).item())
    assert metrics["ratio_q50"] == pytest.approx(torch.quantile(expected_ratio, 0.50).item())
    assert metrics["ratio_q95"] == pytest.approx(torch.quantile(expected_ratio, 0.95).item())
    assert metrics["ratio_max"] == pytest.approx(expected_ratio.max().item())
    assert metrics["delta_logprob_mean"] == pytest.approx(expected_delta.mean().item())
    assert metrics["delta_logprob_q50"] == pytest.approx(torch.quantile(expected_delta, 0.50).item())
    assert metrics["delta_logprob_max"] == pytest.approx(expected_delta.max().item())


def test_denoising_ppo_sums_all_active_chunk_events_before_exp() -> None:
    old_log_prob = torch.zeros(1, 1, 3, 2)
    new_log_prob = torch.full_like(old_log_prob, 0.1)

    _loss, metrics = denoising_ppo_loss(
        new_log_prob,
        old_log_prob,
        torch.ones(1),
        step_mask=torch.ones(1, 3, dtype=torch.bool),
        action_dim_mask=torch.ones(2, dtype=torch.bool),
        clip_ratio=0.2,
    )

    assert metrics["delta_logprob_mean"] == pytest.approx(0.6)
    assert metrics["ratio_mean"] == pytest.approx(torch.exp(torch.tensor(0.6)).item())
    assert metrics["event_count_mean"] == pytest.approx(6.0)
    assert metrics["event_count_min"] == pytest.approx(6.0)
    assert metrics["event_count_max"] == pytest.approx(6.0)


def test_reduced_ppo_diagnostics_match_eventwise_diagnostics() -> None:
    old_log_prob = torch.zeros(2, 3, 2, 2)
    new_log_prob = torch.full_like(old_log_prob, 0.1)
    mask = torch.tensor([[True, True], [True, False], [True, True]])
    dim_mask = torch.ones(2, dtype=torch.bool)

    full = denoising_ppo_metrics(
        new_log_prob,
        old_log_prob,
        step_mask=mask,
        action_dim_mask=dim_mask,
        clip_ratio=0.2,
    )
    reduced = denoising_ppo_reduced_metrics(
        reduce_event_log_prob(new_log_prob, step_mask=mask, action_dim_mask=dim_mask),
        reduce_event_log_prob(old_log_prob, step_mask=mask, action_dim_mask=dim_mask),
        step_mask=mask,
        action_dim_mask=dim_mask,
        clip_ratio=0.2,
    )

    for name in ("ratio_mean", "ratio_q50", "delta_logprob_mean", "event_count_mean"):
        assert reduced[name] == pytest.approx(full[name])


def test_denoising_ppo_clips_positive_advantage_ratio() -> None:
    old_log_prob = torch.zeros(1, 1, 1, 1)
    new_log_prob = torch.full_like(old_log_prob, torch.log(torch.tensor(2.0)))

    loss, metrics = denoising_ppo_loss(
        new_log_prob,
        old_log_prob,
        torch.ones(1),
        step_mask=torch.ones(1, 1, dtype=torch.bool),
        action_dim_mask=torch.ones(1, dtype=torch.bool),
        clip_ratio=0.2,
    )

    assert loss.item() == pytest.approx(-1.2)
    assert metrics["ratio_mean"] == pytest.approx(2.0)
    assert metrics["clip_fraction"] == pytest.approx(1.0)


def test_ppo_rejects_shape_mismatch_and_nonfinite_values() -> None:
    log_prob = torch.zeros(2, 1, 2, 3)

    with pytest.raises(ValueError, match="step_mask"):
        reduce_event_log_prob(
            log_prob,
            step_mask=torch.ones(1, 1, dtype=torch.bool),
            action_dim_mask=torch.ones(3, dtype=torch.bool),
        )
    invalid = log_prob.clone()
    invalid[0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        denoising_ppo_loss(
            invalid,
            log_prob,
            torch.ones(1),
            step_mask=torch.ones(1, 2, dtype=torch.bool),
            action_dim_mask=torch.ones(3, dtype=torch.bool),
            clip_ratio=0.2,
        )


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_extreme_joint_log_ratios_remain_finite(advantage: float) -> None:
    old_log_prob = torch.zeros(1, 1, 32, 5)
    new_log_prob = torch.full_like(old_log_prob, 0.56).requires_grad_()

    loss, metrics = denoising_ppo_loss(
        new_log_prob,
        old_log_prob,
        torch.tensor([advantage]),
        step_mask=torch.ones(1, 32, dtype=torch.bool),
        action_dim_mask=torch.ones(5, dtype=torch.bool),
        clip_ratio=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert new_log_prob.grad is not None
    assert torch.isfinite(new_log_prob.grad).all()
    if advantage < 0:
        assert new_log_prob.grad.abs().sum().item() > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    if advantage > 0:
        assert loss.item() == pytest.approx(-1.2)
