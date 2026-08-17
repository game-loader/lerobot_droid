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

import copy
import math

import pytest
import torch

from RL.algorithms.iql import (
    IQL,
    ActionPacker,
    compute_td_target,
    expectile_loss,
)
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.types import DecisionBatch, ObservationBatch


def _active_mask() -> torch.Tensor:
    mask = torch.zeros(14, dtype=torch.bool)
    mask[[0, 1, 2, 12, 13]] = True
    return mask


@pytest.fixture
def decision_batch() -> DecisionBatch:
    torch.manual_seed(3)
    batch_size = 4
    action = torch.randn(batch_size, 4, 14)
    action_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
            [True, False, False, False],
        ]
    )
    return DecisionBatch(
        observation=ObservationBatch(
            {"observation.state": torch.randn(batch_size, 2, 3)}
        ),
        next_observation=ObservationBatch(
            {"observation.state": torch.randn(batch_size, 2, 3)}
        ),
        action=action,
        action_valid=action_valid,
        reward=torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        done=torch.tensor([[False], [True], [False], [True]]),
        discount=torch.tensor([[0.9], [1.0], [0.81], [1.0]]),
    )


def _make_iql() -> IQL:
    return IQL(
        feature_encoder=StateFeatureEncoder(
            state_dim=3, n_obs_steps=2, hidden_dims=(16,), output_dim=8
        ),
        action_dim=14,
        chunk_size=4,
        active_action_mask=_active_mask(),
        hidden_dims=(16, 16),
        expectile=0.7,
        tau=0.1,
        q_lr=3e-4,
        v_lr=3e-4,
    )


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _assert_parameters_unchanged(
    module: torch.nn.Module, before: dict[str, torch.Tensor]
) -> None:
    assert before.keys() == dict(module.named_parameters()).keys()
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _force_large_output(module: torch.nn.Module) -> None:
    linear = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(1e20)


def test_action_packer_excludes_constant_dimensions_and_padding() -> None:
    packer = ActionPacker(
        action_dim=14, chunk_size=4, active_action_mask=_active_mask()
    )
    action = torch.arange(2 * 4 * 14, dtype=torch.float32).reshape(2, 4, 14)
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )
    baseline = packer(action, valid)
    changed = action.clone()
    changed[..., 3:12] = 1e6
    changed[0, 2:] = -1e6

    torch.testing.assert_close(packer(changed, valid), baseline)
    assert baseline.shape == (2, 24)
    assert baseline[0, :20].reshape(4, 5)[2:].eq(0).all()
    torch.testing.assert_close(baseline[:, -4:], valid.float())


def test_action_packer_rejects_nonprefix_or_empty_valid_masks() -> None:
    packer = ActionPacker(
        action_dim=14, chunk_size=4, active_action_mask=_active_mask()
    )
    action = torch.zeros(1, 4, 14)

    with pytest.raises(ValueError, match="prefix"):
        packer(action, torch.tensor([[True, False, True, False]]))
    with pytest.raises(ValueError, match="at least one"):
        packer(action, torch.zeros(1, 4, dtype=torch.bool))


def test_expectile_loss_uses_asymmetric_weights() -> None:
    residual = torch.tensor([[2.0], [-2.0], [0.0]])
    loss = expectile_loss(residual, expectile=0.7, reduction="none")

    torch.testing.assert_close(loss, torch.tensor([[2.8], [1.2], [0.0]]))


def test_td_target_uses_saved_discount_and_terminal_mask() -> None:
    target = compute_td_target(
        reward=torch.tensor([[1.0], [0.0]]),
        discount=torch.tensor([[0.5], [0.81]]),
        done=torch.tensor([[True], [False]]),
        next_value=torch.tensor([[100.0], [2.0]]),
    )

    torch.testing.assert_close(target, torch.tensor([[1.0], [1.62]]))


def test_iql_update_and_advantage_are_finite(decision_batch: DecisionBatch) -> None:
    iql = _make_iql()
    before = [parameter.detach().clone() for parameter in iql.q1.parameters()]

    metrics = iql.update(decision_batch)
    advantage = iql.advantage(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
    )

    assert advantage.shape == (decision_batch.action.shape[0], 1)
    assert torch.isfinite(advantage).all()
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, iql.q1.parameters(), strict=True)
    )


def test_iql_optimizers_do_not_share_parameters() -> None:
    iql = _make_iql()
    q_parameters = {
        id(parameter)
        for group in iql.q_optimizer.param_groups
        for parameter in group["params"]
    }
    v_parameters = {
        id(parameter)
        for group in iql.v_optimizer.param_groups
        for parameter in group["params"]
    }

    assert q_parameters.isdisjoint(v_parameters)
    assert all(not parameter.requires_grad for parameter in iql.target_q1.parameters())
    assert all(not parameter.requires_grad for parameter in iql.target_encoder.parameters())

    iql.train()
    assert not iql.target_encoder.training
    assert not iql.target_q1.training
    assert not iql.target_q2.training


def test_iql_advantage_batch_one_falls_back_to_raw_value() -> None:
    iql = _make_iql()
    observation = ObservationBatch(
        {"observation.state": torch.zeros(1, 2, 3)}
    )
    action = torch.zeros(1, 4, 14)
    valid = torch.ones(1, 4, dtype=torch.bool)

    normalized = iql.advantage(observation, action, valid, normalize=True)
    raw = iql.advantage(observation, action, valid, normalize=False)

    torch.testing.assert_close(normalized, raw)


def test_iql_polyak_update_moves_target_toward_updated_online(
    decision_batch: DecisionBatch,
) -> None:
    iql = _make_iql()
    target_before = next(iql.target_q1.parameters()).detach().clone()

    iql.update(decision_batch)

    online_after = next(iql.q1.parameters()).detach()
    target_after = next(iql.target_q1.parameters()).detach()
    expected = target_before.lerp(online_after, iql.tau)
    torch.testing.assert_close(target_after, expected)


def test_iql_advantage_ignores_inactive_and_padded_values(
    decision_batch: DecisionBatch,
) -> None:
    iql = _make_iql()
    baseline = iql.advantage(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
        normalize=False,
    )
    changed = decision_batch.action.clone()
    changed[..., 3:12] = 1e6
    changed[~decision_batch.action_valid] = -1e6

    updated = iql.advantage(
        decision_batch.observation,
        changed,
        decision_batch.action_valid,
        normalize=False,
    )

    torch.testing.assert_close(updated, baseline)


def test_iql_nonfinite_value_loss_is_rejected_without_partial_update(
    decision_batch: DecisionBatch,
) -> None:
    iql = _make_iql()
    iql.update(decision_batch)
    _force_large_output(iql.target_q1)
    _force_large_output(iql.target_q2)
    parameters_before = _parameter_snapshot(iql)
    q_optimizer_before = copy.deepcopy(iql.q_optimizer.state_dict())
    v_optimizer_before = copy.deepcopy(iql.v_optimizer.state_dict())

    with pytest.raises(ValueError, match="value_loss.*finite"):
        iql.update(decision_batch)

    _assert_parameters_unchanged(iql, parameters_before)
    _assert_nested_equal(iql.q_optimizer.state_dict(), q_optimizer_before)
    _assert_nested_equal(iql.v_optimizer.state_dict(), v_optimizer_before)


def test_iql_nonfinite_q_loss_is_rejected_without_partial_update(
    decision_batch: DecisionBatch,
) -> None:
    iql = _make_iql()
    iql.update(decision_batch)
    _force_large_output(iql.q1)
    _force_large_output(iql.q2)
    parameters_before = _parameter_snapshot(iql)
    q_optimizer_before = copy.deepcopy(iql.q_optimizer.state_dict())
    v_optimizer_before = copy.deepcopy(iql.v_optimizer.state_dict())

    with pytest.raises(ValueError, match="q_loss.*finite"):
        iql.update(decision_batch)

    _assert_parameters_unchanged(iql, parameters_before)
    _assert_nested_equal(iql.q_optimizer.state_dict(), q_optimizer_before)
    _assert_nested_equal(iql.v_optimizer.state_dict(), v_optimizer_before)
