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

from RL.algorithms.dynamics import PolicyPromotionGate, StateDynamicsEnsemble
from RL.policy.observation_encoder import StateFeatureEncoder
from RL.types import DecisionBatch, ObservationBatch


def _active_mask() -> torch.Tensor:
    mask = torch.zeros(14, dtype=torch.bool)
    mask[[0, 1, 2, 12, 13]] = True
    return mask


@pytest.fixture
def decision_batch() -> DecisionBatch:
    torch.manual_seed(9)
    batch_size = 6
    state = torch.randn(batch_size, 2, 3)
    next_state = state + 0.05 * torch.randn_like(state)
    return DecisionBatch(
        observation=ObservationBatch({"observation.state": state}),
        next_observation=ObservationBatch(
            {"observation.state": next_state}
        ),
        action=torch.randn(batch_size, 4, 14),
        action_valid=torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, False, False, False],
                [True, True, False, False],
            ]
        ),
        reward=torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [0.0]]),
        done=torch.tensor([[False], [True], [False], [True], [False], [False]]),
        discount=torch.full((batch_size, 1), 0.9),
    )


def _make_dynamics() -> StateDynamicsEnsemble:
    return StateDynamicsEnsemble(
        feature_encoder=StateFeatureEncoder(
            state_dim=3, n_obs_steps=2, hidden_dims=(16,), output_dim=8
        ),
        state_dim=3,
        n_obs_steps=2,
        action_dim=14,
        chunk_size=4,
        active_action_mask=_active_mask(),
        hidden_dims=(16, 16),
        ensemble_size=3,
        learning_rate=1e-3,
        bootstrap_seed=17,
    )


def _assert_nested_exact(expected: object, actual: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_exact(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_exact(expected_item, actual_item)
    else:
        assert actual == expected


def test_dynamics_predicts_next_state_history_reward_and_done(
    decision_batch: DecisionBatch,
) -> None:
    dynamics = _make_dynamics()
    prediction = dynamics.predict(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
    )
    next_state = dynamics.next_state_history(
        decision_batch.observation, prediction
    )

    assert prediction.state_delta.shape == (3, 6, 2, 3)
    assert prediction.reward_logit.shape == (3, 6, 1)
    assert prediction.done_logit.shape == (3, 6, 1)
    assert next_state.shape == (3, 6, 2, 3)
    assert torch.isfinite(next_state).all()


def test_dynamics_update_is_finite_and_keeps_encoder_frozen(
    decision_batch: DecisionBatch,
) -> None:
    dynamics = _make_dynamics()
    before = [
        parameter.detach().clone() for parameter in dynamics.feature_encoder.parameters()
    ]

    metrics = dynamics.update(decision_batch)
    validation = dynamics.validation_loss(decision_batch)
    disagreement = dynamics.disagreement(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
    )

    assert all(math.isfinite(value) for value in metrics.values())
    assert math.isfinite(validation)
    assert disagreement.shape == (6, 1)
    assert torch.isfinite(disagreement).all()
    assert all(not parameter.requires_grad for parameter in dynamics.feature_encoder.parameters())
    dynamics.train()
    assert not dynamics.feature_encoder.training
    for previous, current in zip(
        before, dynamics.feature_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(previous, current)


def test_dynamics_feature_encoder_can_follow_trained_iql_encoder() -> None:
    source = StateFeatureEncoder(
        state_dim=3, n_obs_steps=2, hidden_dims=(16,), output_dim=8
    )
    dynamics = StateDynamicsEnsemble(
        feature_encoder=StateFeatureEncoder(
            state_dim=3, n_obs_steps=2, hidden_dims=(16,), output_dim=8
        ),
        state_dim=3,
        n_obs_steps=2,
        action_dim=14,
        chunk_size=4,
        active_action_mask=_active_mask(),
        hidden_dims=(16,),
        ensemble_size=2,
    )
    with torch.no_grad():
        next(source.parameters()).add_(0.25)
    dynamics.sync_feature_encoder(source)
    for expected, actual in zip(
        source.parameters(), dynamics.feature_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)
    assert all(not parameter.requires_grad for parameter in dynamics.feature_encoder.parameters())


def test_dynamics_update_is_transactional_when_later_member_loss_overflows(
    decision_batch: DecisionBatch,
) -> None:
    dynamics = _make_dynamics()
    dynamics.update(decision_batch)
    final_layer = dynamics.members[-1].network[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    state_values = dynamics.n_obs_steps * dynamics.state_dim
    finite_overflow_bias = 2.0 * math.sqrt(torch.finfo(final_layer.bias.dtype).max)
    with torch.no_grad():
        final_layer.bias[:state_values].fill_(finite_overflow_bias)
    assert torch.isfinite(final_layer.bias).all()
    prediction = dynamics.predict(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
    )
    assert torch.isfinite(prediction.state_delta).all()

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in dynamics.members.named_parameters()
    }
    optimizers_before = [
        copy.deepcopy(optimizer.state_dict()) for optimizer in dynamics.optimizers
    ]

    with pytest.raises(ValueError, match=r"member 2.*state_loss"):
        dynamics.update(decision_batch)

    for name, parameter in dynamics.members.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
    for optimizer_before, optimizer in zip(
        optimizers_before, dynamics.optimizers, strict=True
    ):
        _assert_nested_exact(optimizer_before, optimizer.state_dict())


def test_dynamics_update_is_transactional_when_later_member_gradient_overflows(
    decision_batch: DecisionBatch,
) -> None:
    dynamics = _make_dynamics()
    dynamics.update(decision_batch)
    assert math.isfinite(dynamics.validation_loss(decision_batch))
    final_layer = dynamics.members[-1].network[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    hook = final_layer.weight.register_hook(
        lambda gradient: torch.full_like(gradient, torch.inf)
    )
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in dynamics.members.named_parameters()
    }
    optimizers_before = [
        copy.deepcopy(optimizer.state_dict()) for optimizer in dynamics.optimizers
    ]

    try:
        with pytest.raises(ValueError, match=r"member 2.*gradient.*finite"):
            dynamics.update(decision_batch)
    finally:
        hook.remove()

    for name, parameter in dynamics.members.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
        assert parameter.grad is None
    for optimizer_before, optimizer in zip(
        optimizers_before, dynamics.optimizers, strict=True
    ):
        _assert_nested_exact(optimizer_before, optimizer.state_dict())


def test_dynamics_bootstrap_indices_are_deterministic_and_member_specific() -> None:
    dynamics = _make_dynamics()

    first = dynamics.bootstrap_indices(8, member_index=0)
    repeated = dynamics.bootstrap_indices(8, member_index=0)
    other = dynamics.bootstrap_indices(8, member_index=1)

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, other)


def test_dynamics_ignores_inactive_and_padded_action_values(
    decision_batch: DecisionBatch,
) -> None:
    dynamics = _make_dynamics()
    baseline = dynamics.predict(
        decision_batch.observation,
        decision_batch.action,
        decision_batch.action_valid,
    )
    changed = decision_batch.action.clone()
    changed[..., 3:12] = 1e6
    changed[~decision_batch.action_valid] = -1e6
    updated = dynamics.predict(
        decision_batch.observation, changed, decision_batch.action_valid
    )

    torch.testing.assert_close(updated.state_delta, baseline.state_delta)
    torch.testing.assert_close(updated.reward_logit, baseline.reward_logit)
    torch.testing.assert_close(updated.done_logit, baseline.done_logit)


def test_promotion_gate_requires_valid_dynamics_and_margin() -> None:
    gate = PolicyPromotionGate(relative_margin=0.05, max_validation_loss=0.1)
    invalid_model = gate.decide(
        candidate_return=0.55,
        behavior_return=0.5,
        critic_return=0.4,
        dynamics_validation_loss=0.2,
    )
    promoted = gate.decide(
        candidate_return=0.55,
        behavior_return=0.5,
        critic_return=0.4,
        dynamics_validation_loss=0.05,
    )

    assert not invalid_model.promote
    assert invalid_model.reason == "dynamics_validation_loss"
    assert promoted.promote
    assert promoted.reference_return == pytest.approx(0.5)
    assert promoted.required_return == pytest.approx(0.525)


def test_promotion_gate_uses_stronger_critic_reference_and_rejects_nonfinite() -> None:
    gate = PolicyPromotionGate(relative_margin=0.05, max_validation_loss=0.1)

    rejected = gate.decide(
        candidate_return=0.55,
        behavior_return=0.5,
        critic_return=0.6,
        dynamics_validation_loss=0.05,
    )
    assert not rejected.promote
    assert rejected.reference_return == pytest.approx(0.6)
    with pytest.raises(ValueError, match="finite"):
        gate.decide(
            candidate_return=torch.nan,
            behavior_return=0.5,
            critic_return=0.4,
            dynamics_validation_loss=0.05,
        )


def test_promotion_gate_handles_zero_baseline_and_validation_boundary() -> None:
    gate = PolicyPromotionGate(
        relative_margin=0.05, max_validation_loss=0.1, epsilon=1e-6
    )

    equal = gate.decide(
        candidate_return=0.0,
        behavior_return=0.0,
        critic_return=0.0,
        dynamics_validation_loss=0.1,
    )
    improved = gate.decide(
        candidate_return=1e-6,
        behavior_return=0.0,
        critic_return=0.0,
        dynamics_validation_loss=0.1,
    )

    assert not equal.promote
    assert improved.promote


def test_promotion_gate_rejects_exact_required_return_boundary() -> None:
    gate = PolicyPromotionGate(relative_margin=0.25, max_validation_loss=0.1)

    boundary = gate.decide(
        candidate_return=5.0,
        behavior_return=4.0,
        critic_return=3.0,
        dynamics_validation_loss=0.1,
    )

    assert boundary.required_return == pytest.approx(5.0)
    assert not boundary.promote
    assert boundary.reason == "insufficient_return"


def test_dynamics_rejects_nonbinary_sparse_reward(decision_batch: DecisionBatch) -> None:
    dynamics = _make_dynamics()
    invalid = DecisionBatch(
        observation=decision_batch.observation,
        next_observation=decision_batch.next_observation,
        action=decision_batch.action,
        action_valid=decision_batch.action_valid,
        reward=torch.full_like(decision_batch.reward, 0.5),
        done=decision_batch.done,
        discount=decision_batch.discount,
    )

    with pytest.raises(ValueError, match="binary"):
        dynamics.update(invalid)


def test_promotion_gate_rejects_negative_validation_loss() -> None:
    gate = PolicyPromotionGate(relative_margin=0.05, max_validation_loss=0.1)

    with pytest.raises(ValueError, match="nonnegative"):
        gate.decide(
            candidate_return=0.6,
            behavior_return=0.5,
            critic_return=0.4,
            dynamics_validation_loss=-0.1,
        )


def test_paper_amq_gate_uses_behavior_reference_and_inclusive_margin() -> None:
    gate = PolicyPromotionGate(
        relative_margin=0.05,
        max_validation_loss=1.0,
        use_critic_reference=False,
        inclusive_margin=True,
    )
    decision = gate.decide(
        candidate_return=0.525,
        behavior_return=0.5,
        critic_return=100.0,
        dynamics_validation_loss=0.1,
    )
    assert decision.promote
    assert decision.reference_return == pytest.approx(0.5)


def test_paper_amq_gate_zero_baseline_has_zero_required_margin() -> None:
    gate = PolicyPromotionGate(
        relative_margin=0.05,
        max_validation_loss=1.0,
        use_critic_reference=False,
        inclusive_margin=True,
    )
    decision = gate.decide(
        candidate_return=0.0,
        behavior_return=0.0,
        critic_return=100.0,
        dynamics_validation_loss=0.1,
    )
    assert decision.required_return == pytest.approx(0.0)
    assert decision.promote
