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

from dataclasses import FrozenInstanceError

import pytest
import torch

from RL.algorithms.gae import GAEResult, compute_gae, compute_vector_gae


def _valid_inputs() -> dict[str, torch.Tensor]:
    shape = (3, 2, 1)
    return {
        "values": torch.zeros(shape),
        "next_values": torch.zeros(shape),
        "reward": torch.ones(shape),
        "discount": torch.full(shape, 0.9),
        "done": torch.zeros(shape, dtype=torch.bool),
    }


def _canonical_name(name: str) -> str:
    return {"values": "value", "next_values": "next_value"}.get(name, name)


def test_gae_matches_hand_calculated_reverse_time_recurrence() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).view(3, 1, 1)
    next_values = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64).view(3, 1, 1)
    reward = torch.tensor([0.5, 1.0, -0.5], dtype=torch.float64).view(3, 1, 1)
    discount = torch.full((3, 1, 1), 0.9, dtype=torch.float64)
    done = torch.zeros((3, 1, 1), dtype=torch.bool)

    result = compute_gae(
        values=values,
        next_values=next_values,
        reward=reward,
        discount=discount,
        done=done,
        gae_lambda=0.8,
    )

    expected_advantage = torch.tensor([2.57584, 1.772, 0.1], dtype=torch.float64).view(3, 1, 1)
    assert isinstance(result, GAEResult)
    torch.testing.assert_close(result.advantage, expected_advantage, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(result.returns, expected_advantage + values, rtol=0.0, atol=1e-12)


def test_gae_keeps_world_recursions_independent() -> None:
    inputs = _valid_inputs()
    inputs["reward"][:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    baseline = compute_gae(**inputs, gae_lambda=0.75)

    changed_inputs = {name: value.clone() for name, value in inputs.items()}
    changed_inputs["reward"][:, 1, 0] = torch.tensor([100.0, -50.0, 25.0])
    changed_inputs["values"][:, 1, 0] = torch.tensor([-30.0, 20.0, 10.0])
    changed = compute_gae(**changed_inputs, gae_lambda=0.75)

    torch.testing.assert_close(changed.advantage[:, 0], baseline.advantage[:, 0])
    assert not torch.equal(changed.advantage[:, 1], baseline.advantage[:, 1])


def test_gae_supports_single_time_step_and_single_world() -> None:
    values = torch.tensor([[[2.0]]])
    result = compute_gae(
        values=values,
        next_values=torch.tensor([[[100.0]]]),
        reward=torch.tensor([[[5.0]]]),
        discount=torch.tensor([[[0.95]]]),
        done=torch.tensor([[[True]]]),
        gae_lambda=0.0,
    )

    torch.testing.assert_close(result.advantage, torch.tensor([[[3.0]]]))
    torch.testing.assert_close(result.returns, torch.tensor([[[5.0]]]))


def test_compute_vector_gae_exposes_time_major_contract() -> None:
    inputs = _valid_inputs()

    result = compute_vector_gae(
        reward=inputs["reward"],
        value=inputs["values"],
        next_value=inputs["next_values"],
        done=inputs["done"],
        discount=inputs["discount"],
        gae_lambda=0.95,
    )

    assert result.advantage.shape == (3, 2, 1)


def test_gae_rejects_discount_outside_unit_interval() -> None:
    inputs = _valid_inputs()
    inputs["discount"][0, 0, 0] = 1.01

    with pytest.raises(ValueError, match="discount"):
        compute_gae(**inputs, gae_lambda=0.95)


def test_gae_uses_per_step_discounts_in_delta_and_recursion() -> None:
    result = compute_gae(
        values=torch.zeros(3, 1, 1),
        next_values=torch.tensor([10.0, 20.0, 30.0]).view(3, 1, 1),
        reward=torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1),
        discount=torch.tensor([0.5, 0.0, 0.25]).view(3, 1, 1),
        done=torch.zeros(3, 1, 1, dtype=torch.bool),
        gae_lambda=0.5,
    )

    torch.testing.assert_close(result.advantage, torch.tensor([6.5, 2.0, 10.5]).view(3, 1, 1))


def test_done_cuts_bootstrap_and_future_advantage_recursion() -> None:
    result = compute_gae(
        values=torch.zeros(3, 1, 1),
        next_values=torch.tensor([10.0, 20.0, 30.0]).view(3, 1, 1),
        reward=torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1),
        discount=torch.full((3, 1, 1), 0.9),
        done=torch.tensor([False, True, False]).view(3, 1, 1),
        gae_lambda=1.0,
    )

    torch.testing.assert_close(result.advantage, torch.tensor([11.8, 2.0, 30.0]).view(3, 1, 1))


def test_gae_result_is_frozen() -> None:
    result = compute_gae(**_valid_inputs(), gae_lambda=0.95)

    with pytest.raises(FrozenInstanceError):
        result.advantage = torch.zeros_like(result.advantage)


@pytest.mark.parametrize("name", ["values", "next_values", "reward", "discount"])
def test_gae_rejects_non_floating_numeric_inputs(name: str) -> None:
    inputs = _valid_inputs()
    inputs[name] = inputs[name].to(dtype=torch.int64)

    with pytest.raises(ValueError, match=_canonical_name(name)):
        compute_gae(**inputs, gae_lambda=0.95)


def test_gae_rejects_non_boolean_done() -> None:
    inputs = _valid_inputs()
    inputs["done"] = inputs["done"].float()

    with pytest.raises(ValueError, match="done"):
        compute_gae(**inputs, gae_lambda=0.95)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("values", torch.zeros(3, 2)),
        ("values", torch.zeros(3, 2, 2)),
        ("values", torch.zeros(0, 2, 1)),
        ("next_values", torch.zeros(2, 3, 1)),
        ("reward", torch.zeros(3, 1, 1)),
        ("discount", torch.zeros(4, 2, 1)),
        ("done", torch.zeros(3, 3, 1, dtype=torch.bool)),
    ],
)
def test_gae_rejects_malformed_or_mismatched_shapes(name: str, invalid: torch.Tensor) -> None:
    inputs = _valid_inputs()
    inputs[name] = invalid

    with pytest.raises(ValueError, match=_canonical_name(name)):
        compute_gae(**inputs, gae_lambda=0.95)


@pytest.mark.parametrize("name", ["values", "next_values", "reward", "discount"])
@pytest.mark.parametrize("nonfinite", [torch.nan, torch.inf, -torch.inf])
def test_gae_rejects_nonfinite_tensor_values(name: str, nonfinite: float) -> None:
    inputs = _valid_inputs()
    inputs[name][0, 0, 0] = nonfinite

    with pytest.raises(ValueError, match=_canonical_name(name)):
        compute_gae(**inputs, gae_lambda=0.95)


@pytest.mark.parametrize(
    "gae_lambda",
    [-0.01, 1.01, torch.nan, torch.inf, -torch.inf, True, "0.95", torch.tensor(0.95)],
)
def test_gae_rejects_invalid_lambda(gae_lambda: object) -> None:
    with pytest.raises(ValueError, match="gae_lambda"):
        compute_gae(**_valid_inputs(), gae_lambda=gae_lambda)  # type: ignore[arg-type]
