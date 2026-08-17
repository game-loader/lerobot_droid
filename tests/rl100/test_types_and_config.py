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

import dataclasses
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import get_type_hints

import pytest
import torch

from RL.config import RLConfig, TraceConfig
from RL.types import DecisionBatch, DenoisingTrace, ObservationBatch


def _observation(batch_size: int = 2) -> ObservationBatch:
    return ObservationBatch(
        features={
            "observation.state": torch.arange(batch_size * 2 * 39, dtype=torch.float32).reshape(
                batch_size, 2, 39
            ),
            "observation.images.front": torch.zeros(batch_size, 2, 3, 8, 8),
        }
    )


def test_observation_batch_preserves_state_and_optional_images() -> None:
    observation = _observation()

    assert observation.batch_size() == 2
    assert set(observation.features) == {"observation.state", "observation.images.front"}
    assert observation.features["observation.state"].shape == (2, 2, 39)
    assert observation.features["observation.images.front"].shape == (2, 2, 3, 8, 8)

    selected = observation.index_select(torch.tensor([1]))
    assert selected.batch_size() == 1
    torch.testing.assert_close(
        selected.features["observation.state"], observation.features["observation.state"][1:2]
    )
    torch.testing.assert_close(
        selected.features["observation.images.front"],
        observation.features["observation.images.front"][1:2],
    )
    moved = observation.to("cpu")
    assert all(tensor.device == torch.device("cpu") for tensor in moved.features.values())


def test_observation_batch_defensively_copies_feature_mapping() -> None:
    state = torch.zeros(2, 2, 39)
    features = {"observation.state": state}
    observation = ObservationBatch(features)

    features["observation.state"] = torch.ones_like(state)
    features["observation.extra"] = torch.zeros(2, 1)

    assert observation.features["observation.state"] is state
    assert "observation.extra" not in observation.features


def test_observation_batch_feature_mapping_is_read_only() -> None:
    observation = _observation()

    assert isinstance(observation.features, dict)
    assert get_type_hints(ObservationBatch)["features"] == dict[str, torch.Tensor]
    with pytest.raises(TypeError):
        observation.features["observation.extra"] = torch.zeros(2, 1)
    with pytest.raises(TypeError):
        del observation.features["observation.state"]
    with pytest.raises(TypeError):
        observation.features.update({"observation.extra": torch.zeros(2, 1)})
    with pytest.raises(TypeError):
        observation.features.clear()
    with pytest.raises(TypeError):
        observation.features.pop("observation.state")
    with pytest.raises(TypeError):
        observation.features.popitem()
    with pytest.raises(TypeError):
        observation.features.setdefault("observation.extra", torch.zeros(2, 1))
    with pytest.raises(TypeError):
        observation.features.__ior__({"observation.extra": torch.zeros(2, 1)})

    replaced = dataclasses.replace(observation)
    assert isinstance(replaced.features, dict)
    assert set(replaced.features) == set(observation.features)


def test_observation_batch_pickle_round_trip_preserves_features() -> None:
    observation = _observation()

    restored = pickle.loads(pickle.dumps(observation))

    assert isinstance(restored.features, dict)
    assert set(restored.features) == set(observation.features)
    for key, tensor in observation.features.items():
        torch.testing.assert_close(restored.features[key], tensor)
    with pytest.raises(TypeError):
        restored.features["observation.extra"] = torch.zeros(2, 1)


@pytest.mark.parametrize(
    "features, error",
    [
        ({}, r"features.*\{\}"),
        (
            {"observation.state": torch.full((2, 2, 39), torch.nan)},
            r"observation\.state.*nan",
        ),
        (
            {
                "observation.state": torch.zeros(2, 2, 39),
                "observation.images.front": torch.zeros(3, 2, 3, 8, 8),
            },
            "batch size",
        ),
    ],
)
def test_observation_batch_rejects_invalid_shapes(features: dict[str, torch.Tensor], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ObservationBatch(features=features)


def test_decision_batch_validates_shapes_and_dtypes() -> None:
    observation = _observation()
    decision = DecisionBatch(
        observation=observation,
        next_observation=observation,
        action=torch.zeros(2, 32, 14),
        action_valid=torch.ones(2, 32, dtype=torch.bool),
        reward=torch.zeros(2, 1),
        done=torch.zeros(2, 1, dtype=torch.bool),
        discount=torch.full((2, 1), 0.99),
    )

    decision.validate(state_dim=39, action_dim=14, chunk_size=32, n_obs_steps=2)
    moved = decision.to("cpu")
    assert moved.action.device == torch.device("cpu")
    assert all(tensor.device == torch.device("cpu") for tensor in moved.observation.features.values())

    with pytest.raises(ValueError, match="action_valid"):
        DecisionBatch(
            observation=observation,
            next_observation=observation,
            action=torch.zeros(2, 32, 14),
            action_valid=torch.ones(2, 31, dtype=torch.bool),
            reward=torch.zeros(2, 1),
            done=torch.zeros(2, 1, dtype=torch.bool),
            discount=torch.full((2, 1), 0.99),
        )

    with pytest.raises(ValueError, match="done"):
        DecisionBatch(
            observation=observation,
            next_observation=observation,
            action=torch.zeros(2, 32, 14),
            action_valid=torch.ones(2, 32, dtype=torch.bool),
            reward=torch.zeros(2, 1),
            done=torch.zeros(2, 1),
            discount=torch.full((2, 1), 0.99),
        )

    with pytest.raises(ValueError, match="reward"):
        DecisionBatch(
            observation=observation,
            next_observation=observation,
            action=torch.zeros(2, 32, 14),
            action_valid=torch.ones(2, 32, dtype=torch.bool),
            reward=torch.full((2, 1), torch.inf),
            done=torch.zeros(2, 1, dtype=torch.bool),
            discount=torch.full((2, 1), 0.99),
        )

    wrong_history = ObservationBatch({"observation.state": torch.zeros(2, 1, 39)})
    with pytest.raises(ValueError, match="next_observation.state"):
        DecisionBatch(
            observation=observation,
            next_observation=wrong_history,
            action=torch.zeros(2, 32, 14),
            action_valid=torch.ones(2, 32, dtype=torch.bool),
            reward=torch.zeros(2, 1),
            done=torch.zeros(2, 1, dtype=torch.bool),
            discount=torch.full((2, 1), 0.99),
        ).validate(state_dim=39, action_dim=14, chunk_size=32, n_obs_steps=2)


def test_decision_batch_validate_supports_custom_state_key() -> None:
    observation = ObservationBatch(
        {
            "robot.state": torch.zeros(2, 2, 39),
            "observation.images.front": torch.zeros(2, 2, 3, 8, 8, dtype=torch.uint8),
        }
    )
    decision = DecisionBatch(
        observation=observation,
        next_observation=observation,
        action=torch.zeros(2, 32, 14),
        action_valid=torch.ones(2, 32, dtype=torch.bool),
        reward=torch.zeros(2, 1),
        done=torch.zeros(2, 1, dtype=torch.bool),
        discount=torch.full((2, 1), 0.99),
    )

    decision.validate(
        state_dim=39,
        action_dim=14,
        chunk_size=32,
        n_obs_steps=2,
        state_key="robot.state",
    )


def test_decision_batch_validate_reports_missing_custom_state_key() -> None:
    observation = _observation()
    decision = DecisionBatch(
        observation=observation,
        next_observation=observation,
        action=torch.zeros(2, 32, 14),
        action_valid=torch.ones(2, 32, dtype=torch.bool),
        reward=torch.zeros(2, 1),
        done=torch.zeros(2, 1, dtype=torch.bool),
        discount=torch.full((2, 1), 0.99),
    )

    with pytest.raises(ValueError, match=r"state_key='robot\.state'.*actual=<missing>"):
        decision.validate(
            state_dim=39,
            action_dim=14,
            chunk_size=32,
            n_obs_steps=2,
            state_key="robot.state",
        )


def test_decision_batch_validate_rejects_integer_state() -> None:
    observation = ObservationBatch(
        {"observation.state": torch.zeros(2, 2, 39, dtype=torch.int64)}
    )
    decision = DecisionBatch(
        observation=observation,
        next_observation=observation,
        action=torch.zeros(2, 32, 14),
        action_valid=torch.ones(2, 32, dtype=torch.bool),
        reward=torch.zeros(2, 1),
        done=torch.zeros(2, 1, dtype=torch.bool),
        discount=torch.full((2, 1), 0.99),
    )

    with pytest.raises(ValueError, match=r"observation\.state.*floating.*torch\.int64"):
        decision.validate(state_dim=39, action_dim=14, chunk_size=32, n_obs_steps=2)


def test_denoising_trace_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="next_latents"):
        DenoisingTrace(
            latents=torch.zeros(10, 2, 64, 14),
            next_latents=torch.zeros(9, 2, 64, 14),
            timesteps=torch.arange(10),
            old_log_prob=torch.zeros(10, 2, 64, 14),
            final_actions=torch.zeros(2, 64, 14),
        )


def test_trace_config_rejects_non_positive_sigma_min() -> None:
    with pytest.raises(ValueError, match="sigma_min"):
        TraceConfig(sigma_min=0.0)


def test_rl_config_json_round_trip_is_deterministic(tmp_path: Path) -> None:
    config = RLConfig(
        trace=TraceConfig(num_inference_steps=8, eta=0.75, sigma_min=0.01, sigma_max=0.2),
        state_key="observation.state",
        n_obs_steps=2,
        state_dim=39,
        action_dim=14,
        chunk_size=32,
        gamma=0.97,
    )

    payload = config.to_json()
    restored = RLConfig.from_json(payload)
    path = tmp_path / "rl_config.json"
    config.save_json(path)
    loaded = RLConfig.load_json(path)

    assert restored == config
    assert loaded == config
    assert restored.to_json() == payload
    assert loaded.to_json() == payload
    assert path.read_text(encoding="utf-8") == f"{payload}\n"
    assert json.loads(payload)["trace"]["num_inference_steps"] == 8


def test_rl_config_json_rejects_unknown_fields() -> None:
    payload = json.loads(RLConfig().to_json())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="Unknown RLConfig field"):
        RLConfig.from_json(json.dumps(payload))

    nested_payload = json.loads(RLConfig().to_json())
    nested_payload["trace"]["unexpected"] = True

    with pytest.raises(ValueError, match="Unknown TraceConfig field"):
        RLConfig.from_json(json.dumps(nested_payload))


def test_rl_config_rejects_none_trace_with_actual_value() -> None:
    with pytest.raises(ValueError, match=r"trace.*actual=None"):
        RLConfig(trace=None)  # type: ignore[arg-type]


def test_rl_config_json_rejects_non_object_trace_with_actual_value() -> None:
    with pytest.raises(ValueError, match=r"trace.*actual=\[\]"):
        RLConfig.from_json('{"trace": []}')


@pytest.mark.parametrize(
    ("loader", "payload", "error"),
    [
        (RLConfig.from_json, "[]", r"RLConfig.*actual=\[\]"),
        (TraceConfig.from_json, "null", r"TraceConfig.*actual=None"),
        (RLConfig.from_json, "not-json", r"RLConfig.*actual='not-json'"),
    ],
)
def test_config_json_errors_include_the_actual_payload(
    loader: Callable[[str], object], payload: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        loader(payload)
