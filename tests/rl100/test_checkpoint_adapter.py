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

from pathlib import Path

import pytest
import torch

from RL.adapters.checkpoint import CheckpointAdapter
from RL.types import ObservationBatch

REAL_CHECKPOINT = Path(
    "outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model"
)


@pytest.fixture(scope="module")
def checkpoint_adapter() -> CheckpointAdapter:
    if not REAL_CHECKPOINT.is_dir():
        pytest.skip(f"real checkpoint is unavailable: {REAL_CHECKPOINT}")
    return CheckpointAdapter.load(REAL_CHECKPOINT, device="cpu")


def test_checkpoint_adapter_loads_policy_and_processors(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    assert checkpoint_adapter.policy.config.device == "cpu"
    assert not checkpoint_adapter.policy.training
    assert checkpoint_adapter.active_action_mask.dtype == torch.bool
    assert checkpoint_adapter.active_action_mask.shape == (14,)


def test_checkpoint_adapter_normalization_round_trip(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    raw = torch.full((2, 14), 0.01)

    normalized = checkpoint_adapter.normalize_action(raw)
    restored = checkpoint_adapter.unnormalize_action(normalized)

    torch.testing.assert_close(restored, raw, rtol=1e-5, atol=1e-6)


def test_checkpoint_adapter_normalizes_batched_state_history(
    checkpoint_adapter: CheckpointAdapter,
) -> None:
    observation = ObservationBatch(
        {
            "observation.state": torch.zeros(2, 2, 39),
            "observation.images.front": torch.zeros(2, 2, 3, 4, 4, dtype=torch.uint8),
        }
    )

    normalized = checkpoint_adapter.normalize_observation(observation)

    assert normalized.features["observation.state"].shape == (2, 2, 39)
    assert normalized.features["observation.images.front"].dtype == torch.uint8
    torch.testing.assert_close(
        normalized.features["observation.images.front"],
        observation.features["observation.images.front"],
    )
    assert not torch.equal(
        normalized.features["observation.state"], observation.features["observation.state"]
    )


def test_active_action_mask_uses_saved_ranges(checkpoint_adapter: CheckpointAdapter) -> None:
    assert checkpoint_adapter.active_action_indices.tolist() == [0, 1, 2, 12, 13]


def test_processor_fingerprint_is_stable_sha256(checkpoint_adapter: CheckpointAdapter) -> None:
    fingerprint = checkpoint_adapter.processor_fingerprint()
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")
    assert checkpoint_adapter.processor_fingerprint() == fingerprint
