from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples.franka_duo_real_recorder.manual_record_franka_duo import (
    annotate_episode_reward,
    parse_reward,
    prompt_reward,
)
from examples.franka_duo_real_recorder.record_franka_duo import CameraSpec, build_features


def test_manual_features_use_canonical_transition_fields() -> None:
    camera = CameraSpec(
        key="head",
        image_topic="/head/image",
        width=8,
        height=6,
        fps=30,
        depth_topic="/head/depth",
        camera_info_topic="/head/info",
        record_depth=True,
    )
    features = build_features({"head": camera}, include_transition_fields=True)
    assert features["next.reward"] == {"dtype": "float32", "shape": (1,), "names": None}
    assert features["next.done"] == {"dtype": "bool", "shape": (1,), "names": None}
    assert features["next.truncated"] == {"dtype": "bool", "shape": (1,), "names": None}


def test_parse_reward_rejects_non_finite_values() -> None:
    assert parse_reward(" -1.25 ") == -1.25
    for value in ("nan", "inf", "-inf", "not-a-number"):
        with pytest.raises(ValueError, match="finite"):
            parse_reward(value)


def test_prompt_reward_retries_after_invalid_input() -> None:
    values = iter(["bad", "2.5"])
    messages: list[str] = []
    assert prompt_reward(lambda _prompt: next(values), messages.append) == 2.5
    assert len(messages) == 1
    assert "finite" in messages[0]


def test_annotate_episode_reward_is_terminal_only() -> None:
    buffer = {
        "size": 3,
        "next.reward": [np.zeros((1,), dtype=np.float32) for _ in range(3)],
        "next.done": [np.zeros((1,), dtype=np.bool_) for _ in range(3)],
        "next.truncated": [np.zeros((1,), dtype=np.bool_) for _ in range(3)],
    }
    dataset = SimpleNamespace(writer=SimpleNamespace(episode_buffer=buffer))

    annotate_episode_reward(dataset, 4.25)

    np.testing.assert_array_equal(np.asarray(buffer["next.reward"])[:, 0], [0.0, 0.0, 4.25])
    np.testing.assert_array_equal(np.asarray(buffer["next.done"])[:, 0], [False, False, True])
    np.testing.assert_array_equal(np.asarray(buffer["next.truncated"])[:, 0], [False, False, False])
    assert all(value.dtype == np.dtype("float32") for value in buffer["next.reward"])
    assert all(value.dtype == np.dtype("bool") for value in buffer["next.done"])


def test_annotate_episode_reward_can_mark_truncated() -> None:
    buffer = {
        "size": 1,
        "next.reward": [np.zeros((1,), dtype=np.float32)],
        "next.done": [np.zeros((1,), dtype=np.bool_)],
        "next.truncated": [np.zeros((1,), dtype=np.bool_)],
    }
    annotate_episode_reward(
        SimpleNamespace(writer=SimpleNamespace(episode_buffer=buffer)), 0.0, truncated=True
    )
    assert bool(buffer["next.done"][0][0])
    assert bool(buffer["next.truncated"][0][0])


def test_annotate_episode_reward_requires_pending_transition_fields() -> None:
    dataset = SimpleNamespace(writer=SimpleNamespace(episode_buffer={"size": 1}))
    with pytest.raises(ValueError, match="transition annotation"):
        annotate_episode_reward(dataset, 1.0)
